// Pure-JS inference for the char-level GPT (mirrors train.py: pre-LN blocks, causal attention, GELU MLP, tied head).
// Decoding uses a KV cache. Past `block` tokens the oldest half is dropped and the rest re-prefilled, so positions
// stay within the range the model was trained on (this differs from Python's sliding window beyond `block` only).
import { readFileSync } from "node:fs";

export function loadModel(dir) {
  const meta = JSON.parse(readFileSync(`${dir}/model.json`, "utf-8"));
  const buf = readFileSync(`${dir}/weights.bin`);
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const w = {};
  for (const t of meta.tensors) w[t.name] = all.subarray(t.offset, t.offset + t.length);
  const { block, n_layer, n_head, n_embd } = meta.config;
  const chars = meta.chars;
  return {
    block, nLayer: n_layer, nHead: n_head, nEmbd: n_embd, chars, step: meta.step, w,
    stoi: new Map(chars.map((c, i) => [c, i])),
  };
}

// Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7); torch's nn.GELU uses the exact erf form.
function erf(x) {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

// out[o] = b[o] + sum_i W[o*in + i] * x[i]
function linear(x, W, b, outDim, inDim, out) {
  for (let o = 0; o < outDim; o++) {
    let s = b ? b[o] : 0;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
}

function layerNorm(x, g, b, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
}

export class Session {
  constructor(model) {
    this.m = model;
    const { nLayer, block, nEmbd } = model;
    this.k = Array.from({ length: nLayer }, () => new Float32Array(block * nEmbd));
    this.v = Array.from({ length: nLayer }, () => new Float32Array(block * nEmbd));
    this.len = 0;
    this.tokens = [];
    this.logits = null;
    const E = nEmbd;
    this.x = new Float32Array(E); this.h = new Float32Array(E); this.qkv = new Float32Array(3 * E);
    this.att = new Float32Array(E); this.proj = new Float32Array(E); this.mlp = new Float32Array(4 * E);
    this.scores = new Float32Array(block);
  }

  // Feed one token at position this.len; returns logits over the vocab.
  step(tok) {
    const { w, nLayer, nHead, nEmbd: E, chars } = this.m;
    const hd = E / nHead, pos = this.len, x = this.x, h = this.h;
    const te = w["tok.weight"], pe = w["pos.weight"];
    for (let i = 0; i < E; i++) x[i] = te[tok * E + i] + pe[pos * E + i];

    for (let l = 0; l < nLayer; l++) {
      const p = `blocks.${l}.`;
      layerNorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"], h);
      linear(h, w[p + "qkv.weight"], w[p + "qkv.bias"], 3 * E, E, this.qkv);
      this.k[l].set(this.qkv.subarray(E, 2 * E), pos * E);
      this.v[l].set(this.qkv.subarray(2 * E, 3 * E), pos * E);
      const K = this.k[l], V = this.v[l], sc = this.scores, scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < nHead; hh++) {
        const o = hh * hd;
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          for (let i = 0; i < hd; i++) s += this.qkv[o + i] * K[t * E + o + i];
          s *= scale; sc[t] = s; if (s > max) max = s;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { sc[t] = Math.exp(sc[t] - max); sum += sc[t]; }
        for (let i = 0; i < hd; i++) {
          let a = 0;
          for (let t = 0; t <= pos; t++) a += sc[t] * V[t * E + o + i];
          this.att[o + i] = a / sum;
        }
      }
      linear(this.att, w[p + "proj.weight"], w[p + "proj.bias"], E, E, this.proj);
      for (let i = 0; i < E; i++) x[i] += this.proj[i];
      layerNorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"], h);
      linear(h, w[p + "mlp.0.weight"], w[p + "mlp.0.bias"], 4 * E, E, this.mlp);
      for (let i = 0; i < 4 * E; i++) this.mlp[i] = 0.5 * this.mlp[i] * (1 + erf(this.mlp[i] * Math.SQRT1_2));
      linear(this.mlp, w[p + "mlp.2.weight"], w[p + "mlp.2.bias"], E, 4 * E, this.proj);
      for (let i = 0; i < E; i++) x[i] += this.proj[i];
    }
    layerNorm(x, w["ln_f.weight"], w["ln_f.bias"], h);
    const V_ = chars.length, logits = new Float32Array(V_);
    linear(h, te, null, V_, E, logits); // tied head
    this.len++;
    this.tokens.push(tok);
    this.logits = logits;
    return logits;
  }

  // Reset the cache and feed `tokens` from position 0.
  prefill(tokens) {
    this.len = 0; this.tokens = [];
    for (const t of tokens) this.step(t);
    return this.logits;
  }

  push(tok) {
    if (this.len >= this.m.block) this.prefill(this.tokens.slice(-(this.m.block >> 1)));
    return this.step(tok);
  }
}

export function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sample(logits, temperature, topK, rand) {
  const n = logits.length, k = Math.max(1, Math.min(topK, n));
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  const mx = logits[idx[0]];
  const p = idx.map((i) => Math.exp((logits[i] - mx) / temperature));
  const sum = p.reduce((a, b) => a + b, 0);
  let r = rand() * sum;
  for (let j = 0; j < k; j++) { r -= p[j]; if (r <= 0) return idx[j]; }
  return idx[k - 1];
}

// Yields generated characters. `prompt` is any string; unknown characters are dropped by the caller via encode().
export function* generate(model, ids, { maxChars = 300, temperature = 0.8, topK = 40, seed } = {}) {
  const rand = mulberry32(seed ?? Math.floor(Math.random() * 2 ** 31));
  const s = new Session(model);
  let logits = s.prefill(ids.slice(-model.block));
  for (let i = 0; i < maxChars; i++) {
    const t = sample(logits, temperature, topK, rand);
    yield model.chars[t];
    logits = s.push(t);
  }
}

export function encode(model, text) {
  const ids = [];
  let dropped = 0;
  for (const c of text) { const i = model.stoi.get(c); if (i === undefined) dropped++; else ids.push(i); }
  return { ids, dropped };
}
