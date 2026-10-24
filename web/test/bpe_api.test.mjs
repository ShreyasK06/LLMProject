// Calls the Vercel handler with mock req/res against the tiny random-init BPE fixture (tests/make_fixture_model.py).
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { encode, generate, halfToFloat, loadModel, Session } from "../lib/gpt.js";

const fixture = (name) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));
const BPE_DIR = fixture("bpe_model");
const FP16_DIR = fixture("bpe_model_fp16");

// half -> float converter
assert.equal(halfToFloat(0x3c00), 1);
assert.equal(halfToFloat(0xc000), -2);
assert.equal(halfToFloat(0x0000), 0);
assert.ok(Object.is(halfToFloat(0x8000), -0));
assert.equal(halfToFloat(0x7bff), 65504);
assert.equal(halfToFloat(0x0001), 2 ** -24);
assert.equal(halfToFloat(0x3555), 0.333251953125);
assert.equal(halfToFloat(0x7c00), Infinity);
assert.equal(halfToFloat(0xfc00), -Infinity);
assert.ok(Number.isNaN(halfToFloat(0x7e00)));

process.env.MODEL_DIR = BPE_DIR; // before the handler loads (and caches) a model
const { default: handler } = await import("../api/generate.js");

function call(method, body, headers = {}) {
  const res = new EventEmitter();
  Object.assign(res, {
    statusCode: 200, headers: {}, chunks: [],
    setHeader(k, v) { this.headers[k.toLowerCase()] = v; },
    write(c) { this.chunks.push(c); return true; },
    end(c) { if (c) this.chunks.push(c); this.done = true; },
  });
  return handler({ method, body, headers }, res).then(() => res);
}
const text = async (body) => (await call("POST", body)).chunks.join("");

// GET reports the tokenizer and the token count
const info = await call("GET");
assert.equal(info.statusCode, 200);
const meta = JSON.parse(info.chunks.join(""));
assert.equal(meta.ready, true);
assert.equal(meta.tokenizer, "bpe");
assert.equal(meta.vocab, 300);
assert.equal(meta.block, 16);

// generate() yields decoded token text, never more than maxChars characters in total
const model = loadModel(BPE_DIR);
assert.equal(model.tok.vocabSize, 300);
const prompt = "User:\nHello\n\nAssistant:\n";
const { ids } = encode(model, prompt);
assert.deepEqual(ids, model.tok.encode(prompt).ids);
const pieces = [...generate(model, ids, { maxChars: 200, seed: 5 })];
assert.ok(pieces.every((p) => typeof p === "string" && p.length > 0));
assert.ok(pieces.some((p) => p.length > 1), "BPE tokens decode to multi-character pieces");
assert.equal(pieces.join("").length, 200);

// exact maxChars, also when the last token has to be cut
for (const maxChars of [1, 7, 40]) {
  for (const seed of [1, 2, 3]) {
    assert.equal((await text({ prompt, maxChars, seed })).length, maxChars, `maxChars ${maxChars} seed ${seed}`);
  }
}
const full = await text({ prompt, maxChars: 200, seed: 5 });
assert.equal(full, pieces.join(""), "handler streams exactly what generate() yields");
assert.equal(await text({ prompt, maxChars: 7, seed: 5 }), full.slice(0, 7), "truncation keeps the sampled prefix");

// Stop strings are matched on the decoded text stream: output ends right before the first match, which is never sent.
async function checkStop(seed, stop, why) {
  const unstopped = await text({ prompt, maxChars: 400, seed });
  const at = unstopped.indexOf(stop);
  assert.ok(at >= 0, `${why}: stop occurs in the unstopped output`);
  const got = await text({ prompt, maxChars: 400, seed, stop: [stop] });
  assert.equal(got, unstopped.slice(0, at), `${why}: ends right before the stop string`);
  assert.ok(!got.includes(stop), `${why}: stop string is never sent`);
}
// token boundaries (character offsets) of the unstopped output for a seed
const boundaries = (seed) => {
  const out = [0];
  for (const p of generate(model, ids, { maxChars: 400, seed })) out.push(out.at(-1) + p.length);
  return out;
};
const spans = (b, start, len) => b.some((x) => x > start && x < start + len);

// 1) "\n\n" made of more than one token
let found = null;
for (let seed = 0; seed < 300 && found === null; seed++) {
  const out = [...generate(model, ids, { maxChars: 400, seed })].join("");
  const at = out.indexOf("\n\n");
  if (at >= 0 && spans(boundaries(seed), at, 2)) found = seed;
}
assert.ok(found !== null, "some seed produces a \\n\\n that spans tokens");
await checkStop(found, "\n\n", `"\\n\\n" spanning tokens (seed ${found})`);

// 2) a stop completed in the middle of a token: the text before it in that token is still sent
{
  const seed = 11;
  const ps = [...generate(model, ids, { maxChars: 400, seed })];
  const out = ps.join("");
  let off = 0, stop = null;
  for (const p of ps) {
    if (p.length >= 3) {
      const s = p.slice(1, -1);
      if (out.indexOf(s) === off + 1) { stop = s; break; }
    }
    off += p.length;
  }
  assert.ok(stop, "found a stop strictly inside one token");
  await checkStop(seed, stop, `stop ${JSON.stringify(stop)} inside a token`);
}

// 3) a stop that crosses two token boundaries
{
  const seed = 12;
  const ps = [...generate(model, ids, { maxChars: 400, seed })];
  const out = ps.join("");
  let off = ps[0].length, stop = null;
  for (let i = 1; i + 1 < ps.length; i++) {
    const s = ps[i - 1].slice(-1) + ps[i] + ps[i + 1].slice(0, 1);
    if (out.indexOf(s) === off - 1) { stop = s; break; }
    off += ps[i].length;
  }
  assert.ok(stop, "found a stop spanning three tokens");
  await checkStop(seed, stop, `stop ${JSON.stringify(stop)} across three tokens`);
  // the earliest of several stops wins
  const other = out.slice(-3);
  const at = Math.min(...[stop, other].map((s) => out.indexOf(s)));
  assert.equal(await text({ prompt, maxChars: 400, seed, stop: [other, stop] }), out.slice(0, at));
}

// past the context window (block 16 tokens): completes and is deterministic per seed
const long1 = await text({ prompt: "The", maxChars: 300, seed: 9 });
const long2 = await text({ prompt: "The", maxChars: 300, seed: 9 });
assert.equal(long1.length, 300);
assert.equal(long1, long2, "same seed gives the same text");
assert.notEqual(long1, await text({ prompt: "The", maxChars: 300, seed: 10 }));

// prompts: nothing encodable -> 400; mixed -> X-Dropped-Chars; other validation unchanged
assert.equal((await call("POST", { prompt: "é中文" })).statusCode, 400);
const mixed = await call("POST", { prompt: "User:é中文", maxChars: 5 });
assert.equal(mixed.statusCode, 200);
assert.equal(mixed.headers["x-dropped-chars"], "3");
assert.equal((await call("POST", {})).statusCode, 400);
assert.equal((await call("POST", { prompt: "x".repeat(5000) })).statusCode, 400);
assert.equal((await call("PUT", {})).statusCode, 405);
assert.equal((await text({ prompt: "The", maxChars: 99999, topK: 99999 })).length, 1000, "maxChars clamps to 1000");
assert.equal((await text({ prompt: "The" })).length, 300, "default maxChars");

// fp16 weights load as float32 and agree with the float32 export
const m16 = loadModel(FP16_DIR);
assert.ok(m16.w["tok.weight"] instanceof Float32Array);
const s32 = new Session(model), s16 = new Session(m16);
const l32 = s32.prefill(ids), l16 = s16.prefill(ids);
assert.equal(l16.length, 300);
for (let i = 0; i < l32.length; i++) assert.ok(Math.abs(l32[i] - l16[i]) < 1e-2, `logit ${i}`);
process.env.MODEL_DIR = FP16_DIR;
assert.equal(JSON.parse((await call("GET")).chunks.join("")).tokenizer, "bpe");
// Untrained weights give near-uniform logits, so fp16 rounding eventually flips a sample; the logit check above is
// the equivalence test, and the fp16 model must still stream through the handler.
const f1 = await text({ prompt, maxChars: 200, seed: 5 });
assert.equal(f1.length, 200);
assert.equal(f1, await text({ prompt, maxChars: 200, seed: 5 }), "fp16 is deterministic per seed");

// missing tokenizer.json -> 503 that names the tokenizer
const tmp = mkdtempSync(join(tmpdir(), "bpe-model-"));
try {
  cpSync(BPE_DIR, tmp, { recursive: true, filter: (src) => !src.endsWith("tokenizer.json") });
  process.env.MODEL_DIR = tmp;
  const r = await call("GET");
  assert.equal(r.statusCode, 503);
  const err = JSON.parse(r.chunks.join(""));
  assert.equal(err.ready, false);
  assert.match(err.error, /tokenizer/i);
  assert.equal((await call("POST", { prompt })).statusCode, 503);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

// copy of the fixture model in a temp dir, with `edit(dir)` applied
function withModelCopy(edit, fn) {
  const dir = mkdtempSync(join(tmpdir(), "bpe-model-"));
  cpSync(BPE_DIR, dir, { recursive: true });
  edit(dir);
  return Promise.resolve(fn(dir)).finally(() => rmSync(dir, { recursive: true, force: true }));
}

// unparseable tokenizer.json -> loadModel throws "Tokenizer file ... invalid: ...", and the handler's 503 carries it
await withModelCopy((d) => writeFileSync(join(d, "tokenizer.json"), "{ not json"), async (dir) => {
  assert.throws(() => loadModel(dir), /^Error: Tokenizer file tokenizer\.json .*invalid: /);
  process.env.MODEL_DIR = dir;
  const r = await call("GET");
  assert.equal(r.statusCode, 503);
  assert.match(JSON.parse(r.chunks.join("")).error, /^Tokenizer file tokenizer\.json .*invalid: /);
  // The failure is cached for this MODEL_DIR: fixing the file does not trigger a reload of the weights.
  cpSync(join(BPE_DIR, "tokenizer.json"), join(dir, "tokenizer.json"));
  const again = await call("GET");
  assert.equal(again.statusCode, 503, "a tokenizer failure is cached per MODEL_DIR");
  assert.match(JSON.parse(again.chunks.join("")).error, /invalid: /);
});

// tokenizer.json whose vocabulary size differs from the embedding rows in weights.bin -> error naming the tokenizer
await withModelCopy((d) => {
  const t = JSON.parse(readFileSync(join(d, "tokenizer.json"), "utf8"));
  t.merges = t.merges.slice(0, -5); t.vocab = t.vocab.slice(0, -5);
  writeFileSync(join(d, "tokenizer.json"), JSON.stringify(t));
}, async (dir) => {
  assert.throws(() => loadModel(dir), (e) => /tokenizer/i.test(e.message) && /295/.test(e.message) && /300/.test(e.message));
  process.env.MODEL_DIR = dir;
  const r = await call("GET");
  assert.equal(r.statusCode, 503);
  assert.match(JSON.parse(r.chunks.join("")).error, /tokenizer/i);
});

// MODEL_DIR without weights -> 503 too, and that is NOT cached: once weights appear, the next request loads them
await withModelCopy((d) => rmSync(join(d, "weights.bin")), async (dir) => {
  process.env.MODEL_DIR = dir;
  assert.equal((await call("GET")).statusCode, 503);
  cpSync(join(BPE_DIR, "weights.bin"), join(dir, "weights.bin"));
  assert.equal((await call("GET")).statusCode, 200, "a missing weights file is retried");
});
process.env.MODEL_DIR = join(tmpdir(), "no-such-model-dir");
assert.equal((await call("GET")).statusCode, 503);

console.log("all BPE API tests passed");
