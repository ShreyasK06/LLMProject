// Vercel serverless function (Node runtime). POST streams generated text as text/plain; GET reports model status.
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { encode, generate, loadModel, TokenizerError } from "../lib/gpt.js";

const MAX_PROMPT = 4000;
const clamp = (v, lo, hi, d) => (Number.isFinite(+v) ? Math.min(hi, Math.max(lo, +v)) : d);

// MODEL_DIR (if set) is the only directory searched; the cache is keyed on it so a changed value takes effect.
// A tokenizer failure (bad or mismatched tokenizer.json) is also cached per MODEL_DIR for the process lifetime, so
// each request does not re-read the weights; missing weights are not cached, so a later request retries.
let cached = null, cachedKey = null;
const failed = new Map();
function getModel() {
  const key = process.env.MODEL_DIR || "";
  if (cached && cachedKey === key) return cached;
  if (failed.has(key)) throw failed.get(key);
  const here = dirname(fileURLToPath(import.meta.url));
  const dirs = key ? [resolve(key)] : [join(here, "..", "model"), join(process.cwd(), "model")];
  const dir = dirs.find((d) => existsSync(join(d, "weights.bin")));
  if (!dir) {
    throw new Error(`Model weights not found${key ? ` in MODEL_DIR (${key})` : ""}. Run \`python export_weights.py --ckpt <checkpoint>\` and redeploy.`);
  }
  let model;
  try {
    model = loadModel(dir);
  } catch (e) {
    if (e instanceof TokenizerError) failed.set(key, e);
    throw e;
  }
  model.nParams = Object.values(model.w).reduce((n, t) => n + t.length, 0);
  cached = model; cachedKey = key;
  return cached;
}

function sendJson(res, status, obj) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(obj));
}

// The frontend on GitHub Pages calls this function from another origin, so allow it (and local dev) explicitly.
// Override with a comma-separated ALLOWED_ORIGINS environment variable on the host.
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "https://shreyask06.github.io").split(",").map((s) => s.trim());
function applyCors(req, res) {
  const origin = req.headers?.origin;
  if (!origin || !(ALLOWED_ORIGINS.includes(origin) || /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin))) return;
  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Access-Control-Expose-Headers", "X-Dropped-Chars");
  res.setHeader("Vary", "Origin");
}

export default async function handler(req, res) {
  applyCors(req, res);
  if (req.method === "OPTIONS") { // CORS preflight for the JSON POST: answer before loading the model
    res.statusCode = 204;
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Max-Age", "86400");
    return res.end();
  }

  let model;
  try {
    model = getModel();
  } catch (e) {
    return sendJson(res, 503, { ready: false, error: e.message });
  }

  if (req.method === "GET") {
    return sendJson(res, 200, {
      ready: true, params: model.nParams, step: model.step, block: model.block,
      layers: model.nLayer, vocab: model.vocabSize, tokenizer: model.tok ? "bpe" : "char",
    });
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return sendJson(res, 405, { error: "Use POST." });
  }

  let body = req.body;
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch { body = null; }
  }
  if (!body || typeof body.prompt !== "string") return sendJson(res, 400, { error: "Body must be JSON with a string `prompt`." });
  if (body.prompt.length > MAX_PROMPT) return sendJson(res, 400, { error: `Prompt too long (max ${MAX_PROMPT} characters).` });

  const { ids, dropped } = encode(model, body.prompt);
  if (!ids.length) {
    return sendJson(res, 400, { error: "None of those characters are in the model's vocabulary (letters, digits, basic punctuation)." });
  }

  const opts = {
    maxChars: Math.round(clamp(body.maxChars, 1, 1000, 300)),
    temperature: clamp(body.temperature, 0.1, 1.5, 0.8),
    topK: Math.round(clamp(body.topK, 1, model.vocabSize, 40)),
    seed: Number.isInteger(body.seed) ? body.seed : undefined,
  };

  res.statusCode = 200;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-store, no-transform");
  res.setHeader("X-Accel-Buffering", "no");
  res.setHeader("X-Dropped-Chars", String(dropped));

  // Optional stop strings end the reply early (e.g. "\n\n" = end of a speech). They are matched on the decoded text,
  // so they may span tokens, and one token's text may contain (part of) one. Output that could still turn into a stop
  // string is held back, so the stop string itself is never sent; text before the first match always is.
  const stops = Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === "string" && s).slice(0, 4) : [];
  const couldStart = (s) => stops.some((t) => t.startsWith(s));

  let closed = false;
  res.on("close", () => { closed = true; });
  let held = "";
  for (const piece of generate(model, ids, opts)) {
    if (closed) return;
    held += piece;
    const at = Math.min(...stops.map((t) => held.indexOf(t)).filter((i) => i >= 0));
    if (at !== Infinity) {
      if (at) res.write(held.slice(0, at));
      return res.end();
    }
    let k = 0;
    while (k < held.length && !couldStart(held.slice(k))) k++;
    if (k) { res.write(held.slice(0, k)); held = held.slice(k); }
  }
  if (held) res.write(held);
  res.end();
}
