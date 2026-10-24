// Pins model behavior across speed work: greedy outputs on the deployed model must match the ones saved from the
// original (pre-optimization) code exactly, and the first-step logits must agree within 1e-3.
// Regenerate the fixture only from the original code: node bench/bench.mjs --lib <old gpt.js> --greedy-out ...
// The fixture's `model` key identifies the model it was captured from (parameter count, layers, width, step,
// tokenizer, and a hash of model.json). If web/model holds a different model (for example after swapping in a BPE
// model), the outputs cannot match, so the test prints "speed_parity: skipped, fixture is for a different model" and
// exits 0. For the model the fixture was captured from, any change in output still fails.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { encode, generate, loadModel, Session } from "../lib/gpt.js";

const ref = JSON.parse(readFileSync(new URL("./fixtures/greedy_before.json", import.meta.url), "utf-8"));
const MODEL_DIR = fileURLToPath(new URL("../model", import.meta.url));
const model = loadModel(MODEL_DIR);

function modelIdentity(m, modelJsonBytes) {
  return {
    params: Object.values(m.w).reduce((n, t) => n + t.length, 0),
    layers: m.nLayer,
    width: m.nEmbd,
    step: m.step ?? null,
    tokenizer: m.tok ? "bpe" : null,
    model_json_sha256: createHash("sha256").update(modelJsonBytes).digest("hex").slice(0, 16),
  };
}

assert.ok(ref.model, "greedy_before.json must record the model it was captured from (`model` key)");
const have = modelIdentity(model, readFileSync(new URL("../model/model.json", import.meta.url)));
const mismatch = Object.keys(ref.model).filter((k) => ref.model[k] !== have[k]);
if (mismatch.length) {
  const diff = mismatch.map((k) => `${k}: fixture ${JSON.stringify(ref.model[k])}, loaded ${JSON.stringify(have[k])}`).join("; ");
  console.log(`speed_parity: skipped, fixture is for a different model (${diff})`);
  process.exit(0);
}

const { maxChars, temperature, topK, seed } = ref;

for (const c of ref.cases) {
  const { ids } = encode(model, c.prompt);
  const logits = new Session(model).prefill(ids.slice(-model.block));
  let argmax = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
  assert.equal(argmax, c.argmax, `argmax for ${JSON.stringify(c.prompt)}`);
  const diffs = c.first_logits.map((v, i) => Math.abs(logits[i] - v)).concat(Math.abs(logits[argmax] - c.argmax_logit));
  const maxDiff = Math.max(...diffs);
  assert.ok(maxDiff < 1e-3, `first-step logits differ by ${maxDiff} for ${JSON.stringify(c.prompt)}`);

  const out = Array.from(generate(model, ids, { maxChars, temperature, topK, seed })).join("");
  assert.equal(out, c.output, `greedy output for ${JSON.stringify(c.prompt)}`);
}
console.log(`speed_parity: ${ref.cases.length} prompts OK`);
