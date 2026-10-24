// Pins model behavior across speed work: greedy outputs on the deployed model must match the ones saved from the
// original (pre-optimization) code exactly, and the first-step logits must agree within 1e-3.
// Regenerate the fixture only from the original code: node bench/bench.mjs --lib <old gpt.js> --greedy-out ...
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { encode, generate, loadModel, Session } from "../lib/gpt.js";

const ref = JSON.parse(readFileSync(new URL("./fixtures/greedy_before.json", import.meta.url), "utf-8"));
const model = loadModel(fileURLToPath(new URL("../model", import.meta.url)));
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
