// Inference speed benchmark for the pure-JS GPT.
//   node bench/bench.mjs [--lib PATH] [--model DIR] [--out FILE] [--greedy-out FILE]
// --lib points at any gpt.js-compatible module (default ../lib/gpt.js), so an older version of the code can be timed
// against the same model. --greedy-out also writes the greedy outputs and first-step logits used by the parity test.
// Works with modules whose generate() yields single characters or multi-character strings: it counts characters.
import { writeFileSync } from "node:fs";
import os from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const PROMPTS = [
  "User:\nWhat is the capital of France?\n\nAssistant:\n",
  "User:\nGive me some ideas for a birthday party.\n\nAssistant:\n",
  "User:\nExplain what a computer is.\n\nAssistant:\n",
];
const RUN = { maxChars: 200, seed: 1, temperature: 0.8, topK: 40 };
const WARMUP = 1, RUNS = 3;

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;

const libPath = resolve(arg("--lib", resolve(here, "../lib/gpt.js")));
const modelDir = resolve(arg("--model", resolve(here, "../model")));
const out = arg("--out", null);
const greedyOut = arg("--greedy-out", null);

const lib = await import(pathToFileURL(libPath).href);
const t0 = performance.now();
const model = lib.loadModel(modelDir);
const loadMs = performance.now() - t0;
const params = Object.values(model.w).reduce((n, t) => n + t.length, 0);

function timedRun(ids) {
  const start = performance.now();
  let ttfc = null, chars = 0;
  for (const piece of lib.generate(model, ids, RUN)) {
    if (ttfc === null) ttfc = performance.now() - start;
    chars += Array.from(piece).length;
  }
  const total = performance.now() - start;
  return { ttfc, cps: chars / (total / 1000) };
}

const cps = [], ttfc = [];
for (const prompt of PROMPTS) {
  const { ids } = lib.encode(model, prompt);
  for (let r = 0; r < WARMUP; r++) timedRun(ids);
  for (let r = 0; r < RUNS; r++) { const res = timedRun(ids); cps.push(res.cps); ttfc.push(res.ttfc); }
}

const result = {
  node: process.version,
  cpu: os.cpus()[0]?.model?.trim() ?? "unknown",
  cpu_count: os.cpus().length,
  platform: `${process.platform}-${process.arch}`,
  model: { layers: model.nLayer, width: model.nEmbd, heads: model.nHead, block: model.block, vocab: model.vocabSize ?? model.chars.length },
  params,
  runs: { prompts: PROMPTS.length, warmup_per_prompt: WARMUP, timed_per_prompt: RUNS, ...RUN },
  chars_per_sec: round(median(cps)),
  ttfc_ms: round(median(ttfc)),
  load_ms: round(loadMs),
};
console.log(JSON.stringify(result, null, 2));
if (out) writeFileSync(resolve(out), JSON.stringify(result, null, 2) + "\n");

if (greedyOut) {
  const cases = PROMPTS.map((prompt) => {
    const { ids } = lib.encode(model, prompt);
    const s = new lib.Session(model);
    const logits = s.prefill(ids.slice(-model.block));
    let argmax = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
    const output = Array.from(lib.generate(model, ids, { maxChars: 120, temperature: 0.8, topK: 1, seed: 1 })).join("");
    return { prompt, output, first_logits: Array.from(logits.subarray(0, 8)), argmax, argmax_logit: logits[argmax] };
  });
  writeFileSync(resolve(greedyOut), JSON.stringify({ maxChars: 120, temperature: 0.8, topK: 1, seed: 1, cases }, null, 2) + "\n");
}
