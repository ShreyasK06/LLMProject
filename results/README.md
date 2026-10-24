# Inference benchmark results

- `bench_before.json`: the original shipped JavaScript inference code (commit `eca0611`, `web/lib/gpt.js`) on the deployed 10.9M-parameter character model in `web/model`.
- `bench_after.json`: the current, optimized `web/lib/gpt.js` on the same model, run back to back with the baseline on the same machine.
- Protocol (`web/bench/bench.mjs`): 3 fixed chat-framed prompts, `maxChars` 200, seed 1, temperature 0.8, top-k 40; 1 warm-up run then 3 timed runs per prompt. `chars_per_sec` and `ttfc_ms` (time from calling `generate`, prompt prefill included, to the first character) are medians over the 9 timed runs. `load_ms` is one `loadModel` call.
- Reproduce: `cd web && node bench/bench.mjs --out ../results/bench_after.json`. To time older code, extract it (`git show eca0611:web/lib/gpt.js > /tmp/old/gpt.js`) and pass `--lib /tmp/old/gpt.js`.
- The numbers depend on the machine, Node version and current load (the CPU, Node version and core count are recorded in each file). Compare only files produced back to back on the same machine; run-to-run noise on a laptop can be 10% or more.
- `web/test/speed_parity.test.mjs` checks that the optimized code gives the same greedy outputs as the original (fixture `web/test/fixtures/greedy_before.json`).
