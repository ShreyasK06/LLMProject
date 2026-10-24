# Results

Files that back the numbers in the [main README](../README.md). Training logs are in [`../logs/`](../logs/).

## Training runs

- `<name>_results.json`: written by `train.py` (`--results_dir`, default `results/`) for each run: config, parameter count, best and last step, and train/val/test loss in nats per token (train: 200 random batches; val and test: full sweeps), perplexity. Runs from after the BPE work also record `bits_per_char`, token and character counts per split, and `unknown_chars`.
  - `exp_a_baseline`, `exp_b_reg`, `exp_c_small`: Shakespeare (`run_experiments.sh`).
  - `dolly_small`, `dolly_large`: Dolly 15k chat, character level (`run_dolly.sh`); `dolly_large` is the deployed model.
  - `bpe_*` (Phase 2, not yet run): BPE chat models (`run_bpe.sh`, see [the GPU runbook](../docs/GPU_RUNBOOK.md)).
- `generate_*.txt`: sample generations from `generate.py` (the current version samples with temperature 0.8, top-k 40, seed 0). `generate_output.txt` is the overfit first Shakespeare run, `generate_output_final.txt` is `exp_c_small`, and `generate_dolly_*.txt` are the chat models.

## Inference benchmark

- `bench_before.json`: the original shipped JavaScript inference code (commit `eca0611`, `web/lib/gpt.js`) on the deployed 10.9M-parameter character model in `web/model`.
- `bench_after.json`: the current, optimized `web/lib/gpt.js` on the same model, run back to back with the baseline on the same machine.
- Protocol (`web/bench/bench.mjs`): 3 fixed chat-framed prompts, `maxChars` 200, seed 1, temperature 0.8, top-k 40; 1 warm-up run then 3 timed runs per prompt. `chars_per_sec` and `ttfc_ms` (time from calling `generate`, prompt prefill included, to the first character) are medians over the 9 timed runs. `load_ms` is one `loadModel` call.
- Reproduce: `cd web && node bench/bench.mjs --out ../results/bench_after.json`. To time older code, extract it (`git show eca0611:web/lib/gpt.js > /tmp/old/gpt.js`) and pass `--lib /tmp/old/gpt.js`.
- The numbers depend on the machine, Node version and current load (the CPU, Node version and core count are recorded in each file). Compare only files produced back to back on the same machine; run-to-run noise on a laptop can be 10% or more.
- `web/test/speed_parity.test.mjs` checks that the optimized code gives the same greedy outputs as the original (fixture `web/test/fixtures/greedy_before.json`).
