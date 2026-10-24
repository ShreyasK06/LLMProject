# BPE Tokenizer, Faster Inference and Repo Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the character tokenizer with a from-scratch BPE tokenizer (Python for training, JavaScript for serving), retrain the chat model on the same Dolly data, make web inference measurably faster, and turn the repo into a portfolio-ready project.

**Architecture:** A character-level BPE module (`bpe.py`) learns merges on the Dolly *train* text only and is saved as `tokenizer.json`. `train.py` gains a `--tokenizer` flag and reports bits per character for every model so char and BPE runs compare fairly. `export_weights.py` writes the tokenizer next to the weights, and `web/lib/gpt.js` loads it, decodes tokens to text, and streams text capped at `maxChars` characters. Training on the GPU machine is a hand-off between Phase 1 (everything buildable here) and Phase 2 (swap in the trained model, fill in the numbers).

**Tech Stack:** Python 3 + PyTorch (CPU-only here), pytest; Node 22, React 18, Vite 5, Vercel serverless; GitHub Pages workflow already exists.

**Spec:** `docs/superpowers/specs/2026-10-23-bpe-and-polish-design.md`

## Global Constraints

- The model is trained from scratch on the same Dolly data: no pretrained weights, no extra data.
- Tokenizer: character-level BPE over the characters that occur in the data (95 for Dolly), learned on the training split only, vocabulary about 2,000 (builder picks the exact size; record it).
- Merges never cross a pre-split piece boundary, and the splitter is hand-written on ASCII classes (no unicode-category regex), so Python and JavaScript agree exactly. Chat markers `User:` / `Assistant:` and the blank-line stop are ordinary text, not special tokens.
- Bits per character = total negative log likelihood on a split divided by the number of raw characters in that split's text. Report train, validation and test, say which split each number is, select on validation, touch test once. Also report the held-out figure next to any headline number.
- Same data and same split as the character runs: `prepare_dolly.py` output, sequential 80/10/10 cut at the character offsets `int(0.8*n)` and `int(0.9*n)`, then each part tokenized separately.
- The existing API contract keeps working: POST `{prompt, maxChars, temperature, topK, seed, stop}` streams `text/plain`, `maxChars` caps *characters* (clamped 1..1000, default 300), `X-Dropped-Chars` header, 400 on empty/oversized/unencodable prompt, 405 for other methods, CORS rules unchanged, GET returns `{ready, params, step, block, layers, vocab}`.
- The deployed model keeps working until the trained BPE model is validated; the swap of `web/model/` is one change.
- No GPU here: local PyTorch is CPU-only, so smoke-test on tiny configs; real training is run by Shreyas on their GPU machine.
- Windows 11, Git Bash/PowerShell. No Docker, no `gh`. Use a venv.
- Front end: no absolute asset paths (use `import.meta.env.BASE_URL`); do not run a production build over a running dev server that shares the output directory.
- Git: work on branch `bpe-polish`. No Claude attribution in any commit message or PR. Commit and push only after Shreyas has said to; until then leave changes staged or uncommitted. Push goes to `origin` (their own repo).
- Speed claims only from a measured before/after benchmark that states hardware and Node version.
- Nothing is deleted before checking whether the README or a script references it.

## Review Focus

1. Prompt of only characters outside the alphabet (for example `é中文`) must give 400, and mixed prompts must report the dropped count (Task 5).
2. Empty text and text of only newlines must encode and decode without error (Task 1, Task 2).
3. A token longer than the characters still allowed must be cut so exactly `maxChars` characters are sent (Task 5).
4. A stop string that is split across two tokens (for example `"\n\n"` as `"\n"` then `"\n"`) must still stop the reply and never be sent (Task 5).
5. Generation past the context window must keep working when a window holds fewer characters than expected, using a tiny-`block` fixture model (Task 5).
6. `tokenizer.json` missing or corrupt must return the existing 503 with a readable message, not crash the function (Task 5).

---

### Task 1: Python BPE tokenizer

**Files:**
- Create: `bpe.py`, `train_tokenizer.py`, `tests/test_bpe.py`, `tests/fixtures/bpe_parity.json` and `web/test/fixtures/tokenizer.json` (both generated and committed by `test_writes_parity_fixture`; the second is the tokenizer used for the parity ids, vocab about 300)
- Create: `requirements-dev.txt` (`pytest`)

**Interfaces:**
- Produces:
  - `pretokenize(text: str) -> list[str]`: pieces are `[space]?` plus a run of `[A-Za-z0-9]`, or `[space]?` plus a run of other non-space, non-newline characters, or a run of newlines, or a lone run of spaces. Concatenating the pieces gives back `text`.
  - `class BPE` with `vocab: list[str]`, `merges: list[tuple[int, int]]`, `BPE.train(text: str, vocab_size: int) -> BPE`, `encode(text: str) -> list[int]`, `decode(ids: list[int]) -> str`, `save(path: str) -> None`, `BPE.load(path: str) -> BPE`, `char_len(id: int) -> int`, `unknown_chars(text: str) -> int` (characters not in the base alphabet).
  - `tokenizer.json` format: `{"version": 1, "alphabet": [str], "merges": [[int, int]], "vocab": [str]}`; ids `0..len(alphabet)-1` are the alphabet in sorted order; each merge adds id `len(alphabet) + i`.
  - `encode` drops characters outside the alphabet (same behavior as the char model).
  - `train_tokenizer.py --data data/dolly_chat.txt --vocab_size 2000 --out tokenizer.json` trains on the first 80% of characters only.
  - `tests/fixtures/bpe_parity.json`: list of `{"text": str, "ids": [int]}` for fixed strings (Task 2 reads it).

- [ ] **Step 1: Write the failing tests** in `tests/test_bpe.py`: `test_pretokenize_roundtrip` (pieces join to the input for ASCII samples including `"a  b\n\nc"`), `test_encode_decode_roundtrip` (`decode(encode(s)) == s` for strings inside the alphabet), `test_empty_and_newlines` (`encode("") == []`, `decode(encode("\n\n\n")) == "\n\n\n"`), `test_unknown_chars_dropped` (`encode("aé中b") == encode("ab")`, `unknown_chars("aé中b") == 2`), `test_merge_order_deterministic` (training twice on the same text gives identical `merges`; ties broken by lowest pair), `test_never_merges_across_pieces` (no token in `vocab` contains both a space and a letter except a leading single space followed by alphanumerics, and none contains a newline together with another class), `test_save_load_identical` (loaded tokenizer encodes identically), `test_compression` (on a small repeated-English text, tokens per character < 0.5 at vocab 300), and `test_writes_parity_fixture` (writes `tests/fixtures/bpe_parity.json` for these strings: `""`, `"\n"`, `"\n\n"`, `"User:\nHello there\n\nAssistant:\n"`, `"  leading and  double spaces"`, `"Numbers 12345 and punctuation!?;,"`, a 2,000-character repeated paragraph, and a string containing `é` which must encode as if the `é` were absent).

- [ ] **Step 2: Run to verify failure:** `python -m pytest tests/test_bpe.py -v`. Expected: collection error, `bpe` not importable.
- [ ] **Step 3: Implement `pretokenize`, `BPE.train` (pair counting over pre-split piece frequencies, greedy most-frequent pair, ties broken by lowest `(a, b)`), `encode` (apply merges in learned order per piece, with a cache per piece), `decode`, `save`, `load`, `char_len`, `unknown_chars` in `bpe.py`** and the `train_tokenizer.py` CLI.
- [ ] **Step 4: Run to verify pass:** `python -m pytest tests/test_bpe.py -v`. Expected: all pass.
- [ ] **Step 5: Commit** `bpe.py train_tokenizer.py tests requirements-dev.txt` with message `Add a from-scratch BPE tokenizer with tests`.

### Task 2: JavaScript BPE tokenizer with parity test

**Files:**
- Create: `web/lib/bpe.js`, `web/test/bpe.test.mjs`
- Modify: `web/package.json` (`test` script runs `node test/bpe.test.mjs` first)

**Interfaces:**
- Consumes: `tokenizer.json` format and `tests/fixtures/bpe_parity.json` from Task 1.
- Produces: `loadTokenizer(path: string) -> Tokenizer` and `Tokenizer` with `encode(text: string) -> { ids: number[], dropped: number }`, `decode(ids: number[]) -> string`, `pieceOf(id: number) -> string` (the decoded text of one token), `vocabSize: number`, `charLen(id): number`.

- [ ] **Step 1: Write the failing test** `web/test/bpe.test.mjs`: loads a tokenizer built by the Python test run (committed small fixture `web/test/fixtures/tokenizer.json`, produced by `python train_tokenizer.py` on the sample text and copied there by the Task 1 test), then for every entry in `../tests/fixtures/bpe_parity.json` asserts `encode(text).ids` deep-equals `ids` and `decode(ids)` equals the text with out-of-alphabet characters removed; also asserts `encode("é中文").dropped === 3` and `encode("").ids.length === 0`.
- [ ] **Step 2: Run:** `cd web && node test/bpe.test.mjs`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `web/lib/bpe.js`** with the same splitter and merge application as `bpe.py`, matching exactly.
- [ ] **Step 4: Run:** `cd web && node test/bpe.test.mjs`. Expected: PASS.
- [ ] **Step 5: Commit** `web/lib/bpe.js web/test web/package.json`: `Port the BPE tokenizer to JavaScript with a parity test`.

### Task 3: Bits-per-character evaluation and BPE support in training

**Files:**
- Modify: `train.py` (loading, splitting, evaluation, checkpoint, results JSON), `generate.py`
- Create: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `BPE.load`, `BPE.char_len`, `BPE.encode` from Task 1.
- Produces:
  - `train.py --tokenizer PATH` (optional). Without it, behavior is unchanged (character ids).
  - `load_data(path, tokenizer=None) -> (vocab: list[str], tok_chars: torch.LongTensor)`: cut the text at the 80%/90% character offsets, tokenize each part separately; `tok_chars[i]` is the character length of token `i` (all ones for character models). Module-level `splits` stays as the dict of id tensors.
  - `estimate_loss` additionally returns bits per character per split: `bpc = (mean nats per token * tokens) / (chars in targets * ln 2)`, where chars are summed with `tok_chars` over the target ids.
  - Results JSON keys added: `"tokenizer": "char" | <path>`, `"bits_per_char": {"train","val","test"}`, `"tokens": {"train","val","test"}`, `"chars": {"train","val","test"}`. Existing keys stay.
  - Checkpoint dict stores `"tokenizer": <inline tokenizer json or None>` besides `"chars"`.
  - `generate.py` decodes with the tokenizer when the checkpoint has one and prints only the assistant reply as before.

- [ ] **Step 1: Write failing tests** `tests/test_train_smoke.py`: `test_char_bpc_equals_loss_over_ln2` (tiny char-level run of 10 steps on a 5,000-character text: `bits_per_char["val"] == pytest.approx(val_loss / math.log(2))`); `test_bpe_run_writes_results` (train a vocab-300 tokenizer on the same text, run 10 steps with `--tokenizer`, assert all new JSON keys exist, `chars["train"]` equals the length of the first 80% of the text, and `bits_per_char["test"] > 0`); `test_same_split_points` (character and BPE runs report the same `chars` per split).
- [ ] **Step 2: Run:** `python -m pytest tests/test_train_smoke.py -v`. Expected: FAIL (unknown argument `--tokenizer`).
- [ ] **Step 3: Implement the changes in `train.py` and `generate.py` per the Interfaces**, keeping the CPU default config small enough that the smoke tests finish in under a minute.
- [ ] **Step 4: Run:** `python -m pytest tests -v`. Expected: all pass.
- [ ] **Step 5: Commit** `train.py generate.py tests`: `Report bits per character and add BPE support to training`.

### Task 4: Export the tokenizer and optional 16-bit weights

**Files:**
- Modify: `export_weights.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: checkpoint format from Task 3.
- Produces: `export_weights.py --ckpt PATH [--out web/model] [--fp16]` writes `weights.bin`, `model.json`, and (for BPE checkpoints) `tokenizer.json`. `model.json` gains `"tokenizer": "tokenizer.json" | null` and `"dtype": "float32" | "float16"`; `chars` stays for character models.

- [ ] **Step 1: Write failing tests** `tests/test_export.py`: export a tiny BPE checkpoint (made in-test with a 1-layer, width-16 model) and assert `tokenizer.json` exists, `model.json["tokenizer"] == "tokenizer.json"`, `weights.bin` size equals 4 bytes per parameter (2 with `--fp16`) excluding the tied head, and that fp16 round-trips within `1e-2` absolute error.
- [ ] **Step 2: Run:** `python -m pytest tests/test_export.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** in `export_weights.py`.
- [ ] **Step 4: Run:** `python -m pytest tests -v`. Expected: pass.
- [ ] **Step 5: Commit** `export_weights.py tests`: `Export the tokenizer and optional fp16 weights`.

### Task 5: Serve BPE models from the web handler

**Files:**
- Modify: `web/lib/gpt.js` (`loadModel`, `encode`, `generate`, `push`), `web/api/generate.js` (model loading, status, stop and cap handling), `web/src/App.jsx` (vocabulary label, top-k slider max)
- Create: `tests/make_fixture_model.py`, `web/test/fixtures/bpe_model/` (tiny committed model: 2 layers, width 32, `block` 16, about 300 tokens), `web/test/bpe_api.test.mjs`
- Modify: `web/package.json` (`test` also runs `bpe_api.test.mjs`)

**Interfaces:**
- Consumes: `loadTokenizer`, `Tokenizer` from Task 2; export format from Task 4.
- Produces:
  - `loadModel(dir: string) -> Model` now sets `model.tok: Tokenizer | null`; when `model.json["tokenizer"]` is null it keeps the character fallback so the current weights still work.
  - `encode(model, text) -> { ids, dropped }` uses `model.tok` when present.
  - `generate(model, ids, opts) -> Generator<string>` yields the decoded text of each token, truncated so the total yielded length never exceeds `opts.maxChars`.
  - `MODEL_DIR` environment variable overrides the model directory search in `getModel()`.
  - GET adds `tokenizer: "bpe" | "char"`; `vocab` is the token count. The UI shows `"{vocab}-token BPE vocabulary"` for BPE and the existing text for characters, and the top-k slider max is `vocab`.
  - Stop strings are matched against the decoded text stream (they may span tokens).

- [ ] **Step 1: Write failing tests** `web/test/bpe_api.test.mjs` (run with `MODEL_DIR` set to the fixture model, calling the handler the way `api.test.mjs` does): `GET reports tokenizer bpe`; `exact maxChars` (for `maxChars` 1, 7 and 40 the streamed text length equals it); `stop spanning tokens` (find a seed whose unstopped output contains `"\n\n"`, assert stopping with `["\n\n"]` ends right before it and never sends it); `past the context window` (`maxChars: 300` with `block: 16` completes and is deterministic for a seed); `non-alphabet prompt` returns 400 and `X-Dropped-Chars` counts mixed prompts; `missing tokenizer` (point `MODEL_DIR` at a copy of the fixture without `tokenizer.json`) returns 503 with a message that mentions the tokenizer; `char model unchanged` (the existing `api.test.mjs` still passes against `web/model`).
- [ ] **Step 2: Run:** `cd web && npm test`. Expected: the new tests FAIL, existing ones pass.
- [ ] **Step 3: Implement** the `gpt.js`, `generate.js` and `App.jsx` changes; fixture model generated by `python tests/make_fixture_model.py` (random-init weights, tokenizer trained on the sample text).
- [ ] **Step 4: Run:** `cd web && npm test`. Expected: all pass.
- [ ] **Step 5: Commit** `web tests/make_fixture_model.py`: `Serve BPE models from the web handler`.

### Task 6: Faster inference with a before/after benchmark

**Files:**
- Create: `web/bench/bench.mjs`, `results/bench_before.json`, `results/bench_after.json`
- Modify: `web/lib/gpt.js`, `web/api/generate.js`
- Test: `web/test/speed_parity.test.mjs`

**Interfaces:**
- Produces: `node web/bench/bench.mjs [--model DIR] [--out FILE]` prints and writes JSON `{node, cpu, model, params, runs, chars_per_sec, ttfc_ms, load_ms}` for three fixed prompts, 3 warm runs each, `maxChars` 200, seed 1.

- [ ] **Step 1: Capture the baseline before changing anything.** Run `node web/bench/bench.mjs --out results/bench_before.json` against the current `web/model`, and save the greedy (`topK: 1`) output of three fixed prompts as `web/test/fixtures/greedy_before.json`.
- [ ] **Step 2: Write the failing test** `web/test/speed_parity.test.mjs`: greedy output on the current model for the same prompts equals `greedy_before.json` exactly (so the optimizations do not change what the model says), and maximum absolute difference in logits for the first step against the saved reference is below `1e-3`.
- [ ] **Step 3: Run:** `cd web && node test/speed_parity.test.mjs`. Expected: PASS on the unmodified code (this pins behavior); it must still pass after Step 4.
- [ ] **Step 4: Optimize `gpt.js` and `generate.js`.** Candidates in order of expected payoff: hoist per-layer weight lookups out of `step()`; cheaper GELU only if the parity test still passes; partial top-k selection in `sample()`; unrolled inner loops in `linear()`; avoid allocating logits per step; load fp16 weights if `model.json.dtype` says so. Keep each optimization only if the benchmark improves.
- [ ] **Step 5: Run:** `cd web && npm test && node bench/bench.mjs --out ../results/bench_after.json`. Expected: tests pass, and `chars_per_sec` in the after file is higher than before; if an optimization does not help, revert it.
- [ ] **Step 6: Commit** `web results`: `Speed up inference and add a benchmark`.

### Task 7: GPU training scripts and runbook

**Files:**
- Create: `run_bpe.sh`, `docs/GPU_RUNBOOK.md`
- Modify: `requirements.txt` (torch only; unchanged unless needed)

**Interfaces:**
- Produces: `bash run_bpe.sh` runs `prepare_dolly.py` if `data/dolly_chat.txt` is missing, `train_tokenizer.py`, then trains the configs below with `--tokenizer tokenizer.json`, each with 3 seeds (1337, 1, 2), then generates sample replies and writes `<name>_s<seed>_results.json`, `logs/<name>_s<seed>.txt`, `generate_<name>.txt`. Configs (builder may adjust after the smoke runs, record why): `bpe_small` (4 layers, 4 heads, width 192, block 256, dropout 0.3) and `bpe_tiny` (3 layers, 3 heads, width 192, block 256, dropout 0.4). The runbook lists exactly what to copy back: `*_results.json`, `logs/`, `generate_*.txt`, `tokenizer.json`, and the chosen `*_best.pt`.

- [ ] **Step 1: Write `run_bpe.sh`** (same style as `run_dolly.sh`: `set -euo pipefail`, `PY` detection, a `run` helper).
- [ ] **Step 2: Smoke-test locally on CPU** with `STEPS=20 SEEDS=1337 bash run_bpe.sh` (the script reads `STEPS` and `SEEDS` overrides); Expected: every stage completes and a results JSON appears with `bits_per_char`.
- [ ] **Step 3: Write `docs/GPU_RUNBOOK.md`** with the exact commands for the GPU machine (clone, venv, `pip install -r requirements.txt`, `bash run_bpe.sh`) and the copy-back list.
- [ ] **Step 4: Commit** `run_bpe.sh docs`: `Add GPU training script and runbook`.

**Checkpoint: stop here and hand Shreyas the runbook.** Phase 2 starts when they return the outputs.

### Task 8: Repo cleanup and README (Phase 1 content)

**Files:**
- Create: `results/` (moved: `*_results.json`, `generate_*.txt`, existing `logs/` stays), `tests/test_readme_links.py`, `LICENSE` (MIT, author name from git config)
- Modify: `README.md`, `.gitignore`, `run_dolly.sh`, `run_experiments.sh`, `train.py` (`--out` still the checkpoint prefix; results JSON goes to `results/`), `.github/workflows/pages.yml` (no change expected), `web/package.json` name (`tiny-shakespeare-chat` becomes the real project name)
- Delete only after checking references: the stray `-p/` directory and `__pycache__/` (both untracked); leave `.claude-flow/` (ignored, may be in use)

**Interfaces:**
- Produces: README sections in this order: summary and class-project origin; how it works (next-token prediction, architecture diagram, KV cache); tokenizer story (characters then BPE, with a placeholder table for Phase 2 numbers marked `TODO(phase2)`); results with honest limits; how to run (local training, GPU runbook link, web app); repo layout; limits and what I would do next. Links to `results/...` and `docs/...` must resolve.

- [ ] **Step 1: Write the failing test** `tests/test_readme_links.py`: every relative Markdown link and image in `README.md` and `docs/GPU_RUNBOOK.md` points to an existing file, and the README contains the strings `class project` and `from scratch`.
- [ ] **Step 2: Run:** `python -m pytest tests/test_readme_links.py -v`. Expected: FAIL.
- [ ] **Step 3: Move result files, update scripts and `.gitignore`, rewrite the README, add the licence**, using `git mv` so history follows the files.
- [ ] **Step 4: Run:** `python -m pytest tests -v && cd web && npm test`. Expected: pass.
- [ ] **Step 5: Commit** everything: `Tidy the repo layout and rewrite the README`.

### Task 9 (Phase 2, after the GPU run): Swap in the BPE model and publish numbers

**Files:**
- Modify: `web/model/*` (new `weights.bin`, `model.json`, `tokenizer.json`), `README.md` (results table, tokenizer comparison, speed numbers), `results/`

**Interfaces:**
- Consumes: returned `*_results.json`, `logs/`, `generate_*.txt`, `tokenizer.json`, chosen checkpoint.

- [ ] **Step 1: Choose the model by validation bits per character** (mean over the 3 seeds; report the spread); test is read once, after the choice.
- [ ] **Step 2: Export:** `python export_weights.py --ckpt <chosen>_best.pt --fp16` (drop `--fp16` if the logit check from Task 4 or the quality check below fails).
- [ ] **Step 3: Run:** `cd web && npm test` and `node bench/bench.mjs --out ../results/bench_final.json`. Expected: tests pass; record before/after speed with hardware and Node version.
- [ ] **Step 4: Fill the README placeholders** (`TODO(phase2)`): char vs BPE table in bits per character with train/val/test, seeds and spread, the five sample replies from the old and new model side by side, and the speed table. If BPE does not beat `dolly_large`, say so plainly and keep the old model deployed.
- [ ] **Step 5: Run** `python -m pytest tests -v`, `cd web && npm test && npm run build` (no dev server running first), then try the app locally with `cd web && npx vercel dev` or the Vite dev UI against a local handler.
- [ ] **Step 6: Commit and, when Shreyas says to, push** `bpe-polish` to `origin` and open the pull request from the browser (no `gh`).
