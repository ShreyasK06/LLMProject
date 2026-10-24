# Tiny GPT: a small language model trained from scratch

This started as a class project: pretrain a small GPT from scratch (random initialization, no pretrained weights) on a text dataset, then prompt it and see what it learned. I did that on 1 MB of Shakespeare and then chose to extend it: the same code trained as a chat model on Databricks Dolly 15k, a from-scratch BPE tokenizer written in Python and JavaScript, a React chat app with a Vercel back end that runs the model in plain JavaScript, and faster inference measured before and after. The deployed model is the character-level `dolly_large` (10.88M parameters). It writes fluent-looking replies that contain no facts or meaning; the sections below explain why.

```
data/*.txt --> train.py -------------------> <name>_best.pt ----> export_weights.py --> web/model/
               next-token prediction,       (lowest val loss)                         weights.bin, model.json
               AdamW, early stopping                                                         |
                                                                                             v
Browser (React, GitHub Pages) <-- streamed text <-- Vercel function (web/api/generate.js + web/lib/gpt.js)
```

## How it works

**Next-token prediction.** Given the text so far, predict the next token. In training the target is the input window shifted by one, so every position is one example scored with cross-entropy. Generation runs the same prediction in a loop, appending each sampled token.

**Architecture.** A decoder-only transformer like GPT-2, much smaller (`train.py`, mirrored in `web/lib/gpt.js`):

```
token ids -> token embedding (vocab x d) + learned position embedding (block x d)
          -> n_layer x transformer block:
                LayerNorm -> causal multi-head self-attention -> + residual
                LayerNorm -> MLP (d -> 4d -> d, GELU)         -> + residual
          -> LayerNorm -> linear head (d -> vocab), weights tied to the token embedding
          -> softmax over the vocabulary = probability of each next token
```

- **Causal attention:** each position attends only to earlier positions, so all positions train at once and the network can generate one token at a time.
- **Weight tying:** the output head reuses the token embedding matrix.
- **KV cache (JavaScript inference):** past keys and values are kept, so each new token costs one single-position pass. Past `block` tokens the oldest half is dropped and the rest re-read.
- **Sampling:** temperature divides the logits (default 0.8); top-k keeps the k most likely tokens (default 40).

**Training** (`train.py`, PyTorch): AdamW, learning rate 1e-3 with 100 warmup steps and cosine decay to 1e-4, gradient clipping at 1.0, weight decay on the attention and MLP weight matrices only (biases, LayerNorm gains and the token and position embeddings, which the tied head shares, are exempt), dropout, evaluation every 250 steps. The best-validation checkpoint is kept, the run scripts stop after 4 evaluations without improvement, and test is measured once, on the best checkpoint.

## Tokenizer: characters, then BPE

**Why characters first.** One token per character needs no tokenizer and a tiny vocabulary (65 characters for Shakespeare, 95 for Dolly). The cost is that the model must learn spelling and word boundaries one character at a time, and a 512-token context is only 512 characters.

**What BPE changes.** Byte-pair encoding starts from single characters and repeatedly merges the most frequent adjacent pair, so common words and word pieces become single tokens. The same text becomes fewer tokens, the context covers more text, and the model can spend capacity above spelling.

**Implementation.** `bpe.py` is a character-level BPE written from scratch with the Python standard library. Text is pre-split on ASCII character classes and merges never cross those pieces. `train_tokenizer.py` learns merges on the training split only (first 80% of the text; vocabulary 2,000). `web/lib/bpe.js` is a JavaScript port. `tests/test_bpe.py` checks round trips, merge determinism and piece boundaries, and writes `tests/fixtures/bpe_parity.json` (8 strings, including empty, a 2,000-character transcript and non-ASCII text); `web/test/bpe.test.mjs` asserts the JavaScript port produces identical ids. Characters outside the alphabet are dropped and counted.

**Comparing tokenizers fairly.** Loss per token is not comparable across tokenizers, so `train.py` also reports bits per character (bpc): total negative log likelihood on a split divided by (characters covered by the target tokens × ln 2). Both kinds of model use the same text and split offsets. The character-level numbers below are the stored losses in `results/dolly_*_results.json` (nats per character) divided by ln 2.

| Model (Dolly 15k) | Params | Train bpc | Val bpc | Test bpc |
|---|---|---|---|---|
| `dolly_small`, characters | 3.31M | 1.619 | 1.716 | 1.709 |
| `dolly_large`, characters (deployed) | 10.88M | 1.288 | 1.536 | 1.530 |
| BPE (`run_bpe.sh`, 2 configs × 3 seeds) | TODO(phase2) | TODO(phase2) | TODO(phase2) | TODO(phase2) |

For the character models, Train is an estimate over 200 random training batches; Val and Test are full sweeps.

The BPE models have not been trained yet; they need a GPU ([docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md)). The selection rule is fixed in advance: lowest validation bits per character, with test reported once.

## Data

- **Shakespeare (the original class project):** TinyShakespeare (`data/input.txt`, from Karpathy's char-rnn repo), 1,115,394 characters, 65 distinct characters. Split sequentially 80/10/10 (about 892k / 112k / 112k characters), so validation and test are later plays than the training text.
- **Dolly 15k (chat):** [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), about 15,000 human-written instruction and response pairs (CC BY-SA 3.0). `prepare_dolly.py` keeps examples without a context passage, ASCII only, cuts responses at a sentence end (400 characters at most), formats each pair as `User:` / `Assistant:` turns ending in a blank line, and shuffles the examples with seed 0. The result is 2,879,010 characters with 95 distinct characters (`logs/dolly_large.txt`). `train.py` then splits it sequentially 80/10/10, so validation and test come from the same mix as training.

## Results and their limits

All numbers come from `results/*_results.json` and are measured on the best-validation checkpoint. **Train** is 200 random batches from the training split; **val** and **test** are full sweeps of the held-out validation and test splits. Losses are cross-entropy in nats per character (divide by ln 2 ≈ 0.693 for bits).

**Shakespeare** (`run_experiments.sh`, block 256, batch 64):

| Config | Params | Dropout / weight decay | Stopped at | Best step | Train | Val | Test |
|---|---|---|---|---|---|---|---|
| `exp_a_baseline` | 10.77M | 0.2 / 0.1 | 2500 (early stop) | 1500 | 1.1076 | 1.4453 | 1.7995 |
| `exp_b_reg` | 10.77M | 0.3 / 0.2 | 3000 (early stop) | 2000 | 1.1208 | 1.4389 | 1.7804 |
| **`exp_c_small`** (selected on val) | 3.24M | 0.3 / 0.1 | 5000 (step cap) | 5000 | 1.1374 | **1.4338** | 1.8230 |

**Dolly 15k chat** (`run_dolly.sh`, block 512, batch 32, dropout 0.3):

| Config | Params | Weight decay | Best step (last) | Train | Val | Test |
|---|---|---|---|---|---|---|
| `dolly_small` (4 layers, width 256) | 3.31M | 0.1 | 5000 (5000) | 1.1223 | 1.1896 | 1.1842 |
| **`dolly_large`** (6 layers, width 384, deployed) | 10.88M | 0.2 | 4750 (5000) | 0.8931 | **1.0646** | 1.0602 |

**The overfitting story.** The first Shakespeare run (10.77M parameters, a 90/10 split, a fixed 5000 steps, final model saved; `logs/initial_run.txt`) reached its best validation loss of 1.4657 at step 1750, then kept memorizing: at step 5000 training loss was 0.5719 and validation 1.7489. That led to the three-way split, best-validation checkpointing, early stopping and the configurations above; the two 10.77M models still stopped early, while the 3.24M one ran all 5000 steps.

**What to keep in mind when reading these numbers:**

- **One seed per character model (1337), no cross-validation.** The three Shakespeare validation losses are within 0.012 nats, a tie (ranked by test the order would be b, a, c); `exp_c_small` matched the others with a third of the parameters. The Dolly gap (0.125 nats on validation, 0.124 on test) is about ten times larger, but still one seed.
- **Shakespeare test is 0.34 to 0.39 nats above validation for every config.** The split is sequential, so validation and test are different plays: distribution shift, not a bug. Dolly is shuffled by example first, and its validation and test agree closely.
- **The two datasets are not comparable:** different text, vocabulary (65 vs 95 characters) and split.
- `dolly_small` was still improving at the 5000-step cap; `dolly_large` had plateaued and has the larger train/val gap (0.17 vs 0.07 nats). Early-stopped runs did not finish their learning-rate schedule.

**What the models learned.** The Shakespeare model learned play formatting, period vocabulary and plausible speakers, but no meaning across lines ([results/generate_output_final.txt](results/generate_output_final.txt); overfit first run: [results/generate_output.txt](results/generate_output.txt)). The chat model learned the shape of an answer but not facts. Asked `What is the capital of France?`, `dolly_large` writes "There are many capital of France - in the United States of America is Ronaldo and a United States of America - Italy in Asia - …" ([results/generate_dolly_large.txt](results/generate_dolly_large.txt), temperature 0.8, top-k 40, seed 0).

## Speed

`web/bench/bench.mjs` times the JavaScript inference on the deployed 10.88M-parameter character model: 3 chat prompts, 200 characters each, medians over 9 timed runs (protocol in [results/README.md](results/README.md)).

| Build | Characters per second | Time to first character |
|---|---|---|
| Before (`web/lib/gpt.js` at commit `eca0611`, [results/bench_before.json](results/bench_before.json)) | 48.72 | 750.49 ms |
| After (current `web/lib/gpt.js`, [results/bench_after.json](results/bench_after.json)) | 79.94 | 448.91 ms |

Measured back to back on one laptop (11th Gen Intel Core i5-1135G7, 8 logical cores, Windows x64, Node v22.18.0); the numbers are specific to that machine and the Vercel server will differ. The gap covers every change this branch made to `web/lib/gpt.js`, both the decoding-loop restructuring and the BPE plumbing, not the loop changes alone. `web/test/speed_parity.test.mjs` checks that the new code gives the same greedy outputs as the old (it skips itself, exit 0, if `web/model` is not the model its fixture came from).

## How to run

**Train locally** (CPU works for small configs; the reported runs used a CUDA GPU):

```bash
pip install -r requirements.txt   # install a CUDA build of torch for GPU training
bash run_experiments.sh            # Shakespeare: three configs, logs in logs/, results in results/
bash run_dolly.sh                  # Dolly: download, two configs, sample replies into results/
python generate.py --chat --ckpt dolly_large_best.pt "What is the capital of France?"
pip install -r requirements.txt -r requirements-dev.txt   # adds pytest
python -m pytest tests -v          # tokenizer, JS/Python parity, training smoke tests, export, doc links
```

`train.py --out NAME` writes `NAME_best.pt` and `NAME_final.pt` (git-ignored) and `results/NAME_results.json` (`--results_dir` changes the folder); `--tokenizer tokenizer.json` trains on BPE tokens.

**BPE runs on a GPU machine:** `bash run_bpe.sh`; see [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md) for setup, outputs and what to copy back.

**Web app:**

```bash
python export_weights.py --ckpt dolly_large_best.pt   # writes web/model/ (add --fp16 for 16-bit weights)
cd web && npm install && npm test                      # tokenizer parity, API handler, UI, BPE serving, speed parity
npm run dev:ui                                         # UI and API together on localhost:5173
```

**Deployment.** GitHub Pages serves only static files, so `.github/workflows/pages.yml` builds the React front end for Pages, and the model runs in a Vercel serverless function (`web/api/generate.js`, Root Directory `web`). The model runs in JavaScript because Vercel's Python runtime cannot hold PyTorch. Set the repository variable `API_URL` to the Vercel URL (`ALLOWED_ORIGINS` on Vercel limits callers). The UI sends `User:` / `Assistant:` turns and the server stops at the first blank line.

## Repository layout

| Path | Contents |
|---|---|
| `train.py`, `generate.py` | model, training, evaluation; sampling from a checkpoint |
| `bpe.py`, `train_tokenizer.py` | from-scratch BPE tokenizer and its training script |
| `prepare_dolly.py` | builds `data/dolly_chat.txt` from Dolly 15k |
| `run_experiments.sh`, `run_dolly.sh`, `run_bpe.sh` | Shakespeare experiments, Dolly chat models, BPE chat models (GPU) |
| `export_weights.py` | checkpoint to `weights.bin` + `model.json` (+ `tokenizer.json`) for JavaScript |
| `results/`, `logs/` | results JSON, sample generations and benchmarks; training logs |
| `tests/` | Python tests |
| `web/` | React UI (`src/`), Vercel function (`api/`), JS model and tokenizer (`lib/`), weights (`model/`), tests, benchmark |
| `docs/GPU_RUNBOOK.md` | BPE training on a GPU machine |

## Limitations and what would come next

- **The models produce fluent-sounding text without facts or meaning.** At 3 to 11 million parameters and under 3 million characters of data, they capture surface statistics: formatting, vocabulary, the shape of a reply. Correct answers would take far more data and parameters, outside this project's scope (no pretrained weights, no extra data).
- **BPE results are pending** (Phase 2). The table will be filled from `results/bpe_*_results.json` with the mean and spread over three seeds, and the BPE model deployed only if it is better on validation bits per character.
- **One seed per character model, no cross-validation,** so small differences between them are not evidence.
- **No instruction tuning;** the chat model imitates the transcript format it was trained on.
- Next: finish the BPE runs, run more seeds for the character baselines, and try a reserved end-of-reply token instead of the blank-line stop.

## License

MIT, see [LICENSE](LICENSE). Dolly 15k is CC BY-SA 3.0 by Databricks.
