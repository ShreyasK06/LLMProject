# Tiny Shakespeare GPT: a language model trained from scratch

A small GPT-style language model, built and trained from random initialization (**no pretrained weights**) on about 1 MB of Shakespeare, then served through a React chat interface with a Vercel backend.

The goal is to see the whole pretraining pipeline end to end: collect data, prepare it, train with next-token prediction, and prompt the result to see what it learned.

## The project in one picture

```
 TinyShakespeare text           train.py                         export_weights.py
 (1.1M characters)  ──►  tokenize ► GPT (3.2M params) ──►  best checkpoint ──►  weights.bin + model.json
                         next-token prediction              (lowest val loss)            │
                         AdamW, early stopping                                           ▼
                                                              Vercel serverless function (api/generate.js)
   Browser (React UI) ◄──── streamed characters ◄──────────   runs the model in plain JavaScript
```

| Assignment step | Where it happens |
|---|---|
| 1. Find and download a text dataset | `data/input.txt` (TinyShakespeare) |
| 2. Small architecture, no pretrained weights | `train.py`: `GPT` class, random init |
| 3. Pretrain with next-token prediction | `train.py`, `run_experiments.sh` |
| 4. Prompt the trained model and observe | `generate.py`, the web app in `web/`, samples below |

## 1. The idea: language modeling as next-token prediction

A language model learns one thing: given the text so far, predict the next token. Training repeats that on millions of examples; generation repeats it in a loop, appending each prediction to the input and predicting again.

```
text:     T  o     b  e  ,     o  r  ...
input:    T  o  _  b  e  ,  _  o        (a window of 256 characters)
target:   o  _  b  e  ,  _  o  r        (the same window shifted by one)
```

Every position in a window is a separate training example, scored with cross-entropy loss against the true next character. Predicting the next character well forces the model to pick up spelling, formatting, vocabulary and style as a side effect. This is the same objective used to pretrain large LLMs; only the scale differs.

## 2. Data

- **Source:** TinyShakespeare, 1,115,394 characters of Shakespeare plays, from Karpathy's char-rnn repo.
- **Tokenization:** character level. The text has 65 distinct characters (letters, digits, punctuation, newline), and each is one token. This keeps the vocabulary tiny and needs no tokenizer; the tradeoff is that the model must learn spelling and words from scratch.
- **Split (sequential 80/10/10):** about 892k characters for training, 112k for validation, 112k for test. Validation picks the best checkpoint and drives early stopping. Test is measured once at the end, on the chosen checkpoint, and never used for any decision.

## 3. Model architecture

A decoder-only transformer, like GPT-2 at a much smaller scale. The selected model (`exp_c_small`) has **3.24M parameters**.

```
character ids ─► token embedding (65 → 256) + position embedding (256 positions)
              ─► 4 × transformer block:
                    LayerNorm ─► causal multi-head self-attention (4 heads) ─► + residual
                    LayerNorm ─► MLP (256 → 1024 → 256, GELU)               ─► + residual
              ─► LayerNorm ─► output layer (256 → 65 logits)   [weights tied to the token embedding]
              ─► softmax: probability of each possible next character
```

- **Causal self-attention:** each position may look only at earlier positions, never ahead. That is what makes next-token training valid and lets the same network generate one character at a time.
- **Positional embeddings:** attention has no built-in sense of order, so each position gets a learned vector added to the character embedding.
- **Weight tying:** the output layer reuses the token embedding matrix, saving parameters.
- **Context window:** 256 characters.

## 4. Training

`train.py` (PyTorch, runs on GPU or CPU):

- **Optimizer:** AdamW with learning rate 1e-3, linear warmup for 100 steps, then cosine decay to 1e-4, plus gradient clipping at 1.0. Weight decay applies only to weight matrices, not to biases, LayerNorm or embeddings.
- **Batches:** 64 random windows of 256 characters per step, up to 5000 steps (about 82M characters seen, roughly 90 passes over the training text).
- **Regularization:** dropout 0.3 and early stopping on validation loss (patience of 4 evaluations, every 250 steps).
- **Checkpointing:** `<name>_best.pt` is saved whenever validation loss improves, so we keep the best model, not the last one.
- **Sanity check:** the initial loss is about 4.24, close to random guessing over 65 characters (ln 65 = 4.17).

### The overfitting problem, and the fix

The first run (a 10.8M-parameter model trained for a fixed 5000 steps and saving only the final model) overfit. Training loss kept falling while validation loss bottomed at step 1750 and then rose. With only 1 MB of text, a model that large starts memorizing the training set.

In numbers: at step 5000 the training loss was 0.57 but the validation loss was 1.75, against a best validation loss of 1.47 back at step 1750.

The fix was to make training measure what matters: a three-way split, best-validation checkpointing, early stopping, and a comparison of three configurations (`run_experiments.sh`).

## 5. Results

All losses are cross-entropy in nats per character (lower is better; perplexity is e^loss, roughly "how many characters the model is choosing between"). Train is 200 random batches from the train split; validation and test are full sweeps of their held-out splits. This is a single split with a single seed, not cross-validation.

| Config | Params | Dropout / weight decay | Stopped at | Best step | Train | Val | Test | Val ppl | Test ppl |
|---|---|---|---|---|---|---|---|---|---|
| exp_a_baseline | 10.77M | 0.2 / 0.1 | 2500 (early stop) | 1500 | 1.1076 | 1.4453 | 1.7995 | 4.24 | 6.05 |
| exp_b_reg | 10.77M | 0.3 / 0.2 | 3000 (early stop) | 2000 | 1.1208 | 1.4389 | 1.7804 | 4.22 | 5.93 |
| **exp_c_small** (selected) | 3.24M | 0.3 / 0.1 | 5000 (full run) | 5000 | 1.1374 | **1.4338** | 1.8230 | 4.19 | 6.19 |

**Selected `exp_c_small`** because it had the lowest validation loss (test was not used to choose). Things to say honestly:

- **It's effectively a three-way tie.** The validation losses are within 0.011 nats with one seed each. Ranked by test loss the order would be b, a, c, again by small margins. I would not claim the smaller model is truly better, only that it is as good with a third of the parameters and trains in about 5.5 minutes versus 7.5 or more.
- **Overfitting was fixed.** The two 10.8M models still overfit and were stopped early (steps 2500 and 3000, with train loss falling while validation loss rose). The 3.24M model with heavier dropout trained all 5000 steps without overfitting and was still improving slowly.
- **Test loss is about 0.35 nats higher than validation for every config.** The split is sequential, so validation and test are different plays from the training text and from each other. That is distribution shift, not a bug, and validation and test losses are not directly comparable.
- Early-stopped runs stopped before the learning-rate schedule finished, so they are not strictly comparable to the full-length run.

### Initial run (90/10 split, saved the final-step model, no early stopping)

Kept for comparison. It used an older 90/10 split and a fixed 5000 steps, so its numbers are not comparable to the table above.

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.2446 | 4.2393 |
| 500 | 1.4936 | 1.6917 |
| 1000 | 1.2551 | 1.5280 |
| **1750 (best val)** | 1.0822 | **1.4657** |
| 2500 | 0.9278 | 1.5131 |
| 3500 | 0.7399 | 1.6087 |
| **5000 (final, saved)** | **0.5719** | **1.7489** |

<details>
<summary>Full training logs (all four runs)</summary>

Initial run:
```
device=cuda  vocab=65  params=10.77M
step     0 | train 4.2446 | val 4.2393 |   5s
step   250 | train 1.8906 | val 2.0139 |  52s
step   500 | train 1.4936 | val 1.6917 |  98s
step   750 | train 1.3415 | val 1.5810 | 146s
step  1000 | train 1.2551 | val 1.5280 | 194s
step  1250 | train 1.1831 | val 1.4849 | 242s
step  1500 | train 1.1305 | val 1.4874 | 290s
step  1750 | train 1.0822 | val 1.4657 | 338s
step  2000 | train 1.0256 | val 1.4737 | 387s
step  2250 | train 0.9836 | val 1.4847 | 435s
step  2500 | train 0.9278 | val 1.5131 | 483s
step  2750 | train 0.8729 | val 1.5317 | 532s
step  3000 | train 0.8264 | val 1.5541 | 580s
step  3250 | train 0.7778 | val 1.5845 | 628s
step  3500 | train 0.7399 | val 1.6087 | 675s
step  3750 | train 0.6997 | val 1.6386 | 723s
step  4000 | train 0.6607 | val 1.6506 | 771s
step  4250 | train 0.6313 | val 1.6882 | 819s
step  4500 | train 0.6085 | val 1.6980 | 866s
step  4750 | train 0.5914 | val 1.7211 | 914s
step  5000 | train 0.5719 | val 1.7489 | 962s
```

The final experiments' logs are in `logs/exp_a_baseline.txt`, `logs/exp_b_reg.txt` and `logs/exp_c_small.txt`, with a JSON summary per run in `exp_*_results.json`.
</details>

## 6. Generating text

Generation is a loop: feed the prompt, get a probability for each of the 65 characters, pick one, append it, repeat. Two knobs control the pick:

- **Temperature** rescales the probabilities. Low values make the model cautious and repetitive; high values make it adventurous and error-prone. Default 0.8.
- **Top-k** keeps only the k most likely characters before sampling, which cuts off the long tail of nonsense. Default 40.

Sample outputs from the selected model (`python generate.py --ckpt exp_c_small_best.pt`, 300 characters, seed 0; full text in [`generate_output_final.txt`](generate_output_final.txt)):

```
PROMPT: 'KING HENRY:\nMy lords,'
KING HENRY:
My lords, and purchance, as I saw thee on.

KING RICHARD III:
I would not with my body words thea
Could resign the realm of my order age:
But I am great as thou deny to speak:
Weep'st thou there to pray thee as I am.

STANLEY:
Heavens my gracious lord.

PROMPT: 'First Citizen:\nBefore we proceed'
First Citizen:
Before we proceed to be a certain of his life,
To Brittany, but fear?

First Senator:
Sir, you must perform you be to the country:
So you are slain; true, to the consul.

MENENIUS:
How is't the very points have found, I must be
believe it so, and well feather of wanton; to complain.
```

**What it learned:** play formatting (speaker names in capitals followed by a colon, line breaks, verse-length lines), Early Modern English vocabulary and spelling, and contextually plausible speakers (MENENIUS after a Roman-citizen prompt, KING RICHARD III and STANLEY after KING HENRY).

**What it did not learn:** meaning. Individual lines are often grammatical, but nothing connects from one line to the next, and there are occasional invented words ("thea", "receivern"). That is expected from a 3M-parameter character-level model trained on 1 MB: it captures surface statistics and local structure, not content or plot. (The first run's samples, from the overfit model, are in `generate_output.txt`; they show the same pattern.)

## 7. The chat web app (`web/`)

A React (Vite) interface with a Vercel serverless backend.

- **Why the model runs in JavaScript:** Vercel's Python runtime cannot hold PyTorch. The model is small (13 MB of weights), so `web/lib/gpt.js` reimplements the forward pass in plain JavaScript with a key/value cache, so each new character costs only a single-position computation. `export_weights.py` converts the checkpoint into `weights.bin` plus `model.json`.
- **Correctness check:** the JavaScript forward pass was compared against PyTorch on the same input; the largest difference in the output logits was 8e-7 over 100 positions.
- **Backend (`web/api/generate.js`):** `POST` with a prompt and settings streams the generated characters back as plain text; `GET` reports model status. Inputs are validated and clamped (prompt length, reply length, temperature, top-k).
- **Frontend (`web/src/App.jsx`):** a chat view that renders the stream as it arrives, with a Stop button, example prompts, and settings for temperature, top-k, reply length and seed. There is also an option to carry the conversation as context.
- **It is not an assistant.** This is a base model: each message is treated as the start of a passage and the model continues it, like a play script. Typing a speaker name and a colon works best. Characters outside the 65-character vocabulary are dropped, and the UI says so.
- **Tests:** `cd web && npm test` runs the API handler tests (streaming, seed determinism, limits, long generation) and a simulated-browser test of the real React app. A seed reproduces output within the web app, though not identically to the Python script (different random number generator). Beyond the 256-character context the web version drops the oldest half and re-reads the rest, rather than sliding one character at a time.

## 8. Reproduce it

```
pip install -r requirements.txt            # install a CUDA build of torch for GPU
python train.py --out exp_c_small --n_layer 4 --n_head 4 --n_embd 256 --dropout 0.3 --patience 4
bash run_experiments.sh                    # the three experiments, logs in logs/
python generate.py --ckpt exp_c_small_best.pt "ROMEO:"

# web app
python export_weights.py --ckpt exp_c_small_best.pt   # writes web/model/{weights.bin,model.json}
cd web && npm install && npm run dev                  # UI and API together, no Vercel login needed
```

To deploy, import the repo into Vercel and set the **Root Directory** to `web`; `web/vercel.json` bundles the model files into the function. The weights (`web/model/`) must be committed first, since the checkpoints themselves (`*.pt`) are git-ignored.

## 9. Repository map

| Path | Purpose |
|---|---|
| `data/input.txt` | training text |
| `train.py` | model, training loop, early stopping, final train/val/test report |
| `generate.py` | sample text from a saved checkpoint |
| `run_experiments.sh`, `logs/`, `exp_*_results.json` | the three experiments and their results |
| `export_weights.py` | checkpoint to JavaScript-readable weights |
| `web/` | React UI, Vercel function, JavaScript model, tests |
| `generate_output*.txt` | full sample generations (initial run and final model) |

## 10. Limitations and what I would do next

- Results come from a single split and a single seed; the three configurations are statistically tied. Running several seeds would show whether the differences are real.
- The corpus is tiny, so the model is data-limited. More text would help more than a bigger model.
- A character-level model must spend capacity on spelling. A subword (BPE) tokenizer would give longer effective context and more coherent output.
- Validation and test are different plays from the training text, so the gap between them reflects distribution shift as well as generalization.
- The model has no instruction tuning, so it cannot answer questions; adding that is a separate fine-tuning stage.
