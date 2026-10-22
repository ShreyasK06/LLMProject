# LLMProject

A small GPT-style language model trained from scratch (random initialization, no pretrained weights) with the standard next-token-prediction objective.

## Data
`data/input.txt` is TinyShakespeare (~1.1 MB, 65 distinct characters), from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt.
The first 90% is used for training and the last 10% for validation.

## Model
Decoder-only transformer over characters (one character = one token): pre-LayerNorm blocks, causal self-attention, GELU MLP, learned position embeddings, tied input/output embeddings.

| | GPU | CPU fallback |
|---|---|---|
| layers / heads / embed | 6 / 6 / 384 | 4 / 4 / 128 |
| context / batch | 256 / 64 | 128 / 32 |
| parameters | ~10.7M | ~0.8M |
| steps | 5000 | 3000 |

The config is chosen automatically from `torch.cuda.is_available()`. Training uses AdamW with warmup + cosine learning-rate decay and gradient clipping.

## Run
```
pip install -r requirements.txt   # install a CUDA build of torch for GPU
python train.py                   # prints train/val loss every 250 steps, saves model.pt
python generate.py                # samples from several prompts
python generate.py "HAMLET:"      # or your own prompts
```

## Chat web app (`web/`)
A React (Vite) chat UI with a Vercel serverless backend. Vercel's Python runtime can't hold PyTorch, so the backend runs the model in plain JavaScript (`web/lib/gpt.js`: KV-cached decoding, no dependencies) from exported weights. The model is a base LM, not an assistant: each message is the start of a passage and the model continues it, streamed character by character. Settings: temperature, top-k, reply length, seed, and an option to carry the conversation as context.

1. Export the trained checkpoint (writes `web/model/weights.bin` ~13 MB and `web/model/model.json`), then commit both:
   ```
   python export_weights.py --ckpt exp_c_small_best.pt
   ```
2. Run locally: `cd web && npm install && npm run dev` (serves the UI and `/api/generate` on one port; no Vercel login needed).
3. Deploy: import the repo in Vercel and set **Root Directory** to `web`. `web/vercel.json` bundles `model/**` into the function.
4. Tests: `cd web && npm test` (API handler incl. seed determinism and limits, plus a jsdom test of the React UI). The JS forward pass was checked against PyTorch logits on the same input: max absolute difference 8e-7 over 100 positions.

Inference differs from `generate.py` in two small ways: sampling uses a JS PRNG (a seed reproduces output within the web app, not against PyTorch), and beyond the context window the oldest half is dropped and re-encoded instead of sliding one character at a time.

## Results

### Final experiments (80/10/10 split, early stopping on validation loss)
`run_experiments.sh` trains three configs on GPU (block 256, batch 64, lr 1e-3, patience 4 evals of 250 steps, seed 1337). Train/val/test are cross-entropy in nats per character for the best-validation checkpoint; val and test are full sweeps over the held-out splits (the last 20% of the text, split sequentially into val then test), train is 200 random batches from the train split. This is a single split with a single seed, not cross-validation.

| Config | Params | Dropout / weight decay | Best step | Train | Val | Test | Val ppl | Test ppl |
|---|---|---|---|---|---|---|---|---|
| exp_a_baseline | 10.77M | 0.2 / 0.1 | 1500 | 1.1076 | 1.4453 | 1.7995 | 4.24 | 6.05 |
| exp_b_reg | 10.77M | 0.3 / 0.2 | 2000 | 1.1208 | 1.4389 | 1.7804 | 4.22 | 5.93 |
| **exp_c_small** (selected) | 3.24M | 0.3 / 0.1 | 5000 | 1.1374 | **1.4338** | 1.8230 | 4.19 | 6.19 |

exp_c_small was selected because it has the lowest validation loss (test was not used for selection). Caveats:
- The three validation losses are within 0.011 nats of each other with one seed, so they are effectively tied; I would not claim exp_c is truly better. Test loss alone would rank them b < a < c, again by small margins.
- Test loss is ~0.35 nats higher than validation for every config. The split is sequential, so val and test are different plays from the training text and from each other; the gap is distribution shift, not a bug, and val and test losses are not exchangeable.
- The larger models (a, b) overfit and early-stopped (steps 2500 and 3000; train loss keeps falling while val rises). The 3.24M model with heavier dropout trained the full 5000 steps without overfitting and was still improving slowly, so the schedule length, not overfitting, limited it.
- Early-stopped runs stopped before the cosine learning-rate decay finished, so they are not strictly comparable to the full-length run.
- These numbers use an 80/10/10 split and are not comparable to the initial run below (90/10, final-step model).

### Sample generations from the selected model (exp_c_small, best checkpoint)
`python generate.py --ckpt exp_c_small_best.pt` (300 characters, temperature 0.8, top-k 40, seed 0). Full output in [`generate_output_final.txt`](generate_output_final.txt); excerpts:

```
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
```

Observations (qualitative, from five samples at one seed, so not a measured comparison with the initial run): play formatting, speaker names, verse-length lines and period vocabulary are all reproduced, and the speakers that appear are plausible for the prompt's context (MENENIUS and SICINIUS after a Roman-citizen prompt; KING RICHARD III and STANLEY after KING HENRY). Individual lines are often grammatical, but meaning does not carry across lines, and there are still occasional non-words ("thea", "receivern", "devotion'd"). This is expected for a character-level model with ~3M parameters trained on ~1 MB: it learns surface statistics and local structure, not content.

### Initial run (90/10 split, final-step model, no early stopping)
Run on GPU (CUDA): 10.77M parameters, 5000 steps, ~16 min. Loss is cross-entropy in nats per character; train and validation numbers come from 40 random batches of the 90% train split and the held-out last-10% validation split respectively (a single split, no k-fold cross-validation).

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.2446 | 4.2393 |
| 500 | 1.4936 | 1.6917 |
| 1000 | 1.2551 | 1.5280 |
| **1750 (best val)** | 1.0822 | **1.4657** |
| 2500 | 0.9278 | 1.5131 |
| 3500 | 0.7399 | 1.6087 |
| **5000 (final, saved model)** | **0.5719** | **1.7489** |

Validation loss bottoms out around step 1750 (1.47, ~4.3 perplexity) and then rises while train loss keeps falling: the model overfits the ~1 MB corpus. The saved `model.pt` is the step-5000 model (val 1.75, ~5.7 perplexity), not the best-validation checkpoint. Fixes would be earlier stopping, more dropout/weight decay, a smaller model, or more data.

Full log:
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

### Sample generations
Generated by the step-5000 model (`generate.py`: 300 characters, temperature 0.8, top-k 40, seed 0). Full output in [`generate_output.txt`](generate_output.txt); excerpts:

```
PROMPT: 'First Citizen:\nBefore we proceed'
First Citizen:
Before we proceed to see the market-place; he
should thrive, by the honour and his neck with him.

Second Citizen:
He's a countenant to use his grace by the world.

PROMPT: 'KING HENRY:\nMy lords,'
KING HENRY:
My lords, at your hands, and my wife's own.

YORK:
The world was not well might die but a place:
You will seem to him protect that you look'd upon.

PROMPT: 'To be, or not to be'
To be, or not to be so great as a bay
To be so to brief; for that my rests are
As is the bed blood wherein I did mew'd.
```

What it learned: play formatting (speaker names in caps followed by a colon, line breaks, verse-length lines), Early-Modern-English vocabulary and spelling, and mostly valid words with a few invented ones ("gientleman", "countenant"). It does not learn meaning: sentences are locally grammatical but incoherent over more than a line or two, and it continues "ROMEO:" with other characters (AUTOLYCUS, POMPEY) because this model only knows character statistics of the corpus, not the plays' plots.
