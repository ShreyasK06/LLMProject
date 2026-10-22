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

## Results
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
_Not yet added: run `python generate.py` and paste the output here._
