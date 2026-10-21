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
_To be filled in after training: final train loss, validation loss (held-out last 10%), and sample generations._
