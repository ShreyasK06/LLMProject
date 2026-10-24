# GPU runbook: BPE chat models

This trains two small BPE chat models (`bpe_small`, `bpe_tiny`) with three seeds each (1337, 1, 2) on Dolly 15k, using one shared tokenizer. It needs a CUDA GPU and internet (once, to download Dolly).

## 1. Set up

```bash
git clone <your repo URL> LLMProject
cd LLMProject
git checkout bpe-polish        # or main, once bpe-polish has been merged
python -m venv .venv
source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate ; PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` only lists `torch>=2.0`, which on many machines installs a CPU-only wheel. Installing the torch wheel that matches your CUDA driver (see pytorch.org "Get Started") is your responsibility. Check it before starting:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## 2. Run everything

```bash
bash run_bpe.sh
```

On Windows, Git Bash works. `run_bpe.sh` does these steps in order:

1. If `data/dolly_chat.txt` is missing, runs `prepare_dolly.py` (downloads Dolly 15k).
2. Trains the tokenizer (`train_tokenizer.py`, vocab 2000, first 80% of the text) and writes `tokenizer.json`.
3. For each seed, trains `bpe_small_s<seed>` (4 layers, 4 heads, width 192, block 256, dropout 0.3) and `bpe_tiny_s<seed>` (3 layers, 3 heads, width 192, block 256, dropout 0.4), both with batch 64, patience 4, 5000 max steps (early stopping on validation loss).
4. Generates sample replies from each best checkpoint.

Optional overrides: `STEPS=5000`, `SEEDS="1337 1 2"`, `DATA=data/dolly_chat.txt`, `VOCAB=2000`. For example, `SEEDS="1337" bash run_bpe.sh` runs a single seed first.

### Without bash

Run the same steps by hand from the repo root (use `python3` if that is your interpreter name). Replace `SEED` with each of 1337, 1 and 2 in turn:

```bash
mkdir -p logs
python prepare_dolly.py                     # only if data/dolly_chat.txt does not exist
python train_tokenizer.py --data data/dolly_chat.txt --vocab_size 2000 --out tokenizer.json
python -u train.py --data data/dolly_chat.txt --out bpe_small_sSEED --block 256 --batch 64 --patience 4 --max_steps 5000 --seed SEED --tokenizer tokenizer.json --n_layer 4 --n_head 4 --n_embd 192 --dropout 0.3
python generate.py --chat --ckpt bpe_small_sSEED_best.pt > generate_bpe_small_sSEED.txt
python -u train.py --data data/dolly_chat.txt --out bpe_tiny_sSEED --block 256 --batch 64 --patience 4 --max_steps 5000 --seed SEED --tokenizer tokenizer.json --n_layer 3 --n_head 3 --n_embd 192 --dropout 0.4
python generate.py --chat --ckpt bpe_tiny_sSEED_best.pt > generate_bpe_tiny_sSEED.txt
```

To keep logs as the script does, add `2>&1 | tee logs/<name>.txt` to each `train.py` command (for example `logs/bpe_small_s1337.txt`).

## 3. Outputs

In the repo root, for each of the 6 runs (`bpe_small_s1337`, `bpe_small_s1`, `bpe_small_s2`, `bpe_tiny_s1337`, `bpe_tiny_s1`, `bpe_tiny_s2`):

- `<name>_results.json`: split sizes, loss, perplexity, `bits_per_char` (train/val/test), `unknown_chars` (characters the tokenizer dropped, per split), config.
- `<name>_best.pt` and `<name>_final.pt`: checkpoints (best validation loss, and last step).
- `logs/<name>.txt`: the training log; `logs/bpe_tokenizer.txt`: the tokenizer log.
- `generate_<name>.txt`: sample chat replies from the best checkpoint.
- `tokenizer.json`: the shared tokenizer.

## 4. How long

I have no timing for these exact runs. For scale: the original 10.9M-parameter character model took about 8 to 16 minutes per run on the same kind of GPU, per the existing logs in `logs/` (for example `logs/dolly_small.txt`). The BPE runs are smaller than that. There are 6 runs, so budget several multiples of one run; the time column in each log shows actual progress.

## 5. What to copy back

Copy these back into the repo on your main machine (same names, same folders):

- all `*_results.json` (all six `bpe_*` ones)
- the whole `logs/` folder
- all `generate_*.txt`
- `tokenizer.json`
- the single chosen `*_best.pt`

Choosing the checkpoint: open each `bpe_*_results.json` and read `bits_per_char`, then `val`. Pick the run with the lowest validation `bits_per_char` (lower is better, and it is comparable across tokenizers). Copy only that run's `<name>_best.pt`, for example `bpe_small_s1337_best.pt`. Do not pick by `test`: test is for the final report, not for selection.

Keep the whole `bpe_*` set of results and logs (all configs and seeds), not just the winner, so the seeds can be compared (mean and spread across seeds, not one lucky run). The other `.pt` files are large and can stay on the GPU machine; they are git-ignored.

Quick table of validation bits per character:

```bash
python -c "import json,glob; [print(f, json.load(open(f))['bits_per_char']['val']) for f in sorted(glob.glob('bpe_*_results.json'))]"
```
