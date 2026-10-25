# GPU runbook: BPE chat models

This trains two small BPE chat models (`bpe_small`, `bpe_tiny`) with three seeds each (1337, 1, 2) on Dolly 15k, using one shared tokenizer. It needs a CUDA GPU and internet (once, to download Dolly).

## 1. Set up

These steps clone from GitHub (the code is on `main`). Without git access, copy the repo folder to the GPU machine instead (a zip or `scp`; `.pt` files and `data/dolly_chat.txt` can be left out) and start at `python -m venv .venv`.

```bash
git clone https://github.com/ShreyasK06/LLMProject.git LLMProject
cd LLMProject
git checkout bpe-polish        # if the branch was pushed but not merged; skip this line once it is merged
python -m venv .venv
source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate ; PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt   # requirements-dev.txt adds pytest, for python -m pytest tests
```

`requirements.txt` lists `torch>=2.0` and `numpy`; on many machines that installs a CPU-only torch wheel. Installing the torch wheel that matches your CUDA driver (see pytorch.org "Get Started") is your responsibility. Check it before starting:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## 2. Run everything

```bash
bash run_bpe.sh
```

On Windows, Git Bash works. The script uses `$PYTHON` if set, else the active venv's interpreter, else `python3` or `python`; if it picks the wrong one, `PYTHON=python bash run_bpe.sh` works. If a script fails with `$'\r': command not found`, its line endings were converted to CRLF; fix with `sed -i 's/\r$//' run_bpe.sh`. `run_bpe.sh` does these steps in order:

1. If `data/dolly_chat.txt` is missing, runs `prepare_dolly.py` (downloads Dolly 15k).
2. Trains the tokenizer (`train_tokenizer.py`, vocab 2000, first 80% of the text) and writes `tokenizer.json`. **This step can take 10 to 20 minutes on the real data** (an estimate from timing synthetic text, not measured on Dolly) and uses only the CPU. It prints `merges N/total` every 100 merges and the elapsed seconds at the end, so it is not stuck.
3. For each seed, trains `bpe_small_s<seed>` (4 layers, 4 heads, width 192, block 256, dropout 0.3) and `bpe_tiny_s<seed>` (3 layers, 3 heads, width 192, block 256, dropout 0.4), both with batch 64, patience 4, 5000 max steps (early stopping on validation loss).
4. Generates sample replies from each best checkpoint.

Optional overrides: `STEPS=5000`, `SEEDS="1337 1 2"`, `DATA=data/dolly_chat.txt`, `VOCAB=2000`. For example, `SEEDS="1337" bash run_bpe.sh` runs a single seed first.

### Without bash

Run the same steps by hand from the repo root (use `python3` if that is your interpreter name). Replace `SEED` with each of 1337, 1 and 2 in turn:

```bash
mkdir -p logs results
python prepare_dolly.py                     # only if data/dolly_chat.txt does not exist
python train_tokenizer.py --data data/dolly_chat.txt --vocab_size 2000 --out tokenizer.json
python -u train.py --data data/dolly_chat.txt --out bpe_small_sSEED --block 256 --batch 64 --patience 4 --max_steps 5000 --seed SEED --tokenizer tokenizer.json --n_layer 4 --n_head 4 --n_embd 192 --dropout 0.3
python generate.py --chat --ckpt bpe_small_sSEED_best.pt > results/generate_bpe_small_sSEED.txt
python -u train.py --data data/dolly_chat.txt --out bpe_tiny_sSEED --block 256 --batch 64 --patience 4 --max_steps 5000 --seed SEED --tokenizer tokenizer.json --n_layer 3 --n_head 3 --n_embd 192 --dropout 0.4
python generate.py --chat --ckpt bpe_tiny_sSEED_best.pt > results/generate_bpe_tiny_sSEED.txt
```

To keep logs as the script does, add `2>&1 | tee logs/<name>.txt` to each `train.py` command (for example `logs/bpe_small_s1337.txt`).

## 3. Outputs

For each of the 6 runs (`bpe_small_s1337`, `bpe_small_s1`, `bpe_small_s2`, `bpe_tiny_s1337`, `bpe_tiny_s1`, `bpe_tiny_s2`):

- `results/<name>_results.json`: split sizes, loss, perplexity, `bits_per_char` (train/val/test), `unknown_chars` (characters the tokenizer dropped, per split), config.
- `<name>_best.pt` and `<name>_final.pt` (repo root): checkpoints (best validation loss, and last step).
- `logs/<name>.txt`: the training log; `logs/bpe_tokenizer.txt`: the tokenizer log.
- `results/generate_<name>.txt`: sample chat replies from the best checkpoint.
- `tokenizer.json` (repo root): the shared tokenizer.

## 4. How long

I have no timing for these exact runs. For scale, the existing logs in `logs/` show: the 10.88M-parameter model (`logs/dolly_large.txt`) took about 17 minutes (1018 s) for 5000 steps; the 3.3M-parameter model (`logs/dolly_small.txt`) took about 6 minutes (362 s) for 5000 steps, and `logs/exp_c_small.txt` (3.24M) about 5.5 minutes (325 s). The earlier 10.77M-parameter runs early-stopped at 2500 steps (454 s) and 3000 steps (552 s). The logs record only `device=cuda`, not the GPU model, so I cannot say how your GPU compares. The BPE runs are expected (not measured) to fall in the smaller-model range, roughly 5.5 to 6 minutes per 5000-step run, because they have fewer parameters. There are 6 runs; the time column in each log shows actual progress.

## 5. What to copy back

Copy these back into the repo on your main machine (same names, same folders):

- the whole `results/` folder (all six `bpe_*_results.json` and `generate_bpe_*.txt`)
- the whole `logs/` folder
- `tokenizer.json`
- the single chosen `*_best.pt`

Choosing the checkpoint: open each `results/bpe_*_results.json` and read `bits_per_char`, then `val`. Pick the run with the lowest validation `bits_per_char` (lower is better, and it is comparable across tokenizers). Copy only that run's `<name>_best.pt`, for example `bpe_small_s1337_best.pt`. Do not pick by `test`: test is for the final report, not for selection.

Keep the whole `bpe_*` set of results and logs (all configs and seeds), not just the winner, so the seeds can be compared (mean and spread across seeds, not one lucky run). The other `.pt` files are large and can stay on the GPU machine; they are git-ignored.

Quick table of validation bits per character:

```bash
python -c "import json,glob; [print(f, json.load(open(f))['bits_per_char']['val']) for f in sorted(glob.glob('results/bpe_*_results.json'))]"
```

## 6. Check the JavaScript tokenizer before deploying

Before swapping the BPE model into `web/model`, check that the JavaScript port encodes the real `tokenizer.json` exactly like Python (needs Node and `data/dolly_chat.txt`; run `python prepare_dolly.py` first if it is missing):

```bash
python tests/make_real_parity.py --tokenizer tokenizer.json --text data/dolly_chat.txt --out parity.json
node web/test/check_real_parity.mjs tokenizer.json parity.json   # must print "JS matches Python" and exit 0
```

After export, the server also refuses a `tokenizer.json` whose vocabulary size does not match the weights.
