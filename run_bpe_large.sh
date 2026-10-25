#!/usr/bin/env bash
# Size-matched BPE run for the GPU machine: the dolly_large architecture (6 layers, 6 heads, width 384, weight decay
# 0.2) on the BPE tokenizer from run_bpe.sh, trained longer because the 5000-step BPE runs were still improving.
# Needs tokenizer.json (from run_bpe.sh) and data/dolly_chat.txt. Overrides (environment): STEPS (default 10000),
# SEEDS (default "1337 1"), DATA (default data/dolly_chat.txt).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results
PY=${PYTHON:-$(command -v python3 || command -v python)}
STEPS=${STEPS:-10000}
SEEDS=${SEEDS:-"1337 1"}
DATA=${DATA:-data/dolly_chat.txt}
[ -f tokenizer.json ] || { echo "tokenizer.json missing; run run_bpe.sh first" >&2; exit 1; }
[ -f "$DATA" ] || { echo "data file not found: $DATA" >&2; exit 1; }

for seed in $SEEDS; do
  name="bpe_large_s$seed"
  echo "=== $name ==="
  "$PY" -u train.py --data "$DATA" --out "$name" --block 256 --batch 64 --patience 4 --max_steps "$STEPS" \
    --seed "$seed" --tokenizer tokenizer.json --n_layer 6 --n_head 6 --n_embd 384 --dropout 0.3 \
    --weight_decay 0.2 2>&1 | tee "logs/$name.txt"
  "$PY" generate.py --chat --ckpt "${name}_best.pt" > "results/generate_${name}.txt"
done
