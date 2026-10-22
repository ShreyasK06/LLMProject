#!/usr/bin/env bash
# Run 3 GPU configs sequentially; each early-stops on val loss (patience counted in eval intervals of 250 steps).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-$(command -v python3 || command -v python)}

run() { # name, extra flags...
  local name=$1; shift
  echo "=== $name ==="
  "$PY" -u train.py --out "$name" --patience 4 --max_steps 5000 "$@" 2>&1 | tee "logs/$name.txt"
}

run exp_a_baseline --n_layer 6 --n_head 6 --n_embd 384 --dropout 0.2
run exp_b_reg      --n_layer 6 --n_head 6 --n_embd 384 --dropout 0.3 --weight_decay 0.2
run exp_c_small    --n_layer 4 --n_head 4 --n_embd 256 --dropout 0.3
