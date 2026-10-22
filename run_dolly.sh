#!/usr/bin/env bash
# Chat model: prepare Dolly 15k, train two configs from scratch (each early-stops on val loss), then sample replies.
# Block 512 so the model can see a whole question and answer; batch 32 keeps memory modest.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-$(command -v python3 || command -v python)}

[ -f data/dolly_chat.txt ] || "$PY" prepare_dolly.py

run() { # name, extra flags...
  local name=$1; shift
  echo "=== $name ==="
  "$PY" -u train.py --data data/dolly_chat.txt --out "$name" --block 512 --batch 32 --patience 4 --max_steps 5000 "$@" \
    2>&1 | tee "logs/$name.txt"
  "$PY" generate.py --chat --ckpt "${name}_best.pt" > "generate_${name}.txt"
}

run dolly_small --n_layer 4 --n_head 4 --n_embd 256 --dropout 0.3
run dolly_large --n_layer 6 --n_head 6 --n_embd 384 --dropout 0.3 --weight_decay 0.2
