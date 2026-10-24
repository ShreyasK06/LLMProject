#!/usr/bin/env bash
# Chat model: prepare Dolly 15k, train two configs from scratch (each early-stops on val loss), then sample replies.
# Block 512 so the model can see a whole question and answer; batch 32 keeps memory modest.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results
# Python: $PYTHON if set, else the active venv's interpreter, else python3 (or python when python3 is missing or is
# the Windows Store stub in WindowsApps).
pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    for p in "$VIRTUAL_ENV/Scripts/python.exe" "$VIRTUAL_ENV/Scripts/python" "$VIRTUAL_ENV/bin/python"; do
      if [ -x "$p" ]; then echo "$p"; return; fi
    done
  fi
  local p3
  p3=$(command -v python3 || true)
  case "$p3" in
    "" | */WindowsApps/*) command -v python || echo "$p3" ;;
    *) echo "$p3" ;;
  esac
}
PY=$(pick_python)
[ -n "$PY" ] || { echo "no python found; set PYTHON=/path/to/python" >&2; exit 1; }

[ -f data/dolly_chat.txt ] || "$PY" prepare_dolly.py

run() { # name, extra flags...
  local name=$1; shift
  echo "=== $name ==="
  "$PY" -u train.py --data data/dolly_chat.txt --out "$name" --block 512 --batch 32 --patience 4 --max_steps 5000 "$@" \
    2>&1 | tee "logs/$name.txt"
  "$PY" generate.py --chat --ckpt "${name}_best.pt" > "results/generate_${name}.txt"
}

run dolly_small --n_layer 4 --n_head 4 --n_embd 256 --dropout 0.3
run dolly_large --n_layer 6 --n_head 6 --n_embd 384 --dropout 0.3 --weight_decay 0.2
