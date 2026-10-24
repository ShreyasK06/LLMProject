#!/usr/bin/env bash
# BPE chat models for the GPU machine: train one tokenizer, then two configs x several seeds (each early-stops on
# val loss), then sample replies from each best checkpoint. Results JSON and samples go to results/, logs to logs/.
# Overrides (environment): STEPS (max_steps, default 5000), SEEDS (default "1337 1 2"),
# DATA (text file, default data/dolly_chat.txt, prepared automatically if missing), VOCAB (default 2000).
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
STEPS=${STEPS:-5000}
SEEDS=${SEEDS:-"1337 1 2"}
DATA=${DATA:-data/dolly_chat.txt}
VOCAB=${VOCAB:-2000}

if [ ! -f "$DATA" ]; then
  if [ "$DATA" = data/dolly_chat.txt ]; then "$PY" prepare_dolly.py
  else echo "data file not found: $DATA" >&2; exit 1; fi
fi

EVAL=()
[ "$STEPS" -lt 250 ] && EVAL=(--eval_every $((STEPS / 2)))

echo "=== tokenizer ==="
"$PY" train_tokenizer.py --data "$DATA" --vocab_size "$VOCAB" --out tokenizer.json 2>&1 | tee logs/bpe_tokenizer.txt

run() { # name, seed, extra flags...
  local name=$1 seed=$2; shift 2
  echo "=== $name ==="
  "$PY" -u train.py --data "$DATA" --out "$name" --block 256 --batch 64 --patience 4 --max_steps "$STEPS" \
    --seed "$seed" --tokenizer tokenizer.json ${EVAL[@]+"${EVAL[@]}"} "$@" 2>&1 | tee "logs/$name.txt"
  "$PY" generate.py --chat --ckpt "${name}_best.pt" > "results/generate_${name}.txt"
}

for seed in $SEEDS; do
  run "bpe_small_s$seed" "$seed" --n_layer 4 --n_head 4 --n_embd 192 --dropout 0.3
  run "bpe_tiny_s$seed" "$seed" --n_layer 3 --n_head 3 --n_embd 192 --dropout 0.4
done
