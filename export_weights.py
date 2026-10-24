"""Export a trained checkpoint to a JS-readable format for the Vercel backend.

Usage: python export_weights.py --ckpt exp_c_small_best.pt [--out web/model] [--fp16]
Writes <out>/weights.bin (little-endian float32, or float16 with --fp16; tensors concatenated),
<out>/model.json (config, chars or tokenizer file name, dtype, tensor index) and, for BPE checkpoints,
<out>/tokenizer.json. Tensor offset/length in model.json are in elements, not bytes.
The tied output head shares tok.weight, so it is not exported separately."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

FP16_MAX = 65504.0  # largest finite float16; --fp16 refuses tensors outside +-FP16_MAX or with NaN/inf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="web/model")
    ap.add_argument("--fp16", action="store_true", help="store weights as little-endian float16")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype, np_dtype, width = ("float16", "<f2", 2) if args.fp16 else ("float32", "<f4", 4)

    tensors, chunks, offset = [], [], 0
    for name, t in ckpt["model"].items():
        if name == "head.weight":  # tied to tok.weight
            continue
        f = t.detach().float().contiguous().numpy()
        if args.fp16 and f.size and not (np.isfinite(f).all() and np.abs(f).max() <= FP16_MAX):
            bad = "non-finite values" if not np.isfinite(f).all() else f"absolute max {np.abs(f).max():g}"
            raise ValueError(f"Tensor {name} cannot be stored as float16: it has {bad} (float16 range is "
                             f"+-{FP16_MAX:g}). Export without --fp16.")
        a = f.astype(np_dtype)
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset, "length": a.size})
        chunks.append(a.tobytes())
        offset += a.size

    (out / "weights.bin").write_bytes(b"".join(chunks))
    tok = ckpt.get("tokenizer")
    if tok:
        (out / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    meta = {
        "config": {k: cfg[k] for k in ("block", "n_layer", "n_head", "n_embd")},
        "chars": None if tok else ckpt["chars"],
        "tokenizer": "tokenizer.json" if tok else None,
        "dtype": dtype,
        "step": ckpt.get("step"),
        "tensors": tensors,
    }
    (out / "model.json").write_text(json.dumps(meta))
    print(f"wrote {out / 'weights.bin'} ({offset * width / 1e6:.1f} MB {dtype}, {len(tensors)} tensors) and {out / 'model.json'}")


if __name__ == "__main__":
    main()
