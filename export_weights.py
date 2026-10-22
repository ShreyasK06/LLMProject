"""Export a trained checkpoint to a JS-readable format for the Vercel backend.

Usage: python export_weights.py --ckpt exp_c_small_best.pt [--out web/model]
Writes <out>/weights.bin (little-endian float32, tensors concatenated) and <out>/model.json (config, vocab, tensor index).
The tied output head shares tok.weight, so it is not exported separately."""
import argparse
import json
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="web/model")
args = ap.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
cfg = ckpt["config"]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

tensors, chunks, offset = [], [], 0
for name, t in ckpt["model"].items():
    if name == "head.weight":  # tied to tok.weight
        continue
    a = t.detach().float().contiguous().numpy().astype("<f4")
    tensors.append({"name": name, "shape": list(a.shape), "offset": offset, "length": a.size})
    chunks.append(a.tobytes())
    offset += a.size

(out / "weights.bin").write_bytes(b"".join(chunks))
meta = {
    "config": {k: cfg[k] for k in ("block", "n_layer", "n_head", "n_embd")},
    "chars": ckpt["chars"],
    "step": ckpt.get("step"),
    "tensors": tensors,
}
(out / "model.json").write_text(json.dumps(meta))
print(f"wrote {out / 'weights.bin'} ({offset * 4 / 1e6:.1f} MB, {len(tensors)} tensors) and {out / 'model.json'}")
