"""Load a trained checkpoint (default: best-val) and generate text from several prompts."""
import argparse

import torch

from train import GPT, decode, device, encode

PROMPTS = [
    "ROMEO:",
    "First Citizen:\nBefore we proceed",
    "To be, or not to be",
    "KING HENRY:\nMy lords,",
    "The",
]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="model_best.pt")
ap.add_argument("prompts", nargs="*")
args = ap.parse_args()

ckpt = torch.load(args.ckpt, map_location=device)
model = GPT(ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

prompts = args.prompts or PROMPTS
torch.manual_seed(0)
for p in prompts:
    out = model.generate(torch.tensor([encode(p)], device=device), 300)[0].tolist()
    print("=" * 60, f"\nPROMPT: {p!r}\n" + "-" * 60)
    print(decode(out))
