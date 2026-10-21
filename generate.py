"""Load the trained model and generate text from several prompts."""
import sys

import torch

from train import GPT, decode, device, encode

PROMPTS = [
    "ROMEO:",
    "First Citizen:\nBefore we proceed",
    "To be, or not to be",
    "KING HENRY:\nMy lords,",
    "The",
]

model = GPT().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device)["model"])
model.eval()

prompts = sys.argv[1:] or PROMPTS
torch.manual_seed(0)
for p in prompts:
    out = model.generate(torch.tensor([encode(p)], device=device), 300)[0].tolist()
    print("=" * 60, f"\nPROMPT: {p!r}\n" + "-" * 60)
    print(decode(out))
