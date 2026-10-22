"""Load a trained checkpoint (default: best-val) and generate text from several prompts.

Plain mode continues each prompt as raw text. --chat wraps each prompt as a User/Assistant exchange (the format
prepare_dolly.py writes) and prints only the assistant's reply, which ends at the first blank line."""
import argparse

import torch

from train import GPT, device

PROMPTS = [
    "ROMEO:",
    "First Citizen:\nBefore we proceed",
    "To be, or not to be",
    "KING HENRY:\nMy lords,",
    "The",
]
CHAT_PROMPTS = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Can you give me some ideas for a birthday party?",
    "Explain what a computer is.",
    "Write a short poem about the sea.",
]
USER, BOT = "User", "Assistant"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="model_best.pt")
ap.add_argument("--chat", action="store_true", help="format prompts as User/Assistant turns and print the reply only")
ap.add_argument("--max_chars", type=int, default=300)
ap.add_argument("prompts", nargs="*")
args = ap.parse_args()

ckpt = torch.load(args.ckpt, map_location=device)
chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]  # characters outside the training vocabulary are dropped
decode = lambda ids: "".join(chars[i] for i in ids)

model = GPT(ckpt["config"], len(chars)).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

prompts = args.prompts or (CHAT_PROMPTS if args.chat else PROMPTS)
torch.manual_seed(0)
for p in prompts:
    framed = f"{USER}:\n{p}\n\n{BOT}:\n" if args.chat else p
    ids = encode(framed)
    out = model.generate(torch.tensor([ids], device=device), args.max_chars)[0].tolist()
    print("=" * 60, f"\nPROMPT: {p!r}\n" + "-" * 60)
    print(decode(out[len(ids):]).split("\n\n")[0] if args.chat else decode(out))
