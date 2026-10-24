"""Write Python BPE encodings of random slices of a text file, for checking the JS port on a real tokenizer.

Usage: python tests/make_real_parity.py --tokenizer tokenizer.json --text data/dolly_chat.txt --out parity.json
Then:  node web/test/check_real_parity.mjs tokenizer.json parity.json
Output: JSON list of {"text", "ids", "dropped"}, one per slice; slices are chosen deterministically from --seed.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe import BPE  # noqa: E402


def make_cases(tok, text, slices=200, slice_chars=500, seed=0):
    rng = random.Random(seed)
    cases = []
    for _ in range(slices):
        start = rng.randrange(max(1, len(text) - slice_chars + 1))
        s = text[start:start + slice_chars]
        cases.append({"text": s, "ids": tok.encode(s), "dropped": tok.unknown_chars(s)})
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", type=int, default=200)
    ap.add_argument("--slice_chars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tok = BPE.load(args.tokenizer)
    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    cases = make_cases(tok, text, args.slices, args.slice_chars, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    print(f"wrote {len(cases)} slices ({sum(len(c['ids']) for c in cases)} tokens) -> {args.out}")


if __name__ == "__main__":
    main()
