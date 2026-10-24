"""Train a BPE tokenizer on the first 80% of characters of a text file."""
import argparse

from bpe import BPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dolly_chat.txt")
    ap.add_argument("--vocab_size", type=int, default=2000)
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    train_text = text[: int(len(text) * 0.8)]
    tok = BPE.train(train_text, args.vocab_size)
    tok.save(args.out)
    ratio = len(tok.encode(train_text)) / max(1, len(train_text))
    print(f"alphabet={len(tok.alphabet)} vocab={len(tok.vocab)} "
          f"tokens/char={ratio:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
