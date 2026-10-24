"""Train a BPE tokenizer on the first 80% of characters of a text file."""
import argparse
import time

from bpe import BPE


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dolly_chat.txt")
    ap.add_argument("--vocab_size", type=int, default=2000)
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args(argv)
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    train_text = text[: int(len(text) * 0.8)]
    start = time.time()

    def progress(done, total):
        print(f"merges {done}/{total} ({time.time() - start:.0f}s)", flush=True)

    tok = BPE.train(train_text, args.vocab_size, progress=progress)
    tok.save(args.out)
    ratio = len(tok.encode(train_text)) / max(1, len(train_text))
    print(f"alphabet={len(tok.alphabet)} vocab={len(tok.vocab)} "
          f"tokens/char={ratio:.3f} -> {args.out}", flush=True)
    print(f"elapsed {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
