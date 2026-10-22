"""Turn Databricks Dolly 15k (human-written instruction/response pairs, CC BY-SA 3.0) into chat text for train.py.

    python prepare_dolly.py                      # downloads the ~13 MB jsonl, writes data/dolly_chat.txt
    python prepare_dolly.py --input local.jsonl  # use a file you already have

Each kept example becomes
    User:
    <instruction>

    Assistant:
    <response>

and examples are concatenated, so a blank line always ends a speech. Choices made for a tiny character-level model:
  * only examples without a context passage (the others need a document that isn't in the prompt);
  * whitespace collapsed to single spaces, so "\\n\\n" appears only between speeches;
  * ASCII only (common typographic punctuation is mapped, anything else drops the example) to keep the vocabulary small;
  * long responses are cut at a sentence end, so replies stay short enough for the model's context window;
  * examples are shuffled (fixed seed) because train.py splits the file sequentially into train/val/test."""
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                            "…": "...", " ": " "})


def clean(s):
    s = s.translate(TYPOGRAPHY)
    s = re.sub(r"\[\d+\]", "", s)  # Wikipedia-style citation markers
    return re.sub(r"\s+", " ", s).strip()


def printable_ascii(s):
    return all(32 <= ord(c) < 127 for c in s)


def trim_to_sentence(s, limit):
    """Return s, cut at the last sentence end within `limit` characters; None if there is no such end."""
    if len(s) <= limit:
        return s
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", s[: limit + 1]) if m.end() <= limit]
    return s[: ends[-1]] if ends else None


def convert(rows, max_instruction, max_response, seed):
    """Return (examples, stats). `rows` are dicts with instruction/context/response."""
    stats = dict(rows=0, has_context=0, empty=0, instruction_too_long=0, no_sentence_end=0, non_ascii=0, duplicate=0)
    seen, examples = set(), []
    for r in rows:
        stats["rows"] += 1
        if clean(r.get("context", "")):
            stats["has_context"] += 1
            continue
        instruction, response = clean(r.get("instruction", "")), clean(r.get("response", ""))
        if not instruction or not response:
            stats["empty"] += 1
            continue
        if not (printable_ascii(instruction) and printable_ascii(response)):
            stats["non_ascii"] += 1
            continue
        if len(instruction) > max_instruction:
            stats["instruction_too_long"] += 1
            continue
        response = trim_to_sentence(response, max_response)
        if response is None:
            stats["no_sentence_end"] += 1
            continue
        if (instruction, response) in seen:
            stats["duplicate"] += 1
            continue
        seen.add((instruction, response))
        examples.append(f"User:\n{instruction}\n\nAssistant:\n{response}\n\n")
    random.Random(seed).shuffle(examples)
    stats["kept"] = len(examples)
    return examples, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", help="local Dolly jsonl (default: download to data/dolly_raw.jsonl)")
    ap.add_argument("--out", default="data/dolly_chat.txt")
    ap.add_argument("--max_instruction", type=int, default=300)
    ap.add_argument("--max_response", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.input) if args.input else Path("data/dolly_raw.jsonl")
    if not src.exists():
        if args.input:
            raise SystemExit(f"{src} not found")
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, src)
    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    examples, stats = convert(rows, args.max_instruction, args.max_response, args.seed)
    text = "".join(examples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:  # no \r\n on Windows: "\n\n" must stay the speech end
        f.write(text)
    print(json.dumps(stats))
    print(f"wrote {out}: {len(examples)} examples, {len(text):,} characters, {len(set(text))}-character vocabulary")


if __name__ == "__main__":
    main()
