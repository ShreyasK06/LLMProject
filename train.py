"""Train a small GPT from scratch (random init, no pretrained weights) on a text file (default
TinyShakespeare; see prepare_dolly.py for chat data) using next-token prediction. Tokens are characters by
default, or BPE tokens with --tokenizer (see train_tokenizer.py).

The raw text is cut 80/10/10 (sequential) at character offsets into train/val/test and each part is tokenized
separately, so character and BPE runs use the same split points. Val drives checkpoint selection and early
stopping; test loss is reported once, at the end, for the best-val checkpoint. Besides loss per token, bits per
character (summed NLL / (characters covered by the target tokens * ln 2)) is reported so that runs with
different tokenizers can be compared."""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe import BPE

device = "cuda" if torch.cuda.is_available() else "cpu"


def default_config():
    """~10.7M params on GPU, a ~0.8M-param config that still trains on CPU."""
    if device == "cuda":
        cfg = dict(block=256, batch=64, n_layer=6, n_head=6, n_embd=384, dropout=0.2,
                   max_steps=5000, lr=1e-3, min_lr=1e-4)
    else:
        cfg = dict(block=128, batch=32, n_layer=4, n_head=4, n_embd=128, dropout=0.1,
                   max_steps=3000, lr=3e-3, min_lr=3e-4)
    cfg.update(weight_decay=0.1, patience=0, eval_every=250, eval_iters=40, warmup=100, seed=1337)
    return cfg


splits = {}  # id tensors per split, filled by load_data()
split_chars = {}  # raw characters per split, filled by load_data()
split_unknown = {}  # raw characters per split outside the tokenizer's alphabet (dropped by encode), filled by load_data()


def tokenizer_dict(tok):
    """Inline JSON form of a BPE tokenizer (same fields as BPE.save)."""
    return {"version": 1, "alphabet": tok.alphabet, "merges": [list(m) for m in tok.merges], "vocab": tok.vocab}


def load_data(path, tokenizer=None):
    """Read a text file, cut it 80/10/10 at character offsets, tokenize each part separately and fill `splits`
    and `split_chars`. `tokenizer` is None (character ids), a BPE instance or a path to a saved BPE.
    Returns (vocab, tok_chars): the token strings and a LongTensor with the character length of each token."""
    text = open(path, encoding="utf-8").read()
    n_train, n_val = int(0.8 * len(text)), int(0.9 * len(text))
    parts = dict(train=text[:n_train], val=text[n_train:n_val], test=text[n_val:])
    if tokenizer is None:
        vocab = sorted(set(text))
        stoi = {c: i for i, c in enumerate(vocab)}
        encode = lambda s: [stoi[c] for c in s]
        unknown = {k: 0 for k in parts}
    else:
        tok = BPE.load(tokenizer) if isinstance(tokenizer, str) else tokenizer
        vocab, encode = tok.vocab, tok.encode
        unknown = {k: tok.unknown_chars(v) for k, v in parts.items()}
    splits.update({k: torch.tensor(encode(v), dtype=torch.long) for k, v in parts.items()})
    split_chars.update({k: len(v) for k, v in parts.items()})
    split_unknown.update(unknown)
    tok_chars = torch.tensor([len(t) for t in vocab], dtype=torch.long)
    return vocab, tok_chars


def get_batch(split, block, batch):
    d = splits[split]
    ix = torch.randint(len(d) - block - 1, (batch,))
    x = torch.stack([d[i : i + block] for i in ix])
    y = torch.stack([d[i + 1 : i + block + 1] for i in ix])
    return x.to(device), y.to(device)


def sweep_batches(split, block, batch):
    d = splits[split]
    starts = torch.arange(0, len(d) - block - 1, block)
    for i in range(0, len(starts), batch):
        ix = starts[i : i + batch]
        yield (torch.stack([d[j : j + block] for j in ix]).to(device),
               torch.stack([d[j + 1 : j + block + 1] for j in ix]).to(device))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e = cfg["n_embd"]
        self.n_head, self.p = cfg["n_head"], cfg["dropout"]
        self.ln1 = nn.LayerNorm(e)
        self.qkv = nn.Linear(e, 3 * e)
        self.proj = nn.Linear(e, e)
        self.ln2 = nn.LayerNorm(e)
        self.mlp = nn.Sequential(nn.Linear(e, 4 * e), nn.GELU(), nn.Linear(4 * e, e), nn.Dropout(self.p))
        self.drop = nn.Dropout(self.p)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).contiguous().view(B, T, C)))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.block = cfg["block"]
        self.tok = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos = nn.Embedding(cfg["block"], cfg["n_embd"])
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block :])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, cfg, which=("train", "val"), iters=None, tok_chars=None):
    """Returns (loss, bpc): mean nats per token and bits per character for each split in `which`.
    bpc = summed NLL / (characters covered by the target tokens * ln 2), where `tok_chars[id]` is the character
    length of token `id` (None = one character per token)."""
    model.eval()
    iters = iters or cfg["eval_iters"]
    if tok_chars is not None:
        tok_chars = tok_chars.to(device)
    loss, bpc = {}, {}
    for s in which:
        if s == "train":  # random windows from the train split
            batches = (get_batch(s, cfg["block"], cfg["batch"]) for _ in range(iters))
        else:  # val/test: deterministic sweep over non-overlapping windows, so numbers are comparable across steps
            batches = sweep_batches(s, cfg["block"], cfg["batch"])
        nll = n_tok = n_chr = 0
        for x, y in batches:
            logits, _ = model(x)
            nll += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
            n_tok += y.numel()
            n_chr += y.numel() if tok_chars is None else tok_chars[y].sum().item()
        loss[s] = nll / n_tok
        bpc[s] = nll / (n_chr * math.log(2))
    model.train()
    return loss, bpc


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * p))


def parse_args(argv=None):
    d = default_config()
    ap = argparse.ArgumentParser(description="Train a char-level or BPE GPT from scratch on a text file.")
    ap.add_argument("--data", default="data/input.txt", help="training text (default: TinyShakespeare)")
    ap.add_argument("--tokenizer", default=None,
                    help="BPE tokenizer JSON from train_tokenizer.py (default: character tokens)")
    for k in ("n_layer", "n_head", "n_embd", "block", "batch", "max_steps", "seed", "eval_every", "eval_iters"):
        ap.add_argument(f"--{k}", type=int, default=d[k])
    for k in ("dropout", "lr", "min_lr", "weight_decay"):
        ap.add_argument(f"--{k}", type=float, default=d[k])
    ap.add_argument("--patience", type=int, default=d["patience"],
                    help="stop after this many eval intervals without val improvement (0 = off)")
    ap.add_argument("--out", default="model", help="checkpoint prefix: <out>_best.pt, <out>_final.pt")
    ap.add_argument("--results_dir", default="results",
                    help="directory for the results JSON: <results_dir>/<basename of out>_results.json")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    not_cfg = ("out", "data", "tokenizer", "results_dir")
    cfg = {**default_config(), **{k: v for k, v in vars(args).items() if k not in not_cfg}}
    assert cfg["n_embd"] % cfg["n_head"] == 0, "n_embd must be divisible by n_head"
    torch.manual_seed(cfg["seed"])
    out = args.out
    tok = BPE.load(args.tokenizer) if args.tokenizer else None
    vocab, tok_chars = load_data(args.data, tok)
    vocab_size = len(vocab)
    model = GPT(cfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} data={args.data} tokenizer={args.tokenizer or 'char'} "
          f"chars={sum(split_chars.values())} tokens={sum(len(s) for s in splits.values())} vocab={vocab_size} "
          f"params={n_params / 1e6:.2f}M out={out}", flush=True)
    print("config:", json.dumps(cfg), flush=True)
    # decay only weight matrices; exempt biases, LayerNorm gains and the (tied) embeddings
    nodecay_ids = {id(p) for n, p in model.named_parameters() if p.ndim < 2 or n.startswith(("tok.", "pos."))}
    decay = [p for p in model.parameters() if id(p) not in nodecay_ids]
    nodecay = [p for p in model.parameters() if id(p) in nodecay_ids]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.95))

    def save(path, step):
        torch.save({"model": model.state_dict(), "chars": None if tok else vocab,
                    "tokenizer": tokenizer_dict(tok) if tok else None, "config": cfg, "step": step}, path)

    best_val, best_step, bad, t0 = float("inf"), 0, 0, time.time()
    for step in range(cfg["max_steps"] + 1):
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            l, _ = estimate_loss(model, cfg, tok_chars=tok_chars)
            tag = ""
            if l["val"] < best_val:
                best_val, best_step, bad = l["val"], step, 0
                save(f"{out}_best.pt", step)
                tag = " *"
            else:
                bad += 1
            print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time() - t0:.0f}s{tag}",
                  flush=True)
            if cfg["patience"] and bad >= cfg["patience"]:
                print(f"early stop at step {step}: no val improvement for {bad} evals (best step {best_step})")
                break
        if step == cfg["max_steps"]:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        _, loss = model(*get_batch("train", cfg["block"], cfg["batch"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    save(f"{out}_final.pt", step)
    print(f"saved {out}_final.pt (last step {step})")

    # report on the best-val checkpoint; test is touched only here, once
    model.load_state_dict(torch.load(f"{out}_best.pt", map_location=device)["model"])
    iters = max(200, cfg["eval_iters"])
    r, bpc = estimate_loss(model, cfg, ("train", "val", "test"), iters, tok_chars=tok_chars)
    ppl = {k: math.exp(v) for k, v in r.items()}
    print(f"best checkpoint (step {best_step}) | train {r['train']:.4f} | val {r['val']:.4f} | test {r['test']:.4f}"
          f" | ppl train {ppl['train']:.2f} val {ppl['val']:.2f} test {ppl['test']:.2f}"
          f" | bpc train {bpc['train']:.4f} val {bpc['val']:.4f} test {bpc['test']:.4f}")
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f"{os.path.basename(out)}_results.json")
    with open(results_path, "w") as f:
        json.dump({"data": args.data, "vocab_size": vocab_size, "config": cfg, "params": n_params,
                   "best_step": best_step, "last_step": step,
                   "train_loss": r["train"], "val_loss": r["val"], "test_loss": r["test"],
                   "perplexity": ppl, "eval_iters": iters,
                   "tokenizer": args.tokenizer or "char", "bits_per_char": bpc,
                   "tokens": {k: len(v) for k, v in splits.items()}, "chars": dict(split_chars),
                   "unknown_chars": dict(split_unknown)}, f, indent=2)
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
