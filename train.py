"""Train a small character-level GPT from scratch (random init, no pretrained weights)
on a text file (default TinyShakespeare; see prepare_dolly.py for chat data) using next-token prediction.

Data is split 80/10/10 (sequential) into train/val/test. Val drives checkpoint selection and
early stopping; test loss is reported once, at the end, for the best-val checkpoint."""
import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

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


splits = {}  # filled by load_data()


def load_data(path):
    """Read a text file, build the character vocabulary, fill `splits`, and return the vocabulary."""
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train, n_val = int(0.8 * len(data)), int(0.9 * len(data))
    splits.update(train=data[:n_train], val=data[n_train:n_val], test=data[n_val:])
    return chars


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
def estimate_loss(model, cfg, which=("train", "val"), iters=None):
    model.eval()
    iters = iters or cfg["eval_iters"]
    out = {}
    for s in which:
        if s == "train":  # random windows from the train split
            out[s] = sum(model(*get_batch(s, cfg["block"], cfg["batch"]))[1].item() for _ in range(iters)) / iters
        else:  # val/test: deterministic sweep over non-overlapping windows, so numbers are comparable across steps
            tot = n = 0
            for x, y in sweep_batches(s, cfg["block"], cfg["batch"]):
                tot += model(x, y)[1].item() * len(x)
                n += len(x)
            out[s] = tot / n
    model.train()
    return out


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * p))


def parse_args():
    d = default_config()
    ap = argparse.ArgumentParser(description="Train a char-level GPT from scratch on a text file.")
    ap.add_argument("--data", default="data/input.txt", help="training text (default: TinyShakespeare)")
    for k in ("n_layer", "n_head", "n_embd", "block", "batch", "max_steps", "seed", "eval_every", "eval_iters"):
        ap.add_argument(f"--{k}", type=int, default=d[k])
    for k in ("dropout", "lr", "min_lr", "weight_decay"):
        ap.add_argument(f"--{k}", type=float, default=d[k])
    ap.add_argument("--patience", type=int, default=d["patience"],
                    help="stop after this many eval intervals without val improvement (0 = off)")
    ap.add_argument("--out", default="model", help="checkpoint prefix: <out>_best.pt, <out>_final.pt")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = {**default_config(), **{k: v for k, v in vars(args).items() if k not in ("out", "data")}}
    assert cfg["n_embd"] % cfg["n_head"] == 0, "n_embd must be divisible by n_head"
    torch.manual_seed(cfg["seed"])
    out = args.out
    chars = load_data(args.data)
    vocab_size = len(chars)
    model = GPT(cfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} data={args.data} chars={sum(len(s) for s in splits.values())} vocab={vocab_size} "
          f"params={n_params / 1e6:.2f}M out={out}", flush=True)
    print("config:", json.dumps(cfg), flush=True)
    # decay only weight matrices; exempt biases, LayerNorm gains and the (tied) embeddings
    nodecay_ids = {id(p) for n, p in model.named_parameters() if p.ndim < 2 or n.startswith(("tok.", "pos."))}
    decay = [p for p in model.parameters() if id(p) not in nodecay_ids]
    nodecay = [p for p in model.parameters() if id(p) in nodecay_ids]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.95))

    def save(path, step):
        torch.save({"model": model.state_dict(), "chars": chars, "config": cfg, "step": step}, path)

    best_val, best_step, bad, t0 = float("inf"), 0, 0, time.time()
    for step in range(cfg["max_steps"] + 1):
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            l = estimate_loss(model, cfg)
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
    r = estimate_loss(model, cfg, ("train", "val", "test"), iters)
    ppl = {k: math.exp(v) for k, v in r.items()}
    print(f"best checkpoint (step {best_step}) | train {r['train']:.4f} | val {r['val']:.4f} | test {r['test']:.4f}"
          f" | ppl train {ppl['train']:.2f} val {ppl['val']:.2f} test {ppl['test']:.2f}")
    with open(f"{out}_results.json", "w") as f:
        json.dump({"data": args.data, "vocab_size": vocab_size, "config": cfg, "params": n_params,
                   "best_step": best_step, "last_step": step,
                   "train_loss": r["train"], "val_loss": r["val"], "test_loss": r["test"],
                   "perplexity": ppl, "eval_iters": iters}, f, indent=2)
    print(f"wrote {out}_results.json")


if __name__ == "__main__":
    main()
