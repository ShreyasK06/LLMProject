import json
import math
import os
import random
import string
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

import train
from bpe import BPE

TINY = ["--n_layer", "1", "--n_head", "2", "--n_embd", "16", "--block", "16", "--batch", "4",
        "--max_steps", "10", "--eval_every", "5", "--eval_iters", "2", "--dropout", "0.0"]

WORDS = ("the cat sat on a mat and the dog ran to the park while User asked Assistant "
         "what is the capital of France Paris is nice in spring").split()


def make_text(n=5000, seed=0):
    rng = random.Random(seed)
    parts, size = [], 0
    while size < n:
        line = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12)))
        line += rng.choice([".", "?", "!", ","]) + "\n"
        parts.append(line)
        size += len(line)
    return "".join(parts)[:n]


@pytest.fixture
def data_file(tmp_path):
    p = tmp_path / "text.txt"
    p.write_text(make_text(), encoding="utf-8")
    return p


@pytest.fixture
def tok_file(tmp_path, data_file):
    text = data_file.read_text(encoding="utf-8")
    tok = BPE.train(text[: int(0.8 * len(text))], 300)
    p = tmp_path / "tok.json"
    tok.save(str(p))
    return p


def run(tmp_path, data_file, name, extra=()):
    out = tmp_path / name
    train.main(["--data", str(data_file), "--out", str(out), "--results_dir", str(tmp_path / "results"),
                *TINY, *extra])
    with open(tmp_path / "results" / f"{name}_results.json") as f:
        return json.load(f)


def test_results_go_to_results_dir(tmp_path, data_file):
    run(tmp_path, data_file, "where")
    assert (tmp_path / "where_best.pt").exists() and (tmp_path / "where_final.pt").exists()
    assert not (tmp_path / "where_results.json").exists()
    assert (tmp_path / "results" / "where_results.json").exists()


def test_char_bpc_equals_loss_over_ln2(tmp_path, data_file):
    r = run(tmp_path, data_file, "char")
    assert r["tokenizer"] == "char"
    for s in ("train", "val", "test"):
        assert r["bits_per_char"][s] == pytest.approx(r[f"{s}_loss"] / math.log(2))
        assert r["tokens"][s] == r["chars"][s]


def test_bpe_run_writes_results(tmp_path, data_file, tok_file):
    r = run(tmp_path, data_file, "bpe", ["--tokenizer", str(tok_file)])
    text = data_file.read_text(encoding="utf-8")
    for key in ("tokenizer", "bits_per_char", "tokens", "chars"):
        assert key in r
    for key in ("bits_per_char", "tokens", "chars"):
        assert set(r[key]) == {"train", "val", "test"}
    assert r["tokenizer"] == str(tok_file)
    assert r["vocab_size"] == len(BPE.load(str(tok_file)).vocab) > 60  # merges were learned
    assert r["chars"]["train"] == len(text[: int(0.8 * len(text))])
    assert r["tokens"]["train"] < r["chars"]["train"]
    assert r["bits_per_char"]["test"] > 0
    # existing keys kept
    for key in ("data", "config", "params", "best_step", "last_step", "train_loss", "val_loss",
                "test_loss", "perplexity", "eval_iters"):
        assert key in r
    ckpt = torch.load(f"{tmp_path / 'bpe'}_best.pt", map_location="cpu")
    assert ckpt["chars"] is None
    assert ckpt["tokenizer"]["vocab"] == BPE.load(str(tok_file)).vocab


def test_same_split_points(tmp_path, data_file, tok_file):
    rc = run(tmp_path, data_file, "c")
    rb = run(tmp_path, data_file, "b", ["--tokenizer", str(tok_file)])
    assert rc["chars"] == rb["chars"]
    n = len(data_file.read_text(encoding="utf-8"))
    assert rc["chars"] == {"train": int(0.8 * n), "val": int(0.9 * n) - int(0.8 * n),
                           "test": n - int(0.9 * n)}
    ckpt = torch.load(f"{tmp_path / 'c'}_best.pt", map_location="cpu")
    assert ckpt["tokenizer"] is None and isinstance(ckpt["chars"], list)


@pytest.mark.parametrize("use_bpe", [False, True])
def test_generate_runs(tmp_path, use_bpe):
    # GPT.generate samples with top_k=40, so the vocabulary needs at least 40 tokens
    text = (string.ascii_letters + string.digits + "\n") * 5 + make_text()
    data_file = tmp_path / "gen.txt"
    data_file.write_text(text, encoding="utf-8")
    tok_file = tmp_path / "gen_tok.json"
    BPE.train(text[: int(0.8 * len(text))], 300).save(str(tok_file))
    run(tmp_path, data_file, "g", ["--tokenizer", str(tok_file)] if use_bpe else [])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for extra in ([], ["--chat"]):
        p = subprocess.run([sys.executable, os.path.join(root, "generate.py"), "--ckpt", f"{tmp_path / 'g'}_best.pt",
                            "--max_chars", "40", *extra, "the cat"],
                           capture_output=True, text=True, encoding="utf-8", cwd=root)
        assert p.returncode == 0, p.stderr
        assert "PROMPT: 'the cat'" in p.stdout


def test_bpe_bpc_matches_brute_force(data_file, tok_file):
    tok = BPE.load(str(tok_file))
    vocab, tok_chars = train.load_data(str(data_file), tok)
    assert vocab == tok.vocab
    assert tok_chars.tolist() == [tok.char_len(i) for i in range(len(tok.vocab))]
    cfg = {**train.default_config(), "n_layer": 1, "n_head": 2, "n_embd": 16, "block": 8, "batch": 3,
           "dropout": 0.0, "eval_iters": 2}
    torch.manual_seed(0)
    model = train.GPT(cfg, len(vocab)).to(train.device)
    loss, bpc = train.estimate_loss(model, cfg, ("val", "test"), tok_chars=tok_chars)
    for s in ("val", "test"):
        d = train.splits[s].tolist()
        nll = n_tok = n_chr = 0
        B = cfg["block"]
        for j in range(0, len(d) - B - 1, B):
            x = torch.tensor([d[j : j + B]], device=train.device)
            y = d[j + 1 : j + B + 1]
            with torch.no_grad():
                logp = F.log_softmax(model(x)[0][0].double(), dim=-1)
            nll += -sum(logp[t, y[t]].item() for t in range(B))
            n_tok += B
            n_chr += sum(len(tok.vocab[i]) for i in y)
        assert loss[s] == pytest.approx(nll / n_tok, rel=1e-4)
        assert bpc[s] == pytest.approx(nll / (n_chr * math.log(2)), rel=1e-4)
        assert n_chr > n_tok  # BPE tokens cover more than one character on average


def test_unknown_chars_reported(tmp_path):
    text = make_text(4000)
    cut = int(0.8 * len(text))
    pos = int(0.9 * len(text)) + 1  # inside the test split
    text = text[:pos] + "é" + text[pos + 1 :]  # a character that appears only in the last 20%
    data_file = tmp_path / "unk.txt"
    data_file.write_text(text, encoding="utf-8")
    tok_file = tmp_path / "unk_tok.json"
    BPE.train(text[:cut], 300).save(str(tok_file))
    r = run(tmp_path, data_file, "u", ["--tokenizer", str(tok_file)])
    assert set(r["unknown_chars"]) == {"train", "val", "test"}
    assert r["unknown_chars"]["train"] == 0
    assert r["unknown_chars"]["test"] >= 1
    rc = run(tmp_path, data_file, "uc")
    assert rc["unknown_chars"] == {"train": 0, "val": 0, "test": 0}
