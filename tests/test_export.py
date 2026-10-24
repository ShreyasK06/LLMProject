import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import export_weights
import train
from bpe import BPE

CFG = {"block": 16, "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0}
TEXT = ("the cat sat on the mat. the dog ran to the park.\nUser: what is the capital of France?\n"
        "Assistant: Paris is nice in spring!\n") * 20


def bpe_ckpt(path):
    tok = BPE.train(TEXT, 60)
    torch.manual_seed(0)
    model = train.GPT(CFG, len(tok.vocab))
    torch.save({"model": model.state_dict(), "chars": None, "tokenizer": train.tokenizer_dict(tok),
                "config": CFG, "step": 7}, path)
    return tok


def char_ckpt(path, with_tokenizer_key=True):
    chars = sorted(set(TEXT))
    torch.manual_seed(0)
    model = train.GPT(CFG, len(chars))
    d = {"model": model.state_dict(), "chars": chars, "config": CFG, "step": 3}
    if with_tokenizer_key:
        d["tokenizer"] = None
    torch.save(d, path)


def run(ckpt, out, *extra):
    export_weights.main(["--ckpt", str(ckpt), "--out", str(out), *extra])
    return json.loads((out / "model.json").read_text())


def test_bpe_export(tmp_path):
    tok = bpe_ckpt(tmp_path / "m.pt")
    meta = run(tmp_path / "m.pt", tmp_path / "o")
    loaded = BPE.load(tmp_path / "o" / "tokenizer.json")
    assert loaded.vocab == tok.vocab and loaded.merges == tok.merges
    assert meta["tokenizer"] == "tokenizer.json" and meta["chars"] is None
    assert meta["dtype"] == "float32" and meta["step"] == 7
    assert all(t["name"] != "head.weight" for t in meta["tensors"])
    n = sum(t["length"] for t in meta["tensors"])
    assert (tmp_path / "o" / "weights.bin").stat().st_size == 4 * n


def test_fp16(tmp_path):
    bpe_ckpt(tmp_path / "m.pt")
    m32 = run(tmp_path / "m.pt", tmp_path / "a")
    m16 = run(tmp_path / "m.pt", tmp_path / "b", "--fp16")
    assert m16["dtype"] == "float16"
    assert [(t["name"], t["offset"], t["length"]) for t in m16["tensors"]] == \
           [(t["name"], t["offset"], t["length"]) for t in m32["tensors"]]
    n = sum(t["length"] for t in m16["tensors"])
    assert (tmp_path / "b" / "weights.bin").stat().st_size == 2 * n
    a = np.fromfile(tmp_path / "a" / "weights.bin", dtype="<f4")
    b = np.fromfile(tmp_path / "b" / "weights.bin", dtype="<f2").astype(np.float32)
    assert a.shape == b.shape and np.abs(a - b).max() < 1e-2


def test_char_export(tmp_path):
    char_ckpt(tmp_path / "c.pt")
    meta = run(tmp_path / "c.pt", tmp_path / "o")
    assert meta["chars"] == sorted(set(TEXT)) and meta["tokenizer"] is None and meta["dtype"] == "float32"
    assert not (tmp_path / "o" / "tokenizer.json").exists()


def test_legacy_checkpoint_without_tokenizer_key(tmp_path):
    char_ckpt(tmp_path / "c.pt", with_tokenizer_key=False)
    meta = run(tmp_path / "c.pt", tmp_path / "o")
    assert meta["tokenizer"] is None and meta["chars"]
