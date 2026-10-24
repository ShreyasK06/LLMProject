"""Runs tests/make_real_parity.py and web/test/check_real_parity.mjs together (skipped if node is not installed)."""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from bpe import BPE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")
CHECK = os.path.join(ROOT, "web", "test", "check_real_parity.mjs")
MAKE = os.path.join(ROOT, "tests", "make_real_parity.py")
SHAKESPEARE = os.path.join(ROOT, "data", "input.txt")
FIXTURE_TOK = os.path.join(ROOT, "web", "test", "fixtures", "tokenizer.json")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed; skipping the JS tokenizer parity check")


def make(tok_path, text_path, out, *extra):
    subprocess.run([sys.executable, MAKE, "--tokenizer", str(tok_path), "--text", str(text_path),
                    "--out", str(out), *extra], check=True, cwd=ROOT)
    return json.loads(open(out, encoding="utf-8").read())


def check(tok_path, parity):
    return subprocess.run([NODE, CHECK, str(tok_path), str(parity)], capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def shakespeare_tok(tmp_path_factory):
    with open(SHAKESPEARE, encoding="utf-8") as f:
        text = f.read()[:60000]
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    BPE.train(text, 400).save(path)
    return path


def test_fixture_tokenizer_on_shakespeare(tmp_path):
    out = tmp_path / "p.json"
    cases = make(FIXTURE_TOK, SHAKESPEARE, out, "--slices", "50", "--slice_chars", "300")
    assert len(cases) == 50 and all(len(c["text"]) == 300 for c in cases)
    assert sum(c["dropped"] for c in cases) > 0  # the chat fixture tokenizer lacks some Shakespeare characters
    r = check(FIXTURE_TOK, out)
    assert r.returncode == 0, r.stderr
    assert "real parity: 50 slices" in r.stdout


def test_shakespeare_tokenizer_with_unknown_chars(shakespeare_tok, tmp_path):
    text = tmp_path / "t.txt"
    with open(SHAKESPEARE, encoding="utf-8") as f:
        body = f.read()[60000:90000]
    text.write_text(body + "\ncafé 中文 \U0001F600 done\n", encoding="utf-8")
    out = tmp_path / "p.json"
    cases = make(shakespeare_tok, text, out)
    assert len(cases) == 200
    r = check(shakespeare_tok, out)
    assert r.returncode == 0, r.stderr


def test_deterministic_slices(tmp_path):
    a = make(FIXTURE_TOK, SHAKESPEARE, tmp_path / "a.json", "--slices", "5", "--seed", "3")
    b = make(FIXTURE_TOK, SHAKESPEARE, tmp_path / "b.json", "--slices", "5", "--seed", "3")
    assert a == b


def test_mismatch_exits_nonzero(tmp_path):
    out = tmp_path / "p.json"
    cases = make(FIXTURE_TOK, SHAKESPEARE, out, "--slices", "5")
    cases[2]["ids"][0] += 1
    bad_ids = tmp_path / "bad_ids.json"
    bad_ids.write_text(json.dumps(cases), encoding="utf-8")
    r = check(FIXTURE_TOK, bad_ids)
    assert r.returncode != 0 and "1/5 slices differ" in r.stderr

    cases[2]["ids"][0] -= 1
    cases[4]["dropped"] += 1
    bad_drop = tmp_path / "bad_drop.json"
    bad_drop.write_text(json.dumps(cases), encoding="utf-8")
    assert check(FIXTURE_TOK, bad_drop).returncode != 0
