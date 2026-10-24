import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from bpe import BPE, pretokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAGRAPH = (
    "User:\nWhat is the capital of France?\n\n"
    "Assistant:\nThe capital of France is Paris. It is known for the Eiffel Tower, "
    "the Louvre, and many wonderful cafes along the river.\n\n"
    "User:\nCan you tell me something about the history of the city?\n\n"
    "Assistant:\nParis has been a centre of art, science and politics for more than "
    "2000 years, and there are many stories that the people of the city like to tell.\n\n"
)
TRAIN_TEXT = PARAGRAPH * 8 + "Numbers 12345 and punctuation!?;, are  here too.\n  Leading spaces and  doubles.\n"


@pytest.fixture(scope="module")
def tok():
    return BPE.train(TRAIN_TEXT, 300)


def test_pretokenize_roundtrip():
    samples = ["", "a  b\n\nc", "Hello, world!", " leading", "trailing ", "x\ty  z\n \n",
               "Numbers 12345 and punctuation!?;,", "  two  spaces\n\n\nthree"]
    for s in samples:
        assert "".join(pretokenize(s)) == s
    assert pretokenize("a  b\n\nc") == ["a", " ", " b", "\n\n", "c"]
    assert pretokenize("hi!?  ok") == ["hi", "!?", " ", " ok"]
    assert pretokenize("end  \nx") == ["end", "  ", "\n", "x"]


def test_encode_decode_roundtrip(tok):
    for s in ["Hello there", "The capital of France is Paris.", "  spaced\n\nout  ", "12345!?;,"]:
        s2 = "".join(c for c in s if c in set(tok.alphabet))
        assert tok.decode(tok.encode(s2)) == s2
    s = "The capital of France is Paris."
    assert tok.decode(tok.encode(s)) == s


def test_empty_and_newlines(tok):
    assert tok.encode("") == []
    assert tok.decode(tok.encode("\n\n\n")) == "\n\n\n"


def test_unknown_chars_dropped(tok):
    assert tok.encode("a\u00e9\u4e2db") == tok.encode("ab")
    assert tok.unknown_chars("a\u00e9\u4e2db") == 2


def test_merge_order_deterministic():
    a = BPE.train(TRAIN_TEXT, 300)
    b = BPE.train(TRAIN_TEXT, 300)
    assert a.merges == b.merges
    # tie-break: every pair occurs once, so the lowest (a, b) pair merges first
    t = BPE.train("dc ba", 6)
    ids = {c: i for i, c in enumerate(t.alphabet)}
    assert t.merges[0] == (ids[" "], ids["b"])
    assert t.vocab[len(t.alphabet)] == " b"


def test_never_merges_across_pieces(tok):
    for t in tok.vocab:
        assert len(pretokenize(t)) == 1, repr(t)
        if "\n" in t:
            assert set(t) == {"\n"}
        if " " in t and any(c.isalpha() for c in t):
            assert t[0] == " " and t[1:].isalnum()


def test_save_load_identical(tok, tmp_path):
    p = str(tmp_path / "tok.json")
    tok.save(p)
    other = BPE.load(p)
    assert other.vocab == tok.vocab and other.merges == tok.merges
    s = PARAGRAPH + "  odd  spacing\n"
    assert other.encode(s) == tok.encode(s)


def test_char_len(tok):
    assert all(tok.char_len(i) == len(tok.vocab[i]) for i in range(len(tok.vocab)))


def test_compression(tok):
    text = PARAGRAPH * 3
    assert len(tok.encode(text)) / len(text) < 0.5


def test_writes_parity_fixture(tok):
    cases = [
        "",
        "\n",
        "\n\n",
        "User:\nHello there\n\nAssistant:\n",
        "  leading and  double spaces",
        "Numbers 12345 and punctuation!?;,",
        (PARAGRAPH * 10)[:2000],
        "caf\u00e9 au lait",
    ]
    assert tok.encode("caf\u00e9 au lait") == tok.encode("caf au lait")
    out = [{"text": s, "ids": tok.encode(s)} for s in cases]
    with open(os.path.join(ROOT, "tests", "fixtures", "bpe_parity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    tok.save(os.path.join(ROOT, "web", "test", "fixtures", "tokenizer.json"))
