"""From-scratch character-level BPE tokenizer (pure standard library).

The pre-tokenizer is hand-written on ASCII classes so a JavaScript port can
match it exactly. Merges never cross a pre-split piece boundary.
"""
import json
from collections import Counter

_NL = "\n"
_SP = " "


def _is_alnum(c):
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9")


def _cls(c):
    """0 = newline, 1 = space, 2 = ASCII alphanumeric, 3 = anything else."""
    if c == _NL:
        return 0
    if c == _SP:
        return 1
    return 2 if _is_alnum(c) else 3


def _run(text, i, k):
    """End index of the run of class k starting at i."""
    n = len(text)
    while i < n and _cls(text[i]) == k:
        i += 1
    return i


def pretokenize(text):
    """Split text into pieces; "".join(pieces) == text.

    Pieces: [space]? + run of [A-Za-z0-9]; [space]? + run of other characters
    that are neither space nor newline; a run of newlines; or a lone run of
    spaces (when not followed by a word, or all but the last space of a run
    that is followed by a word).
    """
    pieces = []
    i, n = 0, len(text)
    while i < n:
        k = _cls(text[i])
        if k == 0:
            j = _run(text, i, 0)
            pieces.append(text[i:j])
        elif k == 1:
            j = _run(text, i, 1)
            if j < n and _cls(text[j]) >= 2:
                if j - 1 > i:
                    pieces.append(text[i:j - 1])
                end = _run(text, j, _cls(text[j]))
                pieces.append(text[j - 1:end])
                j = end
            else:
                pieces.append(text[i:j])
        else:
            j = _run(text, i, k)
            pieces.append(text[i:j])
        i = j
    return pieces


class BPE:
    def __init__(self, alphabet, merges):
        self.alphabet = list(alphabet)
        self.merges = [tuple(m) for m in merges]
        self.vocab = list(self.alphabet)
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._char_id = {c: i for i, c in enumerate(self.alphabet)}
        self._rank = {m: r for r, m in enumerate(self.merges)}
        base = len(self.alphabet)
        self._merged = {m: base + r for r, m in enumerate(self.merges)}
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size, progress=None):
        """Learn merges until the vocabulary has vocab_size entries (or no pair is left).

        progress, if given, is called as progress(merges_done, merges_total) after every 100th merge.
        """
        alphabet = sorted(set(text))
        merges_total = max(0, vocab_size - len(alphabet))
        char_id = {c: i for i, c in enumerate(alphabet)}
        words = Counter(pretokenize(text))
        seqs = [[char_id[c] for c in w] for w in words]
        freqs = list(words.values())
        merges = []
        next_id = len(alphabet)
        while next_id < vocab_size:
            counts = Counter()
            for seq, f in zip(seqs, freqs):
                for p in zip(seq, seq[1:]):
                    counts[p] += f
            if not counts:
                break
            best = min(counts, key=lambda p: (-counts[p], p))
            merges.append(best)
            a, b = best
            for seq in seqs:
                if len(seq) < 2:
                    continue
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                seq[:] = out
            next_id += 1
            if progress is not None and len(merges) % 100 == 0:
                progress(len(merges), merges_total)
        return cls(alphabet, merges)

    def _encode_piece(self, piece):
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        seq = [self._char_id[c] for c in piece]
        while len(seq) > 1:
            best, best_rank = None, None
            for p in zip(seq, seq[1:]):
                r = self._rank.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            new_id = self._merged[best]
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        self._cache[piece] = seq
        return seq

    def encode(self, text):
        text = "".join(c for c in text if c in self._char_id)
        ids = []
        for piece in pretokenize(text):
            ids.extend(self._encode_piece(piece))
        return list(ids)

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def char_len(self, id):
        return len(self.vocab[id])

    def unknown_chars(self, text):
        return sum(1 for c in text if c not in self._char_id)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "alphabet": self.alphabet,
                       "merges": [list(m) for m in self.merges],
                       "vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["alphabet"], d["merges"])
