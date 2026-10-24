// Character-level BPE tokenizer; exact port of bpe.py (encode/decode only).
// Strings are iterated by code point (Array.from) to match Python.
import { readFileSync } from "node:fs";

const isAlnum = (c) =>
  (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || (c >= "0" && c <= "9");

// 0 = newline, 1 = space, 2 = ASCII alphanumeric, 3 = anything else.
const cls = (c) => (c === "\n" ? 0 : c === " " ? 1 : isAlnum(c) ? 2 : 3);

function run(chars, i, k) {
  while (i < chars.length && cls(chars[i]) === k) i++;
  return i;
}

// chars: array of code points. Returns array of pieces (arrays of chars).
export function pretokenize(chars) {
  const pieces = [];
  const n = chars.length;
  let i = 0;
  while (i < n) {
    const k = cls(chars[i]);
    let j;
    if (k === 0) {
      j = run(chars, i, 0);
      pieces.push(chars.slice(i, j));
    } else if (k === 1) {
      j = run(chars, i, 1);
      if (j < n && cls(chars[j]) >= 2) {
        if (j - 1 > i) pieces.push(chars.slice(i, j - 1));
        const end = run(chars, j, cls(chars[j]));
        pieces.push(chars.slice(j - 1, end));
        j = end;
      } else {
        pieces.push(chars.slice(i, j));
      }
    } else {
      j = run(chars, i, k);
      pieces.push(chars.slice(i, j));
    }
    i = j;
  }
  return pieces;
}

export class Tokenizer {
  constructor(alphabet, merges) {
    this.alphabet = alphabet.slice();
    this.merges = merges.map(([a, b]) => [a, b]);
    this.vocab = this.alphabet.slice();
    for (const [a, b] of this.merges) this.vocab.push(this.vocab[a] + this.vocab[b]);
    this.vocabSize = this.vocab.length;
    this._charId = new Map(this.alphabet.map((c, i) => [c, i]));
    const base = this.alphabet.length;
    this._rank = new Map();
    this._merged = new Map();
    this.merges.forEach(([a, b], r) => {
      this._rank.set(`${a},${b}`, r);
      this._merged.set(`${a},${b}`, base + r);
    });
    this._cache = new Map();
    this.cacheMax = 50000; // pieces come from untrusted prompts, so the cache is cleared when it grows past this
  }

  _encodePiece(chars) {
    const key = chars.join("");
    const hit = this._cache.get(key);
    if (hit) return hit;
    let seq = chars.map((c) => this._charId.get(c));
    while (seq.length > 1) {
      let best = null;
      let bestRank = null;
      for (let i = 0; i + 1 < seq.length; i++) {
        const p = `${seq[i]},${seq[i + 1]}`;
        const r = this._rank.get(p);
        if (r !== undefined && (bestRank === null || r < bestRank)) {
          best = p;
          bestRank = r;
        }
      }
      if (best === null) break;
      const newId = this._merged.get(best);
      const out = [];
      for (let i = 0; i < seq.length; ) {
        if (i + 1 < seq.length && `${seq[i]},${seq[i + 1]}` === best) {
          out.push(newId);
          i += 2;
        } else {
          out.push(seq[i]);
          i += 1;
        }
      }
      seq = out;
    }
    if (this._cache.size >= this.cacheMax) this._cache.clear();
    this._cache.set(key, seq);
    return seq;
  }

  encode(text) {
    const all = Array.from(text);
    const kept = all.filter((c) => this._charId.has(c));
    const ids = [];
    for (const piece of pretokenize(kept)) {
      for (const id of this._encodePiece(piece)) ids.push(id);
    }
    return { ids, dropped: all.length - kept.length };
  }

  decode(ids) {
    return ids.map((i) => this.vocab[i]).join("");
  }

  pieceOf(id) {
    return this.vocab[id];
  }

  charLen(id) {
    return Array.from(this.vocab[id]).length;
  }
}

export function loadTokenizer(path) {
  const d = JSON.parse(readFileSync(path, "utf8"));
  return new Tokenizer(d.alphabet, d.merges);
}
