import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadTokenizer } from "../lib/bpe.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const tok = loadTokenizer(here("./fixtures/tokenizer.json"));
const cases = JSON.parse(readFileSync(here("../../tests/fixtures/bpe_parity.json"), "utf8"));
const inAlpha = new Set(JSON.parse(readFileSync(here("./fixtures/tokenizer.json"), "utf8")).alphabet);
const strip = (t) => Array.from(t).filter((c) => inAlpha.has(c)).join("");

assert.ok(cases.length > 0);
for (const { text, ids } of cases) {
  const out = tok.encode(text);
  assert.deepEqual(out.ids, ids, `ids mismatch for ${JSON.stringify(text)}`);
  assert.equal(tok.decode(ids), strip(text), `decode mismatch for ${JSON.stringify(text)}`);
}

assert.equal(tok.encode("é中文").dropped, 3);
assert.equal(tok.encode("😀").dropped, 1);
assert.deepEqual(tok.encode("😀").ids, []);
assert.equal(tok.encode("").ids.length, 0);
assert.equal(tok.encode("").dropped, 0);
const oov = String.fromCodePoint(0xe9);
assert.ok(!inAlpha.has(oov));
assert.deepEqual(tok.encode(`a${oov}b`).ids, tok.encode("ab").ids);
assert.equal(tok.encode(`a${oov}b`).dropped, 1);
assert.equal(tok.vocabSize, JSON.parse(readFileSync(here("./fixtures/tokenizer.json"), "utf8")).vocab.length);
for (const id of [0, 1, tok.vocabSize - 1]) {
  assert.equal(tok.charLen(id), Array.from(tok.pieceOf(id)).length);
}
// The per-piece cache is bounded (prompts are untrusted); encoding more distinct pieces than the cap gives the same
// ids as an uncapped tokenizer.
{
  const letters = [...inAlpha].filter((c) => /[a-z]/.test(c)), L = letters.length;
  const words = [];
  for (let i = 0; i < 60000; i++) {
    let w = "";
    for (let n = i; ; n = Math.floor(n / L)) { w += letters[n % L]; if (n < L) break; }
    words.push(w); // distinct for distinct i, all in the tokenizer's alphabet
  }
  const big = words.join(" ");
  const capped = loadTokenizer(here("./fixtures/tokenizer.json"));
  const uncapped = loadTokenizer(here("./fixtures/tokenizer.json"));
  uncapped.cacheMax = Infinity;
  const a = capped.encode(big).ids;
  assert.ok(uncapped._cache.size === 0 && capped._cache.size <= 50000, `cache size ${capped._cache.size}`);
  const b = uncapped.encode(big).ids;
  assert.ok(uncapped._cache.size > 50000);
  assert.deepEqual(a, b);
  assert.deepEqual(capped.encode(big).ids, b, "re-encoding after the cache was cleared");
}
console.log(`bpe tests passed (${cases.length} parity cases)`);
