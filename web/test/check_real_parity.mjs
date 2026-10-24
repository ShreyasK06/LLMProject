// Checks the JS tokenizer against Python encodings of a real tokenizer (written by tests/make_real_parity.py).
// Usage: node web/test/check_real_parity.mjs <tokenizer.json> <parity.json>
// Exits non-zero if any slice's ids or dropped-character count differ.
import { readFileSync } from "node:fs";
import { loadTokenizer } from "../lib/bpe.js";

const [tokPath, parityPath] = process.argv.slice(2);
if (!tokPath || !parityPath) {
  console.error("usage: node web/test/check_real_parity.mjs <tokenizer.json> <parity.json>");
  process.exit(2);
}
const tok = loadTokenizer(tokPath);
const cases = JSON.parse(readFileSync(parityPath, "utf8"));
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
let bad = 0, tokens = 0;
cases.forEach(({ text, ids, dropped }, i) => {
  const out = tok.encode(text);
  tokens += ids.length;
  if (!same(out.ids, ids) || out.dropped !== dropped) {
    if (bad++ < 5) {
      console.error(`slice ${i}: mismatch (python ${ids.length} ids, dropped ${dropped}; js ${out.ids.length} ids, ` +
        `dropped ${out.dropped}) for ${JSON.stringify(text.slice(0, 60))}...`);
    }
  }
});
if (!cases.length) { console.error("real parity: no slices in the parity file"); process.exit(1); }
if (bad) { console.error(`real parity: ${bad}/${cases.length} slices differ`); process.exit(1); }
console.log(`real parity: ${cases.length} slices, ${tokens} tokens, vocab ${tok.vocabSize}: JS matches Python`);
