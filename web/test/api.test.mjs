// Calls the Vercel handler directly with mock req/res. Requires web/model/ (see export_weights.py).
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import handler from "../api/generate.js";

function call(method, body, headers = {}) {
  const res = new EventEmitter();
  Object.assign(res, {
    statusCode: 200, headers: {}, chunks: [],
    setHeader(k, v) { this.headers[k.toLowerCase()] = v; },
    write(c) { this.chunks.push(c); return true; },
    end(c) { if (c) this.chunks.push(c); this.done = true; },
  });
  return handler({ method, body, headers }, res).then(() => res);
}

// Characters outside the model's vocabulary (it is ASCII-only or smaller), so these tests hold for any trained model.
const OUT = "é中文";

const info = await call("GET");
assert.equal(info.statusCode, 200);
const meta = JSON.parse(info.chunks.join(""));
assert.equal(meta.ready, true);
console.log("GET ok:", meta);

const a = await call("POST", { prompt: "ROMEO:", maxChars: 40, seed: 7, temperature: 0.8, topK: 40 });
assert.equal(a.statusCode, 200);
assert.match(a.headers["content-type"], /text\/plain/);
assert.equal(a.chunks.join("").length, 40);
const b = await call("POST", { prompt: "ROMEO:", maxChars: 40, seed: 7 });
assert.equal(a.chunks.join(""), b.chunks.join(""), "same seed must give same output");
console.log("POST ok, deterministic with seed:", JSON.stringify(a.chunks.join("")));

const long = await call("POST", { prompt: "The", maxChars: 1000, seed: 1 });
assert.equal(long.chunks.join("").length, 1000, "generation past the context window must keep working");

const dropped = await call("POST", { prompt: `ROMEO:${OUT}`, maxChars: 5 });
assert.equal(dropped.headers["x-dropped-chars"], "3");

// Stop strings: generation ends at the first match, the match is never sent, and sampling is otherwise unchanged.
// A space and "e " occur in any model trained on English text, so these don't depend on which weights are loaded.
const framed = "User:\nHello\n\nAssistant:\n";
const unstopped = (await call("POST", { prompt: framed, maxChars: 400, seed: 3 })).chunks.join("");
for (const stop of [" ", "e "]) {
  const text = (await call("POST", { prompt: framed, maxChars: 400, seed: 3, stop: [stop] })).chunks.join("");
  assert.ok(text.length < 400, `reply ends at ${JSON.stringify(stop)}, before maxChars`);
  assert.ok(!text.includes(stop), `the stop string ${JSON.stringify(stop)} is never sent`);
  assert.equal(unstopped.indexOf(stop), text.length, "it stops exactly where the stop string first appears");
  assert.ok(unstopped.startsWith(text), "stopping only truncates; it does not change sampling");
}

assert.equal((await call("POST", { prompt: OUT })).statusCode, 400);
assert.equal((await call("POST", {})).statusCode, 400);
assert.equal((await call("POST", { prompt: "x".repeat(5000) })).statusCode, 400);
assert.equal((await call("PUT", {})).statusCode, 405);
const clamped = await call("POST", { prompt: "The", maxChars: 99999, temperature: -5, topK: 0 });
assert.equal(clamped.chunks.join("").length, 1000, "maxChars clamps to 1000");
// CORS: the GitHub Pages origin and localhost are allowed, others are not, and the preflight skips model loading
const pages = "https://shreyask06.github.io";
const pre = await call("OPTIONS", undefined, { origin: pages });
assert.equal(pre.statusCode, 204);
assert.equal(pre.headers["access-control-allow-origin"], pages);
assert.match(pre.headers["access-control-allow-methods"], /POST/);
assert.match(pre.headers["access-control-allow-headers"], /Content-Type/i);
const cross = await call("POST", { prompt: "The", maxChars: 5 }, { origin: pages });
assert.equal(cross.headers["access-control-allow-origin"], pages);
assert.match(cross.headers["access-control-expose-headers"], /X-Dropped-Chars/);
const local = await call("GET", undefined, { origin: "http://localhost:5173" });
assert.equal(local.headers["access-control-allow-origin"], "http://localhost:5173");
const other = await call("POST", { prompt: "The", maxChars: 5 }, { origin: "https://evil.example" });
assert.equal(other.headers["access-control-allow-origin"], undefined, "unlisted origins get no CORS header");
assert.equal((await call("POST", { prompt: "The", maxChars: 5 })).headers["access-control-allow-origin"], undefined);
console.log("all API tests passed");
