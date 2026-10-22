// Calls the Vercel handler directly with mock req/res. Requires web/model/ (see export_weights.py).
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import handler from "../api/generate.js";

function call(method, body) {
  const res = new EventEmitter();
  Object.assign(res, {
    statusCode: 200, headers: {}, chunks: [],
    setHeader(k, v) { this.headers[k.toLowerCase()] = v; },
    write(c) { this.chunks.push(c); return true; },
    end(c) { if (c) this.chunks.push(c); this.done = true; },
  });
  return handler({ method, body }, res).then(() => res);
}

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

const dropped = await call("POST", { prompt: "ROMEO:{}~", maxChars: 5 });
assert.equal(dropped.headers["x-dropped-chars"], "3");

assert.equal((await call("POST", { prompt: "{}~" })).statusCode, 400);
assert.equal((await call("POST", {})).statusCode, 400);
assert.equal((await call("POST", { prompt: "x".repeat(5000) })).statusCode, 400);
assert.equal((await call("PUT", {})).statusCode, 405);
const clamped = await call("POST", { prompt: "The", maxChars: 99999, temperature: -5, topK: 0 });
assert.equal(clamped.chunks.join("").length, 1000, "maxChars clamps to 1000");
console.log("all API tests passed");
