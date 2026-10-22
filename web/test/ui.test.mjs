// Mounts the real <App/> in jsdom with a mocked /api/generate and drives it like a user.
import assert from "node:assert/strict";
import { realpathSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { JSDOM } from "jsdom";
import { createServer } from "vite";

const dom = new JSDOM("<!doctype html><div id='root'></div>", { url: "http://localhost/", pretendToBeVisual: true });
Object.assign(globalThis, { window: dom.window, document: dom.window.document, IS_REACT_ACT_ENVIRONMENT: true });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
dom.window.HTMLElement.prototype.scrollIntoView = () => {};

const enc = new TextEncoder();
const bodies = [];
globalThis.fetch = async (url, opts = {}) => {
  if (!opts.method || opts.method === "GET") {
    return { ok: true, json: async () => ({ ready: true, params: 818048, step: 20, block: 128, layers: 4, vocab: 65 }) };
  }
  bodies.push(JSON.parse(opts.body));
  const chunks = ["Hello ", "there"].map((c) => enc.encode(c));
  return {
    ok: true,
    headers: { get: (k) => (k === "X-Dropped-Chars" ? "0" : null) },
    body: { getReader: () => ({ read: async () => (chunks.length ? { done: false, value: chunks.shift() } : { done: true }) }) },
  };
};

const vite = await createServer({ root: realpathSync.native(process.cwd()), server: { middlewareMode: true }, appType: "custom", logLevel: "error" });
const { default: App } = await vite.ssrLoadModule("/src/App.jsx");
// Resolve React from the canonical (real) path, the one Vite uses: on Windows differing path casing would
// otherwise load two React copies and break hooks.
const req = createRequire(join(realpathSync.native(process.cwd()), "package.json"));
const load = async (id) => { const m = await import(pathToFileURL(req.resolve(id)).href); return m.default ?? m; };
const React = await load("react");
const { act } = React;
const { createRoot } = await load("react-dom/client");

const root = createRoot(document.getElementById("root"));
const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 20)); });
await act(async () => root.render(React.createElement(App)));
await flush();

assert.match(document.body.textContent, /Tiny Shakespeare GPT/);
assert.match(document.body.textContent, /0\.8M params/, "status pill shows the model size");
const chips = [...document.querySelectorAll("button.chip")];
assert.equal(chips.length, 4);

await act(async () => chips[0].click());
await flush();
const bubbles = [...document.querySelectorAll(".bubble")].map((b) => b.textContent);
assert.deepEqual(bubbles, ["ROMEO:", "Hello there"], "user prompt and streamed reply are rendered");
assert.equal(bodies[0].prompt, "ROMEO:");
assert.equal(bodies[0].temperature, 0.8);
assert.equal(bodies[0].seed, null);

// carry-context: second prompt must include the first exchange
await act(async () => { document.getElementById("carry").click(); });
const ta = document.getElementById("prompt");
await act(async () => {
  Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, "value").set.call(ta, "JULIET:");
  ta.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
});
await act(async () => document.querySelector("button.primary").click());
await flush();
assert.equal(bodies[1].prompt, "ROMEO:Hello there\nJULIET:", "context is carried when enabled");
assert.equal(document.querySelectorAll(".bubble").length, 4);

await act(async () => [...document.querySelectorAll("button")].find((b) => b.textContent === "Clear chat").click());
assert.equal(document.querySelectorAll(".bubble").length, 0, "clear empties the chat");

console.log("ui tests passed");
await act(async () => root.unmount());
await vite.close();
process.exit(0);
