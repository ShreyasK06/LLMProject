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
Object.defineProperty(globalThis, "localStorage", { value: dom.window.localStorage, configurable: true });
dom.window.HTMLElement.prototype.scrollIntoView = () => {};

const enc = new TextEncoder();
const bodies = [];
let status = { ready: true, params: 818048, step: 20, block: 128, layers: 4, vocab: 65 };
globalThis.fetch = async (url, opts = {}) => {
  if (!opts.method || opts.method === "GET") {
    return { ok: true, json: async () => status };
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

assert.match(document.body.textContent, /Tiny GPT Chat/);
assert.match(document.body.textContent, /0\.8M params/, "status pill shows the model size");
const chips = [...document.querySelectorAll("button.chip")];
assert.equal(chips.length, 4);

await act(async () => chips[0].click());
await flush();
const bubbles = [...document.querySelectorAll(".bubble")].map((b) => b.textContent);
assert.deepEqual(bubbles, ["Hello, how are you?", "Hello there"], "user message and streamed reply are rendered");
assert.equal(bodies[0].prompt, "User:\nHello, how are you?\n\nAssistant:\n", "message is framed as a User/Assistant transcript");
assert.deepEqual(bodies[0].stop, ["\n\n"], "reply stops at the end of the speech");
assert.equal(bodies[0].temperature, 0.8);
assert.equal(bodies[0].seed, null);

// memory is on by default: the second prompt must include the first exchange; extra whitespace is collapsed
const ta = document.getElementById("prompt");
await act(async () => {
  Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, "value").set.call(ta, "And you?\n\n  Truly.");
  ta.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
});
await act(async () => document.querySelector("button.primary").click());
await flush();
assert.equal(
  bodies[1].prompt,
  "User:\nHello, how are you?\n\nAssistant:\nHello there\n\nUser:\nAnd you? Truly.\n\nAssistant:\n",
  "history is carried and the message is single-line",
);
assert.equal(document.querySelectorAll(".bubble").length, 4);

// with memory off, the next prompt starts a fresh scene
await act(async () => { document.getElementById("carry").click(); });
await act(async () => {
  Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, "value").set.call(ta, "Again");
  ta.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
});
await act(async () => document.querySelector("button.primary").click());
await flush();
assert.equal(bodies[2].prompt, "User:\nAgain\n\nAssistant:\n", "no history when memory is off");

await act(async () => document.querySelector('button[aria-label="New chat"]').click());
assert.equal(document.querySelectorAll(".bubble").length, 0, "new chat empties the thread");
assert.equal(document.querySelectorAll("button.chip").length, 4, "the example prompts come back");

// settings drawer: closed by default, opens from the toggle and closes on Escape
const toggle = document.querySelector('button[aria-label="Settings"]');
const drawer = document.getElementById("settings");
assert.equal(toggle.getAttribute("aria-expanded"), "false");
assert.ok(!drawer.classList.contains("open"));
await act(async () => toggle.click());
assert.equal(toggle.getAttribute("aria-expanded"), "true");
assert.ok(drawer.classList.contains("open"));
await act(async () => { dom.window.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Escape" })); });
assert.ok(!drawer.classList.contains("open"), "Escape closes the drawer");

// theme toggle sets data-theme on <html> and remembers it
await act(async () => document.querySelector('button[aria-label^="Switch to"]').click());
assert.equal(document.documentElement.dataset.theme, "dark");
assert.equal(dom.window.localStorage.getItem("theme"), "dark");

assert.match(document.body.textContent, /65-character vocabulary/);
assert.equal(document.getElementById("topk").max, "65", "top-k slider max is the vocabulary size");

// a BPE model reports its token vocabulary
await act(async () => root.unmount());
status = { ...status, vocab: 2048, tokenizer: "bpe" };
const root2 = createRoot(document.getElementById("root"));
await act(async () => root2.render(React.createElement(App)));
await flush();
assert.match(document.body.textContent, /2048-token BPE vocabulary/);
assert.doesNotMatch(document.body.textContent, /character vocabulary/);
assert.equal(document.getElementById("topk").max, "2048");

console.log("ui tests passed");
await act(async () => root2.unmount());
await vite.close();
process.exit(0);
