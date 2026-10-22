import { useCallback, useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "ROMEO:",
  "First Citizen:\nBefore we proceed",
  "KING HENRY:\nMy lords,",
  "To be, or not to be",
];

const DEFAULTS = { temperature: 0.8, topK: 40, maxChars: 300, seed: "", carry: false };

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [settings, setSettings] = useState(DEFAULTS);
  const [showSettings, setShowSettings] = useState(() => window.innerWidth >= 860);
  const [status, setStatus] = useState({ state: "loading" });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const abortRef = useRef(null);
  const contextRef = useRef("");
  const endRef = useRef(null);

  useEffect(() => {
    fetch("/api/generate")
      .then(async (r) => {
        const j = await r.json();
        setStatus(r.ok ? { state: "ready", ...j } : { state: "error", error: j.error || "Model unavailable" });
      })
      .catch(() => setStatus({ state: "error", error: "Cannot reach the API." }));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const set = (k) => (e) =>
    setSettings((s) => ({ ...s, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));

  const send = useCallback(
    async (raw) => {
      const text = raw.replace(/\s+$/, "");
      if (!text || busy) return;
      setInput("");
      setNotice("");
      setBusy(true);

      let context = settings.carry ? contextRef.current.slice(-1500) : "";
      if (context && !context.endsWith("\n")) context += "\n";
      const prompt = context + text;

      setMessages((m) => [...m, { role: "user", content: text }, { role: "assistant", content: "" }]);
      const patchLast = (fn) =>
        setMessages((m) => {
          const copy = m.slice();
          copy[copy.length - 1] = fn(copy[copy.length - 1]);
          return copy;
        });

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      let reply = "";
      try {
        const seed = settings.seed === "" ? null : parseInt(settings.seed, 10);
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            temperature: Number(settings.temperature),
            topK: Number(settings.topK),
            maxChars: Number(settings.maxChars),
            seed: Number.isInteger(seed) ? seed : null,
          }),
          signal: ctrl.signal,
        });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          patchLast(() => ({ role: "error", content: j.error || `Request failed (${res.status}).` }));
          return;
        }
        const dropped = Number(res.headers.get("X-Dropped-Chars") || 0);
        if (dropped) setNotice(`Ignored ${dropped} character${dropped > 1 ? "s" : ""} the model doesn't know.`);
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          reply += chunk;
          patchLast((l) => ({ ...l, content: l.content + chunk }));
        }
      } catch (e) {
        if (e.name !== "AbortError") patchLast(() => ({ role: "error", content: "Connection lost. Try again." }));
      } finally {
        contextRef.current = prompt + reply;
        abortRef.current = null;
        setBusy(false);
      }
    },
    [busy, settings]
  );

  const stop = () => abortRef.current?.abort();
  const clear = () => {
    abortRef.current?.abort();
    setMessages([]);
    setNotice("");
    contextRef.current = "";
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send(input);
    }
  };

  const ready = status.state === "ready";

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>
            <span aria-hidden="true">🎭</span> Tiny Shakespeare GPT
          </h1>
          <p className="sub">
            A small GPT trained from scratch on Shakespeare. It isn't an assistant: it <em>continues</em> what you
            type, like a play script. Try a speaker name followed by a colon.
          </p>
        </div>
        <div className="top-actions">
          <span className={`pill ${status.state}`} role="status">
            {status.state === "loading" && "Loading…"}
            {status.state === "error" && "Model offline"}
            {ready && `${(status.params / 1e6).toFixed(1)}M params`}
          </span>
          <button type="button" className="ghost" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            Settings
          </button>
        </div>
      </header>

      <div className={`layout ${showSettings ? "with-settings" : ""}`}>
        <main className="chat-col">
          {status.state === "error" && <div className="banner error">{status.error}</div>}
          <section className="chat" aria-live="polite" aria-label="Conversation">
            {messages.length === 0 ? (
              <div className="empty">
                <p>Start a scene, or pick one:</p>
                <div className="chips">
                  {EXAMPLES.map((ex) => (
                    <button key={ex} type="button" className="chip" disabled={!ready || busy} onClick={() => send(ex)}>
                      {ex.replace("\n", " ⏎ ")}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={`msg ${m.role}`}>
                  <span className="who">{m.role === "user" ? "You" : m.role === "error" ? "Error" : "Model"}</span>
                  <div className="bubble">
                    {m.content}
                    {busy && i === messages.length - 1 && m.role === "assistant" && <span className="caret" />}
                  </div>
                </div>
              ))
            )}
            <div ref={endRef} />
          </section>

          {notice && <div className="banner">{notice}</div>}

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            <label htmlFor="prompt" className="sr-only">
              Prompt
            </label>
            <textarea
              id="prompt"
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="ROMEO:   (Enter to send, Shift+Enter for a new line)"
              disabled={!ready}
            />
            {busy ? (
              <button type="button" className="primary stop" onClick={stop}>
                Stop
              </button>
            ) : (
              <button type="submit" className="primary" disabled={!ready || !input.trim()}>
                Send
              </button>
            )}
          </form>
        </main>

        {showSettings && (
          <aside className="settings" aria-label="Generation settings">
            <h2>Settings</h2>
            <label htmlFor="temp">
              Temperature <output>{Number(settings.temperature).toFixed(2)}</output>
            </label>
            <input id="temp" type="range" min="0.1" max="1.5" step="0.05" value={settings.temperature} onChange={set("temperature")} />
            <small>Lower is safer and more repetitive; higher is wilder.</small>

            <label htmlFor="topk">
              Top-k <output>{settings.topK}</output>
            </label>
            <input id="topk" type="range" min="1" max="65" step="1" value={settings.topK} onChange={set("topK")} />
            <small>Sample only from the k likeliest next characters.</small>

            <label htmlFor="len">
              Reply length <output>{settings.maxChars} chars</output>
            </label>
            <input id="len" type="range" min="50" max="1000" step="50" value={settings.maxChars} onChange={set("maxChars")} />

            <label htmlFor="seed">Seed (blank = random)</label>
            <input id="seed" type="number" inputMode="numeric" value={settings.seed} onChange={set("seed")} placeholder="random" />

            <label className="check" htmlFor="carry">
              <input id="carry" type="checkbox" checked={settings.carry} onChange={set("carry")} />
              Carry the conversation as context
            </label>
            <small>Off: every message is a fresh prompt. On: the model also sees what was said before.</small>

            <button type="button" className="ghost wide" onClick={clear}>
              Clear chat
            </button>
            {ready && (
              <p className="meta">
                {status.layers} layers · context {status.block} · {status.vocab}-character vocabulary
                {status.step != null && ` · checkpoint step ${status.step}`}
              </p>
            )}
          </aside>
        )}
      </div>
    </div>
  );
}
