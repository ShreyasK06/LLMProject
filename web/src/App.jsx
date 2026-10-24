import { useCallback, useEffect, useRef, useState } from "react";

const EXAMPLES = ["Hello, how are you?", "What is the capital of France?", "Give me ideas for a birthday party.", "Explain what a computer is."];

// The model only continues text, so a chat is framed as a transcript, the same User/Assistant format prepare_dolly.py
// trains on: the user speaks as "User" and the model writes the "Assistant" reply. A blank line ends a speech, which is
// where the reply stops.
const USER_NAME = "User";
const BOT_NAME = "Assistant";
const STOP = ["\n\n"];
const HISTORY_CHARS = 500; // the model attends to at most ~256 characters anyway

// Empty when the API is on the same origin (dev server, Vercel). On GitHub Pages the build sets VITE_API_URL to the
// Vercel deployment, e.g. https://tiny-gpt.vercel.app
const API = `${(import.meta.env.VITE_API_URL || "").replace(/\/$/, "")}/api/generate`;

// Training replies were cut at 400 characters, so a longer cap lets a reply end at its own blank line.
const DEFAULTS = { temperature: 0.8, topK: 40, maxChars: 400, seed: "", carry: true };

const turn = (who, text) => `${who}:\n${text}`;
function buildPrompt(history, text) {
  const kept = [];
  let size = 0;
  for (let i = history.length - 1; i >= 0; i--) {
    size += history[i].length + 2;
    if (size > HISTORY_CHARS) break;
    kept.unshift(history[i]);
  }
  return [...kept, turn(USER_NAME, text)].join("\n\n") + `\n\n${BOT_NAME}:\n`;
}

const Icon = ({ children }) => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    {children}
  </svg>
);
const PlusIcon = () => <Icon><path d="M12 5v14M5 12h14" /></Icon>;
const CloseIcon = () => <Icon><path d="M6 6l12 12M18 6L6 18" /></Icon>;
const SunIcon = () => (
  <Icon>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
  </Icon>
);
const MoonIcon = () => <Icon><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></Icon>;
const SlidersIcon = () => (
  <Icon>
    <path d="M4 6h8M18 6h2M4 12h2M12 12h8M4 18h10M20 18h0" />
    <circle cx="15" cy="6" r="2" />
    <circle cx="9" cy="12" r="2" />
    <circle cx="17" cy="18" r="2" />
  </Icon>
);
const ArrowUpIcon = () => <Icon><path d="M12 19V5M5 12l7-7 7 7" /></Icon>;
const StopIcon = () => <Icon><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none" /></Icon>;

// The signature element: rests with a slow glow, pulses while the model is generating.
const Orb = ({ size = 28, state = "idle" }) => <span className={`orb ${state}`} style={{ "--orb": `${size}px` }} aria-hidden="true" />;

const storedTheme = () => {
  try {
    return localStorage.getItem("theme");
  } catch {
    return null;
  }
};

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [settings, setSettings] = useState(DEFAULTS);
  const [showSettings, setShowSettings] = useState(false);
  const [status, setStatus] = useState({ state: "loading" });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [theme, setTheme] = useState(storedTheme); // "light" | "dark" | null (follow the system)
  const abortRef = useRef(null);
  const historyRef = useRef([]); // finished turns, formatted as script lines
  const endRef = useRef(null);
  const inputRef = useRef(null);
  const settingsBtnRef = useRef(null);
  const closeBtnRef = useRef(null);
  const settingsSeen = useRef(false);

  useEffect(() => {
    fetch(API)
      .then(async (r) => {
        const j = await r.json();
        setStatus(r.ok ? { state: "ready", ...j } : { state: "error", error: j.error || "Model unavailable" });
      })
      .catch(() => setStatus({ state: "error", error: "Cannot reach the API." }));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    const root = document.documentElement;
    if (theme) root.dataset.theme = theme;
    else delete root.dataset.theme;
  }, [theme]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el || !el.scrollHeight) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  // Settings drawer: Escape closes it, focus moves in on open and back to the toggle on close
  useEffect(() => {
    if (showSettings) {
      settingsSeen.current = true;
      closeBtnRef.current?.focus();
      const onKey = (e) => e.key === "Escape" && setShowSettings(false);
      window.addEventListener("keydown", onKey);
      return () => window.removeEventListener("keydown", onKey);
    }
    if (settingsSeen.current) settingsBtnRef.current?.focus();
  }, [showSettings]);

  const dark = theme ? theme === "dark" : !!window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  const toggleTheme = () => {
    const next = dark ? "light" : "dark";
    setTheme(next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* private mode: the choice just won't persist */
    }
  };

  const set = (k) => (e) =>
    setSettings((s) => ({ ...s, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));

  const send = useCallback(
    async (raw) => {
      const text = raw.replace(/\s+/g, " ").trim(); // a blank line would read as the end of the speech
      if (!text || busy) return;
      setInput("");
      setNotice("");
      setBusy(true);

      const prompt = buildPrompt(settings.carry ? historyRef.current : [], text);

      // Assistant text is kept as the chunks that arrived, so each one can fade in as the model produces it.
      setMessages((m) => [...m, { role: "user", content: text }, { role: "assistant", chunks: [] }]);
      const patchLast = (fn) =>
        setMessages((m) => {
          const copy = m.slice();
          copy[copy.length - 1] = fn(copy[copy.length - 1]);
          return copy;
        });
      const settle = (l) => {
        // trim the tail once the stream ends; never leave an empty bubble
        const chunks = [...l.chunks];
        if (chunks.length) chunks[chunks.length - 1] = chunks[chunks.length - 1].trimEnd();
        return { ...l, chunks: chunks.join("").trim() ? chunks : ["…"] };
      };

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      let reply = "";
      try {
        const seed = settings.seed === "" ? null : parseInt(settings.seed, 10);
        const res = await fetch(API, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            temperature: Number(settings.temperature),
            topK: Number(settings.topK),
            maxChars: Number(settings.maxChars),
            seed: Number.isInteger(seed) ? seed : null,
            stop: STOP,
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
          patchLast((l) => {
            const piece = l.chunks.length ? chunk : chunk.trimStart();
            return piece ? { ...l, chunks: [...l.chunks, piece] } : l;
          });
        }
        patchLast(settle);
      } catch (e) {
        if (e.name === "AbortError") patchLast((l) => (l.role === "assistant" ? settle(l) : l));
        else patchLast(() => ({ role: "error", content: "Connection lost. Try again." }));
      } finally {
        if (reply.trim()) historyRef.current.push(turn(USER_NAME, text), turn(BOT_NAME, reply.trim()));
        abortRef.current = null;
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [busy, settings]
  );

  const stop = () => abortRef.current?.abort();
  const clear = () => {
    abortRef.current?.abort();
    setMessages([]);
    setNotice("");
    historyRef.current = [];
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send(input);
    }
  };

  const ready = status.state === "ready";
  const empty = messages.length === 0;

  return (
    <div className="shell">
      <header className="bar">
        <div className="brand">
          <Orb size={24} state={busy ? "live" : "idle"} />
          <h1>Tiny GPT Chat</h1>
          <span className={`status ${status.state}`} role="status">
            {status.state === "loading" && "Loading…"}
            {status.state === "error" && "Model offline"}
            {ready && `${(status.params / 1e6).toFixed(1)}M params`}
          </span>
        </div>
        <div className="bar-actions">
          <button type="button" className="icon-btn" onClick={clear} aria-label="New chat" title="New chat">
            <PlusIcon />
          </button>
          <button type="button" className="icon-btn" onClick={toggleTheme} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} title="Theme">
            {dark ? <SunIcon /> : <MoonIcon />}
          </button>
          <button
            type="button"
            ref={settingsBtnRef}
            className="icon-btn"
            onClick={() => setShowSettings((s) => !s)}
            aria-expanded={showSettings}
            aria-controls="settings"
            aria-label="Settings"
            title="Settings"
          >
            <SlidersIcon />
          </button>
        </div>
      </header>

      {empty && <div className="ambient" aria-hidden="true" />}

      <main className="scroll">
        <div className={`thread ${empty ? "is-empty" : ""}`}>
          {status.state === "error" && (
            <div className="msg error" role="alert">
              <span className="err-mark" aria-hidden="true">!</span>
              <div className="bubble">{status.error}</div>
            </div>
          )}
          {empty ? (
            <div className="hero">
              <Orb size={56} state="idle" />
              <h2>Ask me anything</h2>
              <p>
                A small GPT trained from scratch, one character at a time. It is tiny, so replies sound like answers
                but are often wrong or nonsensical.
              </p>
              <div className="cards">
                {EXAMPLES.map((ex, i) => (
                  <button key={ex} type="button" className="chip" style={{ "--i": i }} disabled={!ready || busy} onClick={() => send(ex)}>
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div role="log" aria-label="Conversation" aria-live="polite" aria-busy={busy} className="log">
              {messages.map((m, i) => {
                const live = busy && i === messages.length - 1 && m.role === "assistant";
                return (
                  <div key={i} className={`msg ${m.role}`}>
                    {m.role === "assistant" && <Orb size={28} state={live ? "live" : "rest"} />}
                    {m.role === "error" && (
                      <span className="err-mark" aria-hidden="true">!</span>
                    )}
                    <span className="sr-only">{m.role === "user" ? "You" : m.role === "error" ? "Error" : "Assistant"}: </span>
                    <div className="bubble">
                      {m.role === "assistant"
                        ? m.chunks.map((c, j) => (
                            <span key={j} className="ch">
                              {c}
                            </span>
                          ))
                        : m.content}
                      {live && m.chunks.length === 0 && (
                        <span className="dots" aria-hidden="true">
                          <i />
                          <i />
                          <i />
                        </span>
                      )}
                      {live && m.chunks.length > 0 && <span className="caret" aria-hidden="true" />}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div ref={endRef} />
        </div>
      </main>

      <div className="dock">
        <div className="dock-in">
          {notice && <div className="notice">{notice}</div>}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            <label htmlFor="prompt" className="sr-only">
              Message
            </label>
            <textarea
              id="prompt"
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Message Tiny GPT…"
              disabled={!ready}
            />
            {busy ? (
              <button key="stop" type="button" className="primary stop" onClick={stop} aria-label="Stop generating">
                <StopIcon />
              </button>
            ) : (
              <button key="send" type="submit" className="primary" disabled={!ready || !input.trim()} aria-label="Send message">
                <ArrowUpIcon />
              </button>
            )}
          </form>
          <p className="hint">Enter to send · Shift+Enter for a new line · a tiny model, so check anything it says</p>
        </div>
      </div>

      <aside id="settings" className={`drawer ${showSettings ? "open" : ""}`} aria-label="Generation settings" aria-hidden={!showSettings}>
        <div className="drawer-head">
          <h2>Settings</h2>
          <button type="button" ref={closeBtnRef} className="icon-btn" onClick={() => setShowSettings(false)} aria-label="Close settings">
            <CloseIcon />
          </button>
        </div>

        <div className="field">
          <label htmlFor="temp">
            Temperature <output>{Number(settings.temperature).toFixed(2)}</output>
          </label>
          <input id="temp" type="range" min="0.1" max="1.5" step="0.05" value={settings.temperature} onChange={set("temperature")} />
          <small>Lower is safer and more repetitive; higher is wilder.</small>
        </div>

        <div className="field">
          <label htmlFor="topk">
            Top-k <output>{settings.topK}</output>
          </label>
          <input id="topk" type="range" min="1" max={ready ? status.vocab : 95} step="1" value={settings.topK} onChange={set("topK")} />
          <small>Sample only from the k likeliest next characters.</small>
        </div>

        <div className="field">
          <label htmlFor="len">
            Reply length <output>{settings.maxChars} chars</output>
          </label>
          <input id="len" type="range" min="50" max="1000" step="50" value={settings.maxChars} onChange={set("maxChars")} />
        </div>

        <div className="field">
          <label htmlFor="seed">Seed (blank = random)</label>
          <input id="seed" type="number" inputMode="numeric" value={settings.seed} onChange={set("seed")} placeholder="random" />
        </div>

        <div className="field">
          <label className="switch" htmlFor="carry">
            <span>Remember the conversation</span>
            <input id="carry" type="checkbox" checked={settings.carry} onChange={set("carry")} />
          </label>
          <small>On: the model sees the last few turns. Off: every message starts fresh.</small>
        </div>

        {ready && (
          <p className="meta mono">
            {status.layers} layers · context {status.block} ·{" "}
            {status.tokenizer === "bpe" ? `${status.vocab}-token BPE vocabulary` : `${status.vocab}-character vocabulary`}
            {status.step != null && ` · checkpoint step ${status.step}`}
          </p>
        )}
      </aside>
    </div>
  );
}
