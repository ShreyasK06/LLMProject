import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Serves api/generate.js from the Vite dev server so `npm run dev` needs no Vercel login.
function apiDev() {
  return {
    name: "api-dev",
    configureServer(server) {
      server.middlewares.use("/api/generate", async (req, res) => {
        const chunks = [];
        for await (const c of req) chunks.push(c);
        const raw = Buffer.concat(chunks).toString();
        try { req.body = raw ? JSON.parse(raw) : undefined; } catch { req.body = undefined; }
        const { default: handler } = await server.ssrLoadModule("/api/generate.js");
        await handler(req, res);
      });
    },
  };
}

// VITE_BASE is the sub-path when hosted on GitHub Pages (e.g. /LLMProject/); asset URLs are derived from it by Vite.
export default defineConfig({ base: process.env.VITE_BASE || "/", plugins: [react(), apiDev()] });
