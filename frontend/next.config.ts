import type { NextConfig } from "next";

// `npm run build` writes static files (out/), which scripts/sync-static.mjs copies into the Python
// package: `plainml web` serves them, and `plainml web --export` turns them into the in-browser site.
// `npm run dev` instead proxies /api to a running `plainml web` (default port 8765).
const dev = process.env.NODE_ENV === "development";
const api = process.env.PLAINML_API ?? "http://127.0.0.1:8765";

const config: NextConfig = {
  output: dev ? undefined : "export",
  trailingSlash: true, // runs/index.html, which every static host and Starlette serve as /runs/
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
  turbopack: { root: __dirname },
  agentRules: false, // `next dev` would otherwise write an AGENTS.md here
  // In dev, / serves the static landing page (public/index.html); the export copies it to out/ as is.
  ...(dev
    ? {
        rewrites: async () => ({
          beforeFiles: [{ source: "/", destination: "/index.html" }],
          afterFiles: [{ source: "/api/:path*", destination: `${api}/api/:path*` }],
          fallback: [],
        }),
      }
    : {}),
};

export default config;
