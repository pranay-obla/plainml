"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Icon } from "./Icon";
import { useApp } from "./AppContext";
import { cx } from "@/lib/format";

const THEMES = [null, "light", "dark"] as const;
type Theme = (typeof THEMES)[number];

function readTheme(): Theme {
  try {
    const saved = localStorage.getItem("plainml-theme");
    return saved === "light" || saved === "dark" ? saved : null;
  } catch {
    return null;
  }
}

export function applyTheme(theme: Theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  window.dispatchEvent(new Event("plainml-theme"));
}

function ThemeButton() {
  const [theme, setTheme] = useState<Theme>(null);
  useEffect(() => setTheme(readTheme()), []);
  const cycle = () => {
    const next = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    try {
      if (next) localStorage.setItem("plainml-theme", next);
      else localStorage.removeItem("plainml-theme");
    } catch {
      /* private mode */
    }
    setTheme(next);
    applyTheme(next);
  };
  const label = theme ? theme[0].toUpperCase() + theme.slice(1) : "Auto";
  return (
    <button className="theme-btn" type="button" onClick={cycle} title="Switch light / dark / auto" aria-label={`Theme: ${label}`}>
      <Icon name={theme === "light" ? "sun" : theme === "dark" ? "moon" : "auto"} />
      <span className="label">{label}</span>
    </button>
  );
}

export function BootChip() {
  const { browser, boot, info } = useApp();
  if (!browser || !boot) return null;
  if (boot.error) return <span className="boot-chip" title={boot.error}>Python couldn&apos;t start</span>;
  if (boot.ready && info)
    return (
      <span className="boot-chip ready" title="Python and plainml are running in this tab">
        <i className="dot" /> Python ready
      </span>
    );
  return (
    <span className="boot-chip" title={boot.stage} aria-live="polite">
      Setting up Python
      <span className="meter" aria-hidden="true">
        <i style={{ width: `${Math.round(boot.progress * 100)}%` }} />
      </span>
    </span>
  );
}

const TABS = [
  { href: "/", label: "New", match: (p: string) => p === "/" || p.startsWith("/analyze") },
  { href: "/runs/", label: "Runs", match: (p: string) => p.startsWith("/runs") || p.startsWith("/run/") },
  { href: "/predict/", label: "Predict", match: (p: string) => p.startsWith("/predict") },
];

export function Header() {
  const pathname = usePathname() || "/";
  const { info, browser } = useApp();
  return (
    <header className="bar">
      <Link className="brand" href="/" aria-label="plainml home">
        <span className="mark">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 17v-6M12 17V7M18 17v-4" />
          </svg>
        </span>
        <span className="word">plainml</span>
      </Link>
      <nav className="tabs" aria-label="Sections">
        {TABS.map((tab) => (
          <Link key={tab.href} href={tab.href} className={cx(tab.match(pathname) && "on")} aria-current={tab.match(pathname) ? "page" : undefined}>
            {tab.label}
          </Link>
        ))}
      </nav>
      <span className="spacer" />
      <BootChip />
      {info ? <span className="version">v{info.version}{browser ? " · in your browser" : ""}</span> : null}
      <ThemeButton />
    </header>
  );
}

export function Footer() {
  const { info } = useApp();
  return (
    <footer className="site-footer">
      <div className="container">
        <span>
          <b>plainml</b> {info ? `v${info.version}` : ""} · free and open source (MIT) · made at USC
        </span>
        <nav aria-label="More">
          <a href="https://plainml-tpua.vercel.app/docs/">Docs</a>
          <a href="https://github.com/pranay-obla/plainml">GitHub</a>
          <a href="https://pypi.org/project/plainml/">PyPI</a>
          <a href="https://github.com/pranay-obla/plainml/blob/main/CHANGELOG.md">Changelog</a>
        </nav>
      </div>
    </footer>
  );
}
