"use client";

import Link from "next/link";
import type { ApiError } from "@/lib/api";
import { Icon } from "./Icon";

export function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <main className="page">
      <div className="loading">
        <span className="spin" />
        {text}
      </div>
    </main>
  );
}

export function Problem({ error }: { error: ApiError | Error }) {
  const hint = "hint" in error ? error.hint : null;
  return (
    <main className="page">
      <div className="empty">
        <div className="orb">
          <Icon name="alert" />
        </div>
        <h2>{error.message}</h2>
        {hint ? <p className="sub">{hint}</p> : null}
        <p style={{ marginTop: 20 }}>
          <Link className="btn" href="/">
            Start over
          </Link>
        </p>
      </div>
    </main>
  );
}
