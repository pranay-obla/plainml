"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Icon } from "@/components/Icon";
import { Segmented } from "@/components/Options";
import { KINDS, RunRow } from "@/components/Runs";
import { Problem } from "@/components/States";
import { api, ApiError } from "@/lib/api";
import type { RunRow as Run } from "@/lib/types";

export default function RunsPage() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [problem, setProblem] = useState<ApiError | null>(null);
  const [kind, setKind] = useState("all");
  const [query, setQuery] = useState("");

  useEffect(() => {
    api<Run[]>("/api/runs").then(setRuns).catch(setProblem);
  }, []);

  const shown = useMemo(
    () =>
      (runs || []).filter(
        (r) => (kind === "all" || r.kind === kind || (kind === "train" && r.kind === "tune")) && (!query || JSON.stringify(r).toLowerCase().includes(query)),
      ),
    [runs, kind, query],
  );

  if (problem) return <Problem error={problem} />;
  const present = new Set((runs || []).map((r) => (r.kind === "tune" ? "train" : r.kind)));
  const filters = [{ value: "all", label: "All" }, ...Object.entries(KINDS).filter(([k]) => k !== "tune" && present.has(k)).map(([k, [label]]) => ({ value: k, label }))];

  return (
    <main className="page">
      <h1>Runs</h1>
      <p className="lead">Everything you&apos;ve run, newest first. Open one to see its report and download its files.</p>
      {runs === null ? (
        <div className="loading">
          <span className="spin" />
          Loading…
        </div>
      ) : !runs.length ? (
        <div className="empty panel section">
          <div className="orb">
            <Icon name="sparkle" />
          </div>
          <h2>Nothing here yet</h2>
          <p className="sub">Upload a file or try an example to run your first analysis.</p>
          <p style={{ marginTop: 20 }}>
            <Link className="btn primary" href="/new/">
              <Icon name="upload" />
              Get started
            </Link>
          </p>
        </div>
      ) : (
        <>
          <div className="toolbar">
            <Segmented value={kind} onChange={setKind} choices={filters} label="Kind of run" />
            <input className="input" type="search" placeholder="Search runs" aria-label="Search runs" onChange={(e) => setQuery(e.target.value.toLowerCase())} />
          </div>
          <div className="runs">{shown.length ? shown.map((r) => <RunRow key={r.run} run={r} />) : <p className="muted">No runs match.</p>}</div>
        </>
      )}
    </main>
  );
}
