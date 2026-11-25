"use client";

// The start of every analysis (/new/): drop a file, or load one of the example datasets.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { RunRow as Run, Sample, Upload } from "@/lib/types";
import { cx, num } from "@/lib/format";
import samples from "@/lib/samples.json";
import { useApp } from "./AppContext";
import { DropZone } from "./DropZone";
import { Icon } from "./Icon";
import { TASKS } from "./Options";
import { RunRow } from "./Runs";

function analyze(router: ReturnType<typeof useRouter>, upload: Upload) {
  router.push(`/analyze/?upload=${upload.id}`);
}

function Examples() {
  const router = useRouter();
  const { browser, boot, fail } = useApp();
  const [busy, setBusy] = useState<string | null>(null);
  const use = async (sample: Sample) => {
    if (busy) return;
    setBusy(sample.key);
    try {
      analyze(router, await api<Upload>(`/api/samples/${encodeURIComponent(sample.key)}`, { method: "POST" }));
    } catch (error) {
      fail(error);
      setBusy(null);
    }
  };
  const waiting = browser && !boot?.ready;
  return (
    <div className={cx("cards", "examples", busy && "busy")} role="list" id="examples">
      {(samples as Sample[]).map((sample) => {
        const task = TASKS.find((t) => t.key === sample.task) || TASKS[0];
        const mine = busy === sample.key;
        return (
          <button key={sample.key} type="button" role="listitem" className={cx("card", mine && "on")} onClick={() => use(sample)} title={sample.about}>
            <div className="ico">{mine ? <span className="spin" aria-label="Loading" /> : <Icon name={task.icon} />}</div>
            <Icon name="arrow" className="go" />
            <b>{sample.title}</b>
            <p>{sample.question}</p>
            <span className="meta">{mine && waiting ? "Waiting for Python to finish loading…" : `${task.title} · ${num(sample.rows)} rows`}</span>
          </button>
        );
      })}
    </div>
  );
}

function RecentRuns() {
  const { info } = useApp();
  const [runs, setRuns] = useState<Run[]>([]);
  useEffect(() => {
    if (info) api<Run[]>("/api/runs").then(setRuns).catch(() => setRuns([]));
  }, [info]);
  if (!runs.length) return null;
  return (
    <section className="section">
      <div className="section-head">
        <h2>Your recent results</h2>
        <Link href="/runs/">All runs →</Link>
      </div>
      <div className="runs">
        {runs.slice(0, 4).map((r) => (
          <RunRow key={r.run} run={r} />
        ))}
      </div>
    </section>
  );
}

export function Start() {
  const router = useRouter();
  return (
    <main className="page">
      <h1>New analysis</h1>
      <p className="lead">Drop in a spreadsheet, or click an example: it loads made-up data and fills in the settings, so you only press Run.</p>
      <div className="start-grid section">
        <DropZone onFile={(upload) => analyze(router, upload)} />
        <Examples />
      </div>
      <RecentRuns />
    </main>
  );
}
