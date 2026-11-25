"use client";

// The eight jobs as named moments. Five have a real 3D view from a real run (public/data/visuals.json,
// loaded when the section nears the screen); the other three give a report, said in words.

import { useEffect, useRef, useState } from "react";
import type { Visual } from "@/lib/types";
import { SceneStage } from "./SceneStage";

type Key = "models" | "forecast" | "groups" | "unusual" | "columns" | "drift" | "profile" | "clean";
type Job = { key: Key; title: string; about: string; file?: string; still?: string; far?: number; report?: string[] };

const JOBS: Job[] = [
  { key: "models", title: "Predict a column", about: "Up to 17 models compared, the best one explained.", file: "churn.csv", still: "/img/still-models.jpg", far: 1.2 },
  { key: "forecast", title: "Forecast over time", about: "One forecast per store or product, with a likely range.", file: "store_sales.csv", still: "/img/still-forecast.jpg", far: 1.35 },
  { key: "groups", title: "Find groups", about: "Segments in your rows, and what sets each one apart.", file: "customers.csv", still: "/img/still-groups.jpg", far: 1.3 },
  { key: "unusual", title: "Find unusual rows", about: "Every row scored; the ones that don't fit are flagged.", file: "transactions.csv", still: "/img/still-unusual.jpg", far: 1.45 },
  { key: "columns", title: "Rank the columns", about: "Eight methods vote on which columns matter.", file: "churn.csv", still: "/img/still-columns.jpg", far: 1.2 },
  { key: "drift", title: "Check for drift", about: "Has new data moved away from what a model learned?", report: ["Compares a new file with the data a model was trained on.", "Each column is marked stable, a moderate change or a major change."] },
  { key: "profile", title: "Profile the data", about: "Every column summarised, problems flagged first.", report: ["Types, blanks, duplicates and odd values, column by column.", "Saved as a report you can open in any browser."] },
  { key: "clean", title: "Clean the data", about: "A tidied copy to download.", report: ["Fixes types, blanks, duplicates and odd values.", "Then gives you the clean file to download."] },
];

function readout(key: Key, v: Visual | undefined): string {
  if (!v) return "";
  if (v.type === "bars" && key === "models") {
    const best = v.bars.find((b) => b.tone === "best") || v.bars[0];
    return `${best.label} comes out on top of ${v.bars.length} models. Taller is better; the dashed line is a simple guess.`;
  }
  if (v.type === "bars") {
    const kept = v.bars.filter((b) => b.tone === "picked").length;
    return `${kept} of ${v.bars.length} columns worth keeping, by the vote of eight methods.`;
  }
  if (v.type === "series") return `${v.lanes.length} forecasts, one each for ${v.lanes.map((l) => l.name).join(", ")}: the history, then the forecast with a likely range.`;
  if (v.type === "points" && key === "unusual") {
    const flagged = v.group.filter((g) => g === 1).length;
    return `${flagged} unusual payments flagged in gold.`;
  }
  if (v.type === "points") return `${v.groups.length} groups of customers, each its own colour.`;
  return "";
}

export function Jobs() {
  const ref = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<Record<string, Visual> | null>(null);
  const [failed, setFailed] = useState(false);
  const [on, setOn] = useState<Key>("models");

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        io.disconnect();
        fetch("/data/visuals.json")
          .then((r) => r.json())
          .then(setData)
          .catch(() => setFailed(true));
      },
      { rootMargin: "600px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const job = JOBS.find((j) => j.key === on) || JOBS[0];
  const move = (e: React.KeyboardEvent) => {
    const i = JOBS.findIndex((x) => x.key === on);
    const n = JOBS.length;
    const to = e.key === "ArrowRight" || e.key === "ArrowDown" ? (i + 1) % n : e.key === "ArrowLeft" || e.key === "ArrowUp" ? (i - 1 + n) % n : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : -1;
    if (to < 0) return;
    e.preventDefault();
    setOn(JOBS[to].key);
    document.getElementById(`job-${JOBS[to].key}`)?.focus();
  };

  return (
    <div className="jobs" ref={ref}>
      <div className="jobs__list" role="tablist" aria-label="Jobs">
        {JOBS.map((j) => (
          <button
            key={j.key}
            type="button"
            role="tab"
            id={`job-${j.key}`}
            aria-selected={j.key === on}
            aria-controls="job-stage"
            tabIndex={j.key === on ? 0 : -1}
            className="jobs__item"
            onClick={() => setOn(j.key)}
            onKeyDown={move}
          >
            {j.title}
          </button>
        ))}
      </div>
      <div className="jobs__stage" id="job-stage" role="tabpanel" aria-labelledby={`job-${on}`}>
        <p className="jobs__now">{job.about}</p>
        {job.report ? (
          <ul className="jobs__report">
            {job.report.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        ) : (
          <>
            <SceneStage
              key={on}
              className="stage--card"
              visual={data ? data[on] : null}
              tone="band"
              spin={on === "groups" || on === "unusual"}
              far={job.far}
              forceStill={failed}
              still={job.still || ""}
              label={`plainml's 3D view for ${job.title.toLowerCase()}, from ${job.file}`}
            />
            <p className="readout">
              <span className="readout__file">{job.file}</span>
              <span>{failed ? "The 3D view didn't load; this is a still of it." : data ? readout(on, data[on]) : "Loading the 3D view…"}</span>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
