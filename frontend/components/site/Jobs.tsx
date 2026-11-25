"use client";

// The jobs as named moments: pick one and its real 3D view from a real run takes the stage. The views'
// data (public/data/visuals.json) loads when the section nears the screen.

import { useEffect, useRef, useState } from "react";
import type { Visual } from "@/lib/types";
import { SceneStage } from "./SceneStage";

type Key = "models" | "forecast" | "groups" | "unusual" | "columns";

const JOBS: { key: Key; title: string; about: string; file: string; still: string }[] = [
  { key: "models", title: "Predict a column", about: "Up to 17 models compared, the best one explained.", file: "churn.csv", still: "/img/bars-light.webp" },
  { key: "forecast", title: "Forecast over time", about: "One forecast per store or product, with a likely range.", file: "store_sales.csv", still: "/img/series-light.webp" },
  { key: "groups", title: "Find groups", about: "Segments in your rows, and what sets each one apart.", file: "customers.csv", still: "/img/points-light.webp" },
  { key: "unusual", title: "Find unusual rows", about: "Every row scored; the ones that don't fit are flagged.", file: "transactions.csv", still: "/img/points-light.webp" },
  { key: "columns", title: "Rank the columns", about: "Many methods agree on which columns matter.", file: "churn.csv", still: "/img/importance-light.webp" },
];

function readout(key: Key, v: Visual | undefined): string {
  if (!v) return "";
  if (v.type === "bars" && key === "models") {
    const best = v.bars.find((b) => b.tone === "best") || v.bars[0];
    const base = v.lines?.[0]?.value;
    return `${best.label} wins with ${best.value.toFixed(2)}${base != null ? `; a simple guess scores ${base.toFixed(2)}` : ""}.`;
  }
  if (v.type === "bars") {
    const kept = v.bars.filter((b) => b.tone === "picked").length;
    return `${kept} of ${v.bars.length} columns worth keeping.`;
  }
  if (v.type === "series") return `${v.lanes.length} forecasts, one each for ${v.lanes.map((l) => l.name).join(", ")}.`;
  if (v.type === "points" && key === "unusual") {
    const flagged = v.group.filter((g) => g === 1).length;
    return `${flagged} unusual rows flagged.`;
  }
  if (v.type === "points") return `${v.groups.length} groups of customers.`;
  return "";
}

export function Jobs() {
  const ref = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<Record<Key, Visual> | null>(null);
  const [on, setOn] = useState<Key>("models");
  const [failed, setFailed] = useState(false);

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
            className="jobs__item"
            tabIndex={j.key === on ? 0 : -1}
            onClick={() => setOn(j.key)}
            onKeyDown={(e) => {
              const i = JOBS.findIndex((x) => x.key === on);
              const to = e.key === "ArrowRight" || e.key === "ArrowDown" ? (i + 1) % JOBS.length : e.key === "ArrowLeft" || e.key === "ArrowUp" ? (i - 1 + JOBS.length) % JOBS.length : e.key === "Home" ? 0 : e.key === "End" ? JOBS.length - 1 : -1;
              if (to < 0) return;
              e.preventDefault();
              setOn(JOBS[to].key);
              document.getElementById(`job-${JOBS[to].key}`)?.focus();
            }}
          >
            <span className="jobs__title">{j.title}</span>
            <span className="jobs__about">{j.about}</span>
          </button>
        ))}
      </div>
      <div className="jobs__stage" id="job-stage" role="tabpanel" aria-labelledby={`job-${on}`}>
        <p className="jobs__now">{job.about}</p>
        <SceneStage
          key={on}
          className="stage--card"
          visual={data ? data[on] : null}
          spin={on === "groups" || on === "unusual"}
          far={1.15}
          forceStill={failed}
          still={job.still}
          label={`plainml's 3D view for ${job.title.toLowerCase()}, from ${job.file}`}
        />
        <p className="readout">
          <span className="readout__file">{job.file}</span>
          <span>{readout(on, data?.[on])}</span>
        </p>
      </div>
    </div>
  );
}
