"use client";

import Link from "next/link";
import type { RunRow as Run } from "@/lib/types";
import { METRIC_LABELS, num } from "@/lib/format";
import { Icon, type IconName } from "./Icon";

export const KINDS: Record<string, [string, IconName]> = {
  train: ["Model", "target"],
  tune: ["Tuned model", "target"],
  forecast: ["Forecast", "trend"],
  cluster: ["Groups", "groups"],
  anomaly: ["Anomalies", "alert"],
  importance: ["Importance", "bars"],
  drift: ["Drift", "drift"],
};

export function runTitle(run: Run): string {
  const [label] = KINDS[run.kind] || [run.kind];
  if (run.kind === "train" || run.kind === "tune") return `Predicting ${run.target}`;
  if (run.kind === "forecast") return `Forecasting ${run.target}`;
  if (run.kind === "importance") return `Columns that matter for ${run.target}`;
  return `${label} in ${run.data || "data"}`;
}

export function runHref(name: string): string {
  return `/run/?name=${encodeURIComponent(name)}`;
}

export function RunRow({ run }: { run: Run }) {
  const [label, icon] = KINDS[run.kind] || [run.kind, "file" as IconName];
  const hasScore = run.score !== null && run.score !== undefined && run.metric;
  return (
    <Link className="run" href={runHref(run.run)}>
      <span className="kind" title={label}>
        <Icon name={icon} />
      </span>
      <div style={{ minWidth: 0 }}>
        <b>{runTitle(run)}</b>
        <span className="m">{[label, run.data, run.best_model, run.created].filter(Boolean).join(" · ")}</span>
      </div>
      <div className="score">
        {hasScore ? (
          <>
            <b>{num(run.score)}</b>
            <span className="m">{METRIC_LABELS[run.metric!] || run.metric}</span>
          </>
        ) : null}
      </div>
    </Link>
  );
}
