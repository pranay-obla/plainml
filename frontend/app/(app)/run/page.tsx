"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Icon } from "@/components/Icon";
import { FilesPanel, Findings, ReportPanel, Tiles } from "@/components/Results";
import { Loading, Problem } from "@/components/States";
import { VisualPanel } from "@/components/three/VisualPanel";
import { api, ApiError } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

function RunView() {
  const name = useSearchParams().get("name") || "";
  const [run, setRun] = useState<RunSummary | null>(null);
  const [problem, setProblem] = useState<ApiError | null>(null);

  useEffect(() => {
    setRun(null);
    if (!name) return setProblem(new ApiError("No run chosen.", "Pick one from the Runs page."));
    api<RunSummary>(`/api/runs/${encodeURIComponent(name)}`)
      .then(setRun)
      .catch(setProblem);
  }, [name]);

  if (problem) return <Problem error={problem} />;
  if (!run) return <Loading text="Loading the run…" />;
  return (
    <main className="page">
      <div className="crumbs">
        <Link href="/runs/">Runs</Link>/ {run.name}
      </div>
      <div className="job-head">
        <div>
          <h1>{run.title}</h1>
          <p className="sub">{run.subtitle}</p>
        </div>
        {run.predictable ? (
          <Link className="btn primary" href={`/predict/?model=${encodeURIComponent(run.name)}`}>
            <Icon name="sparkle" />
            {run.kind === "forecast" ? "Forecast again" : "Predict with this model"}
          </Link>
        ) : null}
      </div>
      <Tiles tiles={run.tiles} />
      <Findings sentences={run.sentences} />
      {run.visual ? <VisualPanel visual={run.visual} /> : null}
      {run.report ? (
        <div className="result-grid">
          <ReportPanel url={run.report} />
          <aside>
            <FilesPanel files={run.files} />
          </aside>
        </div>
      ) : (
        <div className="files-wide">
          <FilesPanel files={run.files} />
        </div>
      )}
    </main>
  );
}

export default function RunPage() {
  return (
    <Suspense fallback={<Loading />}>
      <RunView />
    </Suspense>
  );
}
