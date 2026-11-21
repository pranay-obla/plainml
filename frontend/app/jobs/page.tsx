"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { useApp } from "@/components/AppContext";
import { OutputView } from "@/components/Results";
import { runHref } from "@/components/Runs";
import { Loading, Problem } from "@/components/States";
import { api, ApiError } from "@/lib/api";
import type { Job } from "@/lib/types";
import { cx, duration } from "@/lib/format";

function JobView() {
  const id = useSearchParams().get("id") || "";
  const router = useRouter();
  const { toast } = useApp();
  const [job, setJob] = useState<Job | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [problem, setProblem] = useState<ApiError | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let from = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    setLines([]);
    const tick = async () => {
      let current: Job;
      try {
        current = await api<Job>(`/api/jobs/${id}?log_from=${from}`);
      } catch (error) {
        return setProblem(error as ApiError);
      }
      if (stopped) return;
      if (current.log.length) setLines((all) => [...all, ...current.log]);
      from = current.log_size;
      setJob(current);
      if (current.status === "queued" || current.status === "running") {
        timer = setTimeout(tick, 700);
      } else if (current.status === "done" && current.result?.kind === "run") {
        toast(`${current.title}: done in ${duration(current.elapsed)}`);
        router.replace(runHref(current.result.run));
      }
    };
    tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [id, router, toast]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines]);

  if (problem) return <Problem error={problem} />;
  if (!job) return <Loading text="Starting the job…" />;

  const active = job.status === "queued" || job.status === "running";
  const share = job.progress !== null && job.status === "running" ? Math.round(job.progress * 100) : null;
  return (
    <main className="page">
      <div className="crumbs">
        <Link href="/runs/">Runs</Link>/ Job
      </div>
      <div className="job-head">
        <h1>{job.title}</h1>
        <span className={cx("status", job.status === "done" && "done", job.status === "failed" && "failed")}>
          {active ? <i className="pulse" /> : null}
          {job.status === "queued" ? "Queued" : job.status === "running" ? "Running" : job.status === "failed" ? "Failed" : `Done in ${duration(job.elapsed)}`}
        </span>
      </div>
      <div className="panel pad">
        <div className="stage">{job.status === "failed" ? "Something went wrong" : job.stage}</div>
        <div className={cx("progress", active && share === null && "indeterminate")} role="progressbar" aria-label="Progress" aria-valuenow={share ?? undefined}>
          <i style={{ width: active ? (share === null ? undefined : `${share}%`) : "100%", background: job.status === "failed" ? "var(--bad)" : undefined }} />
        </div>
        <div className="meta-row">
          <span>{job.status === "queued" ? "" : `${duration(job.elapsed)} elapsed`}</span>
          <span>{share !== null ? `${share}%` : ""}</span>
        </div>
        <details className="more" open={active || job.status === "failed"}>
          <summary>Live log</summary>
          <div className="log" role="log" ref={logRef}>
            {lines.map((line, i) => (
              <div key={i} className={line.startsWith("▸") ? "st" : /^\s/.test(line) ? "dim" : undefined}>
                {line || " "}
              </div>
            ))}
          </div>
        </details>
        {job.status === "failed" && job.error ? (
          <div className="error-box">
            <b>{job.error.message}</b>
            {job.error.hint ? <div>{job.error.hint}</div> : null}
          </div>
        ) : null}
      </div>
      {job.status === "done" && job.result?.kind === "output" ? <OutputView result={job.result} /> : null}
    </main>
  );
}

export default function JobPage() {
  return (
    <Suspense fallback={<Loading />}>
      <JobView />
    </Suspense>
  );
}
