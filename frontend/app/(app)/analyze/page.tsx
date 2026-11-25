"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { useApp } from "@/components/AppContext";
import { DatasetPanel } from "@/components/Data";
import { Icon } from "@/components/Icon";
import { cleanOptions, defaultsFor, OptionsForm, TASKS, type Options, type Task } from "@/components/Options";
import { Loading, Problem } from "@/components/States";
import { api, ApiError } from "@/lib/api";
import type { Job, Upload } from "@/lib/types";
import { cx } from "@/lib/format";

function Analyze() {
  const params = useSearchParams();
  const router = useRouter();
  const { info, fail } = useApp();
  const id = params.get("upload");
  const [upload, setUpload] = useState<Upload | null>(null);
  const [problem, setProblem] = useState<ApiError | null>(null);
  const [task, setTask] = useState<Task | null>(null);
  const [options, setOptions] = useState<Options>({});
  const [running, setRunning] = useState(false);
  const optionsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!id) return setProblem(new ApiError("No data file chosen yet.", "Upload one or pick an example on the home page."));
    setUpload(null);
    api<Upload>(`/api/uploads/${encodeURIComponent(id)}`)
      .then(setUpload)
      .catch(() => setProblem(new ApiError("That upload is gone.", "Upload the file again, or pick an example.")));
  }, [id]);

  const pick = (next: Task, scroll = true) => {
    if (!upload || !info) return;
    setTask(next);
    setOptions(defaultsFor(next.key, upload, info));
    if (scroll) setTimeout(() => optionsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 30);
  };

  useEffect(() => {
    if (upload && info && !task) {
      const preset = TASKS.find((t) => t.key === (params.get("task") || upload.sample?.task));
      if (preset) pick(preset, false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upload, info]);

  if (problem) return <Problem error={problem} />;
  if (!upload || !info) return <Loading text={info ? "Loading your data…" : "Starting plainml…"} />;

  const run = async () => {
    if (!task) return;
    setRunning(true);
    try {
      const job = await api<Job>("/api/jobs", { json: { task: task.key, upload: upload.id, options: cleanOptions(options) } });
      router.push(`/jobs/?id=${job.id}`);
    } catch (error) {
      fail(error);
      setRunning(false);
    }
  };
  const fromSample = upload.sample && task && upload.sample.task === task.key;

  return (
    <main className="page">
      <DatasetPanel upload={upload} onReplace={() => router.push("/new/")} />
      <div className="section">
        <div className="section-head">
          <h2>What do you want to find out?</h2>
        </div>
        <div className="cards" role="list">
          {TASKS.map((t) => (
            <button key={t.key} type="button" role="listitem" className={cx("card", task?.key === t.key && "on")} onClick={() => pick(t)} aria-pressed={task?.key === t.key}>
              <div className="ico">
                <Icon name={t.icon} />
              </div>
              <b>{t.title}</b>
              <p>{t.about}</p>
            </button>
          ))}
        </div>
      </div>
      {task ? (
        <div className="section" ref={optionsRef} style={{ scrollMarginTop: 90 }}>
          <div className="panel pad">
            <div className="options-title">
              <div className="file-badge">
                <Icon name={task.icon} />
              </div>
              <div>
                <h2>{task.title}</h2>
                <div className="sub">{task.about}</div>
              </div>
            </div>
            <div className="form">
              <OptionsForm task={task.key} upload={upload} info={info} options={options} set={(name, value) => setOptions((o) => ({ ...o, [name]: value }))} />
            </div>
            <div className="actions">
              <button className="btn primary big" type="button" onClick={run} disabled={running}>
                <Icon name="play" />
                {running ? "Starting…" : `Run: ${task.title.toLowerCase()}`}
              </button>
              <span className="muted">{fromSample ? "Filled in for this example: press Run, or change anything first." : "Results are saved as a run you can come back to."}</span>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={<Loading />}>
      <Analyze />
    </Suspense>
  );
}
