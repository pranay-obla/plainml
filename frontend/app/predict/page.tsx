"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { useApp } from "@/components/AppContext";
import { DropZone } from "@/components/DropZone";
import { Icon, type IconName } from "@/components/Icon";
import { KINDS, runTitle } from "@/components/Runs";
import { Loading, Problem } from "@/components/States";
import { api, ApiError } from "@/lib/api";
import type { Job, RunRow, Upload } from "@/lib/types";
import { cx, num } from "@/lib/format";

function Predict() {
  const preselected = useSearchParams().get("model");
  const router = useRouter();
  const { fail, toast } = useApp();
  const [models, setModels] = useState<RunRow[] | null>(null);
  const [problem, setProblem] = useState<ApiError | null>(null);
  const [chosen, setChosen] = useState<RunRow | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const [horizon, setHorizon] = useState("");
  const [proba, setProba] = useState(false);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    api<RunRow[]>("/api/models")
      .then((list) => {
        setModels(list);
        setChosen(list.find((m) => m.run === preselected) || list[0] || null);
      })
      .catch(setProblem);
  }, [preselected]);

  if (problem) return <Problem error={problem} />;
  if (!models) return <Loading text="Loading models…" />;

  const forecast = chosen?.kind === "forecast";
  const run = async () => {
    if (!chosen) return;
    if (!forecast && !upload) return toast("Upload the rows to predict first.", null, true);
    setRunning(true);
    const options: Record<string, unknown> = { model: chosen.run };
    if (horizon) options.horizon = horizon;
    if (proba) options.proba = true;
    try {
      const job = await api<Job>("/api/jobs", { json: { task: "predict", upload: upload ? upload.id : null, options } });
      router.push(`/jobs/?id=${job.id}`);
    } catch (error) {
      fail(error);
      setRunning(false);
    }
  };

  return (
    <main className="page">
      <h1>Predict</h1>
      <p className="lead">Use a model you&apos;ve trained on new rows, or continue a forecast.</p>
      {!models.length || !chosen ? (
        <div className="empty panel section">
          <div className="orb">
            <Icon name="model" />
          </div>
          <h2>No models yet</h2>
          <p className="sub">Train a model, find groups, find anomalies or forecast first.</p>
          <p style={{ marginTop: 20 }}>
            <Link className="btn primary" href="/#start">
              <Icon name="upload" />
              Get started
            </Link>
          </p>
        </div>
      ) : (
        <>
          <div className="section">
            <div className="stepnum">
              <b>1</b>Choose a model
            </div>
            <div className="pick">
              {models.map((m) => {
                const [label, icon] = KINDS[m.kind] || [m.kind, "model" as IconName];
                return (
                  <button key={m.run} type="button" className={cx("run", m === chosen && "on")} aria-pressed={m === chosen} title={`${runTitle(m)} (${m.run})`} onClick={() => setChosen(m)}>
                    <span className="kind">
                      <Icon name={icon} />
                    </span>
                    <div style={{ minWidth: 0 }}>
                      <b>{runTitle(m)}</b>
                      <span className="m">{[label, m.best_model, m.created].filter(Boolean).join(" · ")}</span>
                    </div>
                    <span />
                  </button>
                );
              })}
            </div>
          </div>
          <div className="section">
            <div className="stepnum">
              <b>2</b>
              {forecast ? "Continue the forecast" : "Rows to predict"}
            </div>
            <div className="panel pad stack">
              {upload ? (
                <div className="dataset">
                  <div className="file-badge">
                    <Icon name="table" />
                  </div>
                  <div className="grow">
                    <div className="name">{upload.name}</div>
                    <div className="sub">
                      {num(upload.rows)} rows · {upload.columns.length} columns
                    </div>
                  </div>
                  <button className="btn small" type="button" onClick={() => setUpload(null)}>
                    Change
                  </button>
                </div>
              ) : (
                <DropZone compact label={forecast ? "Optional: drop newer history to continue from" : "Drop the rows to predict"} onFile={setUpload} />
              )}
              {forecast ? (
                <div className="form">
                  <div className="field">
                    <label htmlFor="horizon">How far ahead</label>
                    <input id="horizon" className="input" type="number" inputMode="decimal" placeholder="Same as when trained" value={horizon} onChange={(e) => setHorizon(e.target.value)} />
                  </div>
                </div>
              ) : null}
              {chosen.kind === "train" || chosen.kind === "tune" ? (
                <label className="switch">
                  <input type="checkbox" role="switch" checked={proba} onChange={(e) => setProba(e.target.checked)} />
                  <div>
                    <b>Add a probability for each class</b>
                    <span>For classification models.</span>
                  </div>
                </label>
              ) : null}
              <div className="actions">
                <button className="btn primary big" type="button" onClick={run} disabled={running}>
                  <Icon name="play" />
                  {running ? "Starting…" : forecast ? "Forecast" : "Predict"}
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </main>
  );
}

export default function PredictPage() {
  return (
    <Suspense fallback={<Loading />}>
      <Predict />
    </Suspense>
  );
}
