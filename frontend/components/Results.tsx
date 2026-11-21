"use client";

import { useEffect, useRef, useState } from "react";
import { api, isBrowserMode, openFile, readFile, saveFile } from "@/lib/api";
import type { FileEntry, OutputResult, Table, Tile } from "@/lib/types";
import { bytes } from "@/lib/format";
import { useApp } from "./AppContext";
import { DataTable } from "./Data";
import { Icon } from "./Icon";

export function Tiles({ tiles }: { tiles: Tile[] }) {
  return (
    <div className="section tiles">
      {tiles.map((tile, i) => (
        <div className="tile" key={i}>
          <div className="l">{tile.label}</div>
          <div className="v">{tile.value ?? "—"}</div>
          {tile.sub ? <div className="s">{tile.sub}</div> : null}
        </div>
      ))}
    </div>
  );
}

export function Findings({ sentences }: { sentences: string[] }) {
  if (!sentences.length) return null;
  return (
    <div className="section panel pad">
      <h3>Key findings</h3>
      <ul className="findings">
        {sentences.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
    </div>
  );
}

function FileRow({ file }: { file: FileEntry }) {
  const { fail } = useApp();
  const [preview, setPreview] = useState<Table | null>(null);
  const [loading, setLoading] = useState(false);
  const togglePreview = async () => {
    if (preview) return setPreview(null);
    setLoading(true);
    try {
      setPreview(await api<Table>(file.preview!));
    } catch (error) {
      fail(error);
    } finally {
      setLoading(false);
    }
  };
  const opens = ["html", "md", "txt", "json", "yaml"].includes(file.type);
  return (
    <div className="file">
      <div className={`ft ${file.type}`}>{file.type.slice(0, 4) || "file"}</div>
      <div className="t">
        <b>{file.label}</b>
        {file.description ? <span>{file.description}</span> : null}
        <span className="fname">
          {file.name} · {bytes(file.size)}
        </span>
      </div>
      <div className="fa">
        {file.preview ? (
          <button className="btn ghost small" type="button" aria-expanded={!!preview} onClick={togglePreview} title={`Preview ${file.name}`}>
            <Icon name="eye" />
            {loading ? "Loading…" : preview ? "Hide" : "Preview"}
          </button>
        ) : opens ? (
          <button className="btn ghost small" type="button" onClick={() => openFile(file.url).catch(fail)} title={`Open ${file.name} in a new tab`}>
            <Icon name="external" />
            Open
          </button>
        ) : null}
        <button className="btn small" type="button" onClick={() => saveFile(file.download, file.name).catch(fail)} title={`Download ${file.name}`}>
          <Icon name="download" />
          Download
        </button>
      </div>
      <div className="preview">{preview ? <DataTable table={preview} /> : null}</div>
    </div>
  );
}

export function FilesPanel({ files }: { files: FileEntry[] }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Files</h3>
        <span className="muted" style={{ fontSize: 13 }}>
          {files.length} file{files.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="files">
        {files.map((f) => (
          <FileRow key={f.path} file={f} />
        ))}
      </div>
    </div>
  );
}

const FRAME_STYLE = ".theme-toggle,header.top{display:none!important}main{padding-top:4px!important}body{background:transparent!important}";

function syncFrameTheme(frame: HTMLIFrameElement) {
  const doc = frame.contentDocument;
  if (!doc?.documentElement) return;
  const theme = document.documentElement.dataset.theme;
  if (theme) doc.documentElement.dataset.theme = theme;
  else delete doc.documentElement.dataset.theme;
}

export function ReportPanel({ url }: { url: string }) {
  const { fail } = useApp();
  const frame = useRef<HTMLIFrameElement>(null);
  const [html, setHtml] = useState<string | null>(null);
  const browser = isBrowserMode();

  useEffect(() => {
    if (!browser) return;
    readFile(url)
      .then((blob) => blob.text())
      .then(setHtml)
      .catch(fail);
  }, [url, browser, fail]);

  useEffect(() => {
    const sync = () => frame.current && syncFrameTheme(frame.current);
    window.addEventListener("plainml-theme", sync);
    return () => window.removeEventListener("plainml-theme", sync);
  }, []);

  const loaded = () => {
    const el = frame.current;
    const doc = el?.contentDocument;
    if (!el || !doc?.head) return;
    const style = doc.createElement("style");
    style.textContent = FRAME_STYLE;
    doc.head.append(style);
    syncFrameTheme(el);
    const fit = () => {
      el.style.height = `${doc.documentElement.scrollHeight + 4}px`;
    };
    fit();
    new ResizeObserver(fit).observe(doc.body);
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Report</h3>
        <button className="btn ghost small" type="button" onClick={() => openFile(url).catch(fail)}>
          <Icon name="external" />
          Open in a new tab
        </button>
      </div>
      {browser ? (
        html !== null ? <iframe ref={frame} className="report-frame" title="Report" srcDoc={html} onLoad={loaded} /> : <div className="loading" style={{ padding: 24 }}><span className="spin" />Loading the report…</div>
      ) : (
        <iframe ref={frame} className="report-frame" title="Report" src={url} onLoad={loaded} />
      )}
    </div>
  );
}

const OUTPUT_COLUMNS = /^(predicted_|probability_)|^(confidence|cluster|anomaly_score|is_anomaly|forecast|lower_80|upper_80)$/;

export function OutputView({ result }: { result: OutputResult }) {
  return (
    <div>
      <Tiles tiles={result.tiles} />
      {result.preview ? (
        <div className="section">
          <h2 style={{ marginBottom: 12 }}>Preview</h2>
          <DataTable table={result.preview} highlight={result.preview.columns.filter((c) => OUTPUT_COLUMNS.test(c))} />
        </div>
      ) : null}
      {result.report ? (
        <div className="result-grid">
          <ReportPanel url={result.report} />
          <aside>
            <FilesPanel files={result.files} />
          </aside>
        </div>
      ) : (
        <div className="files-wide">
          <FilesPanel files={result.files} />
        </div>
      )}
    </div>
  );
}
