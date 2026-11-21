"use client";

import type { Table, Upload } from "@/lib/types";
import { bytes, cx, KIND_LABEL, num } from "@/lib/format";
import { Icon } from "./Icon";

const SHOWN_KINDS = ["numeric", "categorical", "datetime", "text"];

export function DataTable({ table, highlight = [] }: { table: Table; highlight?: string[] }) {
  const numeric = table.columns.map((_, i) => table.rows.every((r) => r[i] === null || typeof r[i] === "number"));
  return (
    <div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              {table.columns.map((c) => (
                <th key={c} scope="col" className={cx(highlight.includes(c) && "hl")}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr key={r}>
                {row.map((v, i) => (
                  <td
                    key={i}
                    className={cx(v === null && "null", numeric[i] && "num", highlight.includes(table.columns[i]) && "hl") || undefined}
                    title={v === null ? undefined : String(v)}
                  >
                    {v === null ? "—" : typeof v === "number" ? num(v) : String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.total > table.rows.length ? (
        <div className="table-note">
          Showing {table.rows.length} of {num(table.total)} rows. Download the file for all of them.
        </div>
      ) : null}
    </div>
  );
}

export function DatasetPanel({ upload, onReplace }: { upload: Upload; onReplace: () => void }) {
  const kinds: Record<string, number> = {};
  for (const column of upload.columns) kinds[column.kind] = (kinds[column.kind] || 0) + 1;
  const unused = Object.keys(kinds).some((k) => !SHOWN_KINDS.includes(k));
  return (
    <div className="panel pad">
      <div className="dataset">
        <div className="file-badge">
          <Icon name="table" />
        </div>
        <div className="grow">
          <div className="name">{upload.name}</div>
          <div className="sub">
            {num(upload.rows)} rows · {upload.columns.length} columns · {bytes(upload.size)}
          </div>
        </div>
        <button className="btn small" type="button" onClick={onReplace}>
          <Icon name="upload" />
          Use another file
        </button>
      </div>
      {upload.sample ? (
        <p className="sample-note">
          <Icon name="sparkle" />
          <span>
            <b>Example data. </b>
            {upload.sample.about}
          </span>
        </p>
      ) : null}
      <div className="pills" style={{ marginTop: 16 }}>
        {upload.columns.map((c) => (
          <span
            key={c.name}
            className="pill"
            title={`${c.name}: ${KIND_LABEL[c.kind] || c.kind}${c.missing_pct ? `, ${c.missing_pct}% blank` : ""}${c.note ? ` (${c.note})` : ""}`}
          >
            <i className={`dot k-${c.kind}`} />
            <b>{c.name}</b>
          </span>
        ))}
      </div>
      <div className="legend">
        {SHOWN_KINDS.filter((k) => kinds[k]).map((k) => (
          <span key={k}>
            <i className={`dot k-${k}`} />
            {KIND_LABEL[k]} ({kinds[k]})
          </span>
        ))}
        {unused ? (
          <span>
            <i className="dot" />
            not used as input
          </span>
        ) : null}
      </div>
      <details className="more" style={{ marginTop: 14 }}>
        <summary>Preview the first {upload.preview.rows.length} rows</summary>
        <div style={{ marginTop: 12 }}>
          <DataTable table={upload.preview} />
        </div>
      </details>
    </div>
  );
}
