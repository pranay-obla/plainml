"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import type { Visual } from "@/lib/types";
import { num } from "@/lib/format";
import { Segmented } from "../Options";
import { hasWebGL, seriesColor, usePalette } from "./theme";

const Scene = dynamic(() => import("./Scenes"), {
  ssr: false,
  loading: () => (
    <div className="loading" style={{ justifyContent: "center", height: "100%" }}>
      <span className="spin" />
      Building the 3D view…
    </div>
  ),
});

function Legend({ visual }: { visual: Visual }) {
  const palette = usePalette();
  if (!palette) return null;
  if (visual.type === "points") {
    return (
      <div className="visual-legend">
        {visual.groups.map((g, i) => (
          <span key={g}>
            <i style={{ background: visual.size ? (i === 1 ? palette.highlight : palette.other) : seriesColor(palette, i) }} />
            {g}
          </span>
        ))}
      </div>
    );
  }
  if (visual.type === "series") {
    return (
      <div className="visual-legend">
        <span>
          <i style={{ background: palette.ink2 }} />
          History
        </span>
        <span>
          <i style={{ background: palette.highlight }} />
          Forecast
        </span>
        <span>
          <i style={{ background: palette.highlight, opacity: 0.3 }} />
          Likely range (80%)
        </span>
      </div>
    );
  }
  const tones = new Set(visual.bars.map((b) => b.tone || "plain"));
  const names: Record<string, [string, string]> = {
    best: ["Best", palette.highlight],
    picked: ["Worth keeping", palette.highlight],
    warn: ["Moderate change", palette.warn],
    bad: ["Major change", palette.bad],
    plain: [visual.bars.some((b) => b.tone === "picked") ? "Others" : visual.bars.some((b) => b.tone === "best") ? "Other models" : "Stable", palette.other],
  };
  return (
    <div className="visual-legend">
      {[...tones].map((t) => (
        <span key={t}>
          <i style={{ background: names[t][1] }} />
          {names[t][0]}
        </span>
      ))}
      {(visual.lines || []).map((l) => (
        <span key={l.label}>
          <i style={{ background: "transparent", border: `1.5px dashed ${palette.ink2}` }} />
          {l.label}
        </span>
      ))}
    </div>
  );
}

function TableView({ visual }: { visual: Visual }) {
  if (visual.type === "bars") {
    return (
      <div className="table-wrap" style={{ maxHeight: 420, margin: 16 }}>
        <table className="data">
          <thead>
            <tr>
              <th>Name</th>
              <th>{visual.value_label}</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {visual.bars.map((b) => (
              <tr key={b.label}>
                <td>{b.label}</td>
                <td className="num">{num(b.value)}</td>
                <td>{b.note || ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (visual.type === "points") {
    const counts = visual.groups.map((_, i) => visual.group.filter((g) => g === i).length);
    return (
      <div className="table-wrap" style={{ maxHeight: 420, margin: 16 }}>
        <table className="data">
          <thead>
            <tr>
              <th>Group</th>
              <th>Points shown</th>
            </tr>
          </thead>
          <tbody>
            {visual.groups.map((g, i) => (
              <tr key={g}>
                <td>{g}</td>
                <td className="num">{num(counts[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  return (
    <div className="table-wrap" style={{ maxHeight: 420, margin: 16 }}>
      <table className="data">
        <thead>
          <tr>
            <th>Series</th>
            <th>Last value</th>
            <th>Forecast, end</th>
            <th>Likely range</th>
          </tr>
        </thead>
        <tbody>
          {visual.lanes.map((lane) => {
            const end = lane.forecast.y.length - 1;
            return (
              <tr key={lane.name}>
                <td>{lane.name}</td>
                <td className="num">{num(lane.history.y[lane.history.y.length - 1])}</td>
                <td className="num">{num(lane.forecast.y[end])}</td>
                <td className="num">{lane.forecast.lower && lane.forecast.upper ? `${num(lane.forecast.lower[end])} – ${num(lane.forecast.upper[end])}` : ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function VisualPanel({ visual }: { visual: Visual }) {
  const [view, setView] = useState("3d");
  const [webgl, setWebgl] = useState(true);
  useEffect(() => {
    if (!hasWebGL()) {
      setWebgl(false);
      setView("table");
    }
  }, []);
  return (
    <section className="panel visual" aria-label={visual.title}>
      <div className="panel-head">
        <h3>{visual.title}</h3>
        {webgl ? <Segmented value={view} onChange={setView} label="View" choices={[{ value: "3d", label: "3D" }, { value: "table", label: "Table" }]} /> : null}
      </div>
      {view === "3d" ? (
        <div className="visual-stage">
          <Scene visual={visual} />
          <span className="visual-hint">Drag to turn · scroll to zoom · hover for details</span>
        </div>
      ) : (
        <TableView visual={visual} />
      )}
      <div className="visual-foot">
        <Legend visual={visual} />
        <span>{visual.caption}</span>
      </div>
    </section>
  );
}
