"""Self-contained HTML reports: one file, no internet needed, light and dark mode.

Charts are drawn in the browser from JSON embedded in the page, so they resize to any
screen. Every chart also has a table view, and every value shown in a tooltip is also
visible as a label or in a table.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plainml import __version__
from plainml.console import fmt_duration, fmt_num, fmt_pct
from plainml.runs import REPORT_FILE, load_run

# --- small HTML helpers -------------------------------------------------------------------


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return fmt_num(float(value))
    return "—" if value is None else str(value)


def section(title: str, body: str, intro: str | None = None, anchor: str | None = None) -> str:
    anchor_attr = f' id="{esc(anchor)}"' if anchor else ""
    intro_html = f'<p class="intro">{intro}</p>' if intro else ""
    return f"<section{anchor_attr}><h2>{esc(title)}</h2>{intro_html}{body}</section>"


def card(body: str, title: str | None = None, note: str | None = None, wide: bool = False) -> str:
    head = f"<h3>{esc(title)}</h3>" if title else ""
    foot = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="card{" wide" if wide else ""}">{head}{body}{foot}</div>'


def grid(*cards: str) -> str:
    return f'<div class="grid">{"".join(cards)}</div>'


def table(
    headers: list[str],
    rows: list[list[Any]],
    numeric: set[int] | None = None,
    highlight: int | None = None,
) -> str:
    numeric = numeric or set()
    head = "".join(
        f"<th{' class=num' if i in numeric else ''}>{esc(h)}</th>" for i, h in enumerate(headers)
    )
    body = []
    for r, row in enumerate(rows):
        cells = "".join(
            f"<td{' class=num' if i in numeric else ''}>{esc(fmt(v)) if not isinstance(v, RawHTML) else v}</td>"
            for i, v in enumerate(row)
        )
        body.append(f"<tr{' class=hl' if r == highlight else ''}>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


class RawHTML(str):
    """Marks a table cell that's already HTML."""


def as_table_view(inner: str) -> str:
    return f'<details class="table-view"><summary>Show as a table</summary>{inner}</details>'


def stat_tiles(tiles: Sequence[tuple[str, str, str | None]], hero_first: bool = True) -> str:
    items = []
    for i, (label, value, sub) in enumerate(tiles):
        cls = "tile hero" if hero_first and i == 0 else "tile"
        sub_html = f'<div class="tile-sub">{sub}</div>' if sub else ""
        items.append(
            f'<div class="{cls}"><div class="tile-label">{esc(label)}</div><div class="tile-value">{esc(value)}</div>{sub_html}</div>'
        )
    return f'<div class="tiles">{"".join(items)}</div>'


def bar_list(
    items: list[dict[str, Any]],
    *,
    value_format: str = "num",
    emphasis: set[int] | None = None,
    muted: set[int] | None = None,
) -> str:
    """Horizontal bars drawn in HTML (responsive, keyboard focusable, labelled at the tip).

    items: {"label", "value", "err" (optional ±), "tip" (optional extra text)}
    """
    emphasis = emphasis if emphasis is not None else {0}
    muted = muted or set()
    values = [
        float(i["value"]) for i in items if i.get("value") is not None and not _nan(i["value"])
    ]
    if not values:
        return '<p class="note">No values to show.</p>'
    errs = [float(i.get("err") or 0) for i in items]
    lo = min(0.0, min(v - e for v, e in zip(values, errs, strict=False)))
    hi = max(0.0, max(v + e for v, e in zip(values, errs, strict=False)))
    span = (hi - lo) or 1.0

    def pos(v: float) -> float:
        return (v - lo) / span * 100

    zero = pos(0.0)
    zero_line = f'<span class="zero" style="left:{zero:.2f}%"></span>' if lo < 0 else ""
    rows = []
    for index, item in enumerate(items):
        value = item.get("value")
        if value is None or _nan(value):
            continue
        value = float(value)
        shown = fmt_pct(value, 1) if value_format == "pct" else fmt_num(value)
        err = float(item.get("err") or 0)
        left, width = (zero, pos(value) - zero) if value >= 0 else (pos(value), zero - pos(value))
        side = "pos" if value >= 0 else "neg"
        tone = "emph" if index in emphasis else ("muted" if index in muted else "")
        whisker = ""
        if err > 0:
            w_left = pos(value - err)
            whisker = f'<span class="whisker" style="left:{w_left:.2f}%;width:{pos(value + err) - w_left:.2f}%"></span>'
        tip_value = shown + (f" ± {fmt_num(err)}" if err > 0 else "")
        extra = f' data-tip-extra="{esc(item["tip"])}"' if item.get("tip") else ""
        rows.append(
            f'<div class="bar-row" tabindex="0" data-tip-value="{esc(tip_value)}" data-tip-label="{esc(item["label"])}"{extra}>'
            f'<span class="bar-label" title="{esc(item["label"])}">{esc(item["label"])}</span>'
            f'<span class="bar-track">{zero_line}'
            f'<span class="bar {side} {tone}" style="left:{left:.2f}%;width:{max(width, 0.4):.2f}%"></span>{whisker}</span>'
            f'<span class="bar-value">{esc(shown)}</span></div>'
        )
    return f'<div class="bars">{"".join(rows)}</div>'


def _nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def heatmap(labels: list[str], matrix: list[list[int]]) -> str:
    """Confusion matrix: rows are actual classes, columns predicted; shaded by row share."""
    matrix_np = np.asarray(matrix, dtype=float)
    totals = matrix_np.sum(axis=1, keepdims=True)
    shares = np.divide(matrix_np, totals, out=np.zeros_like(matrix_np), where=totals > 0)
    head = "".join(f'<th class="hm-col" title="{esc(lab)}">{esc(lab)}</th>' for lab in labels)
    rows = []
    for i, label in enumerate(labels):
        cells = []
        for j in range(len(labels)):
            share = float(shares[i, j])
            strong = " strong" if share > 0.5 else ""
            diag = " diag" if i == j else ""
            cells.append(
                f'<td class="hm-cell{strong}{diag}" style="--p:{share * 100:.1f}%" tabindex="0" '
                f'data-tip-value="{int(matrix_np[i, j]):,} rows ({fmt_pct(share, 0)})" '
                f'data-tip-label="actual {esc(label)} → predicted {esc(labels[j])}">'
                f'<span class="hm-count">{int(matrix_np[i, j]):,}</span><span class="hm-share">{fmt_pct(share, 0)}</span></td>'
            )
        rows.append(
            f'<tr><th class="hm-row" title="{esc(label)}">{esc(label)}</th>{"".join(cells)}</tr>'
        )
    return (
        '<div class="table-wrap"><table class="heatmap"><thead><tr><th class="hm-corner">actual ↓ · predicted →</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def heat_table(
    row_labels: list[str],
    col_labels: list[str],
    cells: list[list[tuple[str, float]]],
    corner: str = "",
) -> str:
    """A shaded grid. Each cell is (text, strength 0–1); darker means stronger."""
    head = "".join(f'<th class="hm-col" title="{esc(c)}">{esc(c)}</th>' for c in col_labels)
    rows = []
    for label, row in zip(row_labels, cells, strict=False):
        tds = []
        for (text, strength), column in zip(row, col_labels, strict=False):
            share = max(0.0, min(1.0, float(strength)))
            strong = " strong" if share > 0.5 else ""
            tds.append(
                f'<td class="hm-cell{strong}" style="--p:{share * 100:.1f}%" tabindex="0" '
                f'data-tip-value="{esc(text)}" data-tip-label="{esc(label)} · {esc(column)}">'
                f'<span class="hm-count">{esc(text)}</span></td>'
            )
        rows.append(
            f'<tr><th class="hm-row" title="{esc(label)}">{esc(label)}</th>{"".join(tds)}</tr>'
        )
    return (
        f'<div class="table-wrap"><table class="heatmap compact"><thead><tr><th class="hm-corner">{esc(corner)}</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def chart(chart_id: str, height: int | None = None) -> str:
    style = f' style="min-height:{height}px"' if height else ""
    return f'<div class="chart" data-chart="{esc(chart_id)}"{style}></div>'


def issues_list(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return '<p class="note">No problems found.</p>'
    items = []
    for issue in issues:
        warn = issue.get("level") == "warn"
        icon = "!" if warn else "i"
        label = "Warning" if warn else "Note"
        hint = (
            f'<div class="issue-hint">{esc(issue.get("hint"))}</div>' if issue.get("hint") else ""
        )
        items.append(
            f'<li class="issue {"warn" if warn else "info"}"><span class="issue-icon" aria-hidden="true">{icon}</span>'
            f'<div><span class="sr">{label}: </span>{esc(issue["message"])}{hint}</div></li>'
        )
    return f'<ul class="issues">{"".join(items)}</ul>'


def sentences_list(sentences: list[str]) -> str:
    return "<ul class='plain'>" + "".join(f"<li>{esc(s)}</li>" for s in sentences) + "</ul>"


def histogram_bins(values: list[float], max_bins: int = 30) -> list[dict[str, float]]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return []
    bins = min(max_bins, max(5, int(np.sqrt(array.size))))
    counts, edges = np.histogram(array, bins=bins)
    return [
        {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


# --- page -----------------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--accent:#2a78d6;--accent-wash:rgba(42,120,214,.10);
--deemph:#aeaca4;--seq-lo:#eef4fc;--seq-hi:#104281;--hm-ink:#0b0b0b;--good:#006300;--warn:#fab219;--warn-ink:#7a5000;
--info:#2a78d6;--tip-bg:#0b0b0b;--tip-ink:#ffffff;--code:#f0efec}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--accent:#3987e5;
--accent-wash:rgba(57,135,229,.16);--deemph:#6b6a64;--seq-lo:#1f2329;--seq-hi:#3987e5;--hm-ink:#ffffff;--good:#0ca30c;
--warn:#fab219;--warn-ink:#fab219;--info:#3987e5;--tip-bg:#ffffff;--tip-ink:#0b0b0b;--code:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--accent:#3987e5;--accent-wash:rgba(57,135,229,.16);--deemph:#6b6a64;
--seq-lo:#1f2329;--seq-hi:#3987e5;--hm-ink:#ffffff;--good:#0ca30c;--warn:#fab219;--warn-ink:#fab219;--info:#3987e5;
--tip-bg:#ffffff;--tip-ink:#0b0b0b;--code:#262624}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}
header.top{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;margin-bottom:20px}
h1{font-size:26px;line-height:1.25;margin:0 0 4px;font-weight:650;letter-spacing:-.01em}
.subtitle{color:var(--ink-2);margin:0;font-size:14px}
h2{font-size:19px;margin:0 0 6px;font-weight:620}
h3{font-size:14px;margin:0 0 12px;font-weight:600;color:var(--ink-2)}
section{margin-top:36px}
.intro{color:var(--ink-2);margin:0 0 14px;max-width:760px}
.note{color:var(--muted);font-size:13px;margin:10px 0 0}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 18px 16px;min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card.wide{grid-column:1/-1}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;margin:4px 0 0}
nav.toc a{color:var(--ink-2);text-decoration:none;border-bottom:1px solid var(--grid)}
nav.toc a:hover{color:var(--ink);border-color:var(--ink-2)}
.theme-toggle{flex:none;border:1px solid var(--border);background:var(--surface);color:var(--ink-2);border-radius:8px;padding:6px 10px;font:inherit;font-size:13px;cursor:pointer}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;min-width:0}
.tile-label{font-size:13px;color:var(--ink-2)}
.tile-value{font-size:22px;font-weight:620;margin-top:2px;overflow-wrap:anywhere}
.tile.hero .tile-value{font-size:40px;line-height:1.15;letter-spacing:-.02em}
.tile-sub{font-size:13px;color:var(--muted);margin-top:2px}
.summary{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin-top:16px}
ul.plain{margin:0;padding-left:20px}
ul.plain li{margin:4px 0}
.table-wrap{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis);padding:7px 10px 7px 0;white-space:nowrap}
td{border-bottom:1px solid var(--grid);padding:7px 10px 7px 0;vertical-align:top}
td.num,th.num{text-align:right}
tr.hl td{font-weight:620}
tr.hl td:first-child::before{content:"★ ";color:var(--accent)}
details.table-view{margin-top:12px;font-size:13px}
details.table-view summary{cursor:pointer;color:var(--ink-2)}
details.table-view[open] summary{margin-bottom:8px}
.bars{display:grid;grid-template-columns:fit-content(45%) minmax(60px,1fr) auto;column-gap:10px;row-gap:2px}
.bar-row{display:grid;grid-column:1/-1;grid-template-columns:subgrid;align-items:center;padding:3px 4px;border-radius:6px;outline:none}
.bar-row:hover,.bar-row:focus-visible{background:var(--accent-wash)}
.bar-label{font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-track{position:relative;height:20px}
.bar{position:absolute;top:3px;height:14px;background:var(--deemph)}
.bar.emph{background:var(--accent)}
.bar.muted{opacity:.55}
.bar.pos{border-radius:0 4px 4px 0}
.bar.neg{border-radius:4px 0 0 4px}
.zero{position:absolute;top:0;bottom:0;width:1px;background:var(--axis)}
.whisker{position:absolute;top:9px;height:2px;background:var(--ink-2);opacity:.55}
.bar-value{font-size:13px;color:var(--ink-2);font-variant-numeric:tabular-nums;min-width:52px;text-align:right}
.chart{width:100%;min-height:240px;position:relative}
.chart svg{display:block;overflow:visible}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .baseline{stroke:var(--axis);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.chart .axis-label{fill:var(--ink-2);font-size:12px}
.chart .line{fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .area{fill:var(--accent);opacity:.10}
.chart .ref{stroke:var(--axis);stroke-width:1}
.chart .crosshair{stroke:var(--ink-2);stroke-width:1;opacity:.6}
.chart .dot{fill:var(--accent);stroke:var(--surface);stroke-width:2}
.chart .line.context{stroke:var(--ink-2);opacity:.7}
.chart .dot.context{fill:var(--ink-2)}
.chart .band{fill:var(--accent);opacity:.12}
.chart .threshold{stroke:var(--ink);stroke-width:1.5}
.chart .pt.bg{fill:var(--deemph);opacity:.45}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2);margin:0 0 6px}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.legend-key{display:inline-block;width:14px;height:2px;background:var(--accent)}
.legend-key.context{background:var(--ink-2)}
.legend-key.band{height:8px;opacity:.25}
.minis{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.chart .pt{fill:var(--accent);opacity:.55}
.chart .pt.on{opacity:1;stroke:var(--surface);stroke-width:2}
.chart .col{fill:var(--accent)}
.chart .col.on{opacity:.8}
.heatmap{width:auto;min-width:60%}
.heatmap th,.heatmap td{border:none;padding:0}
.hm-corner{font-weight:400;color:var(--muted);font-size:12px;padding:0 10px 6px 0!important}
.hm-col{font-size:12.5px;text-align:center;padding:0 2px 6px!important;max-width:110px;overflow:hidden;text-overflow:ellipsis}
.hm-row{font-size:12.5px;padding:0 10px 0 0!important;max-width:140px;overflow:hidden;text-overflow:ellipsis}
.hm-cell{background:color-mix(in oklab,var(--seq-hi) var(--p),var(--seq-lo));color:var(--hm-ink);text-align:center;
min-width:64px;height:52px;border:2px solid var(--surface)!important;border-radius:6px;outline:none}
.hm-cell.strong{color:#fff}
.hm-cell:hover,.hm-cell:focus-visible{box-shadow:inset 0 0 0 2px var(--ink-2)}
.hm-count{display:block;font-weight:620;font-size:14px}
.heatmap.compact .hm-cell{min-width:44px;height:30px}
.heatmap.compact .hm-count{font-size:12.5px;font-weight:560}
.heatmap.compact .hm-col{font-size:12px;padding:0 2px 6px!important}
.hm-share{display:block;font-size:11.5px;opacity:.85}
.issues{list-style:none;margin:0;padding:0}
.issue{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--grid)}
.issue:last-child{border-bottom:none}
.issue-icon{flex:none;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#0b0b0b;margin-top:1px}
.issue.warn .issue-icon{background:var(--warn)}
.issue.info .issue-icon{background:var(--grid);color:var(--ink-2)}
.issue-hint{color:var(--muted);font-size:13px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
pre{background:var(--code);border-radius:8px;padding:12px 14px;overflow-x:auto;margin:8px 0 0;white-space:pre-wrap;word-break:break-word}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;font-size:13.5px}
.kv dt{color:var(--ink-2)}
.kv dd{margin:0;overflow-wrap:anywhere}
.pill{display:inline-block;font-size:12px;padding:1px 8px;border-radius:999px;background:var(--grid);color:var(--ink-2);margin-left:6px;vertical-align:1px}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--tip-bg);color:var(--tip-ink);border-radius:8px;
padding:7px 10px;font-size:12.5px;box-shadow:0 4px 16px rgba(0,0,0,.18);max-width:280px}
#tip .tip-row{display:flex;gap:8px;align-items:baseline}
#tip .tip-key{display:inline-block;width:12px;height:2px;background:var(--accent);align-self:center}
#tip .tip-label{opacity:.75}
footer{margin-top:48px;color:var(--muted);font-size:12.5px}
@media (max-width:560px){h1{font-size:22px}.tile.hero .tile-value{font-size:32px}.bars{grid-template-columns:fit-content(48%) minmax(50px,1fr) auto}
header.top{flex-direction:column}}
@media print{.theme-toggle,details.table-view summary{display:none}details.table-view>*{display:block}}
"""

JS = r"""
(function(){
const data=JSON.parse(document.getElementById('report-data').textContent);
const NS='http://www.w3.org/2000/svg';
const tip=document.getElementById('tip');
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
function fmt(v){if(v==null||isNaN(v))return'—';const a=Math.abs(v);if(a===0)return'0';if(a>=1e9||a<1e-3)return Number(v).toPrecision(3);
 if(a<1)return v.toFixed(3);const d=Math.floor(Math.log10(a))+1;return v.toLocaleString(undefined,{maximumFractionDigits:Math.max(0,4-d)});}
function pct(v){return(v*100).toFixed(0)+'%';}
function date(v,span){const d=new Date(v);const day=864e5;
 if(span>730*day)return d.toLocaleDateString(undefined,{year:'numeric',month:'short'});
 if(span>60*day)return d.toLocaleDateString(undefined,{month:'short',day:'numeric'});
 if(span>2*day)return d.toLocaleDateString(undefined,{month:'short',day:'numeric'});
 return d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});}
function f(kind,v,span){return kind==='pct'?pct(v):kind==='time'?date(v,span||0):fmt(v);}
function timeTicks(lo,hi,n){const out=[];for(let i=0;i<=n;i++)out.push(lo+(hi-lo)*i/n);return out;}
function ticks(lo,hi,n){const span=(hi-lo)||1;const raw=span/Math.max(1,n);const mag=Math.pow(10,Math.floor(Math.log10(raw)));const r=raw/mag;
 const step=(r>=7.5?10:r>=3.5?5:r>=1.5?2:1)*mag;const out=[];for(let v=Math.ceil(lo/step-1e-9)*step;v<=hi+step*1e-9;v+=step)out.push(+v.toFixed(12));return out;}
function place(e){const p=14;let x=e.clientX+p,y=e.clientY+p;const w=tip.offsetWidth,h=tip.offsetHeight;
 if(x+w>innerWidth-8)x=e.clientX-w-p;if(y+h>innerHeight-8)y=e.clientY-h-p;tip.style.left=Math.max(8,x)+'px';tip.style.top=Math.max(8,y)+'px';}
function show(e,rows){tip.replaceChildren();for(const r of rows){const line=document.createElement('div');line.className='tip-row';
 if(r.key){const k=document.createElement('span');k.className='tip-key';line.appendChild(k);}
 const v=document.createElement('strong');v.textContent=r.value;line.appendChild(v);
 if(r.label){const l=document.createElement('span');l.className='tip-label';l.textContent=r.label;line.appendChild(l);}tip.appendChild(line);}
 tip.hidden=false;place(e);}
function hide(){tip.hidden=true;}
function focusPoint(target){const r=target.getBoundingClientRect();return{clientX:r.left+r.width/2,clientY:r.top};}
document.querySelectorAll('[data-tip-value]').forEach(node=>{
 const rows=()=>{const out=[{value:node.dataset.tipValue,label:node.dataset.tipLabel}];if(node.dataset.tipExtra)out.push({value:'',label:node.dataset.tipExtra});return out;};
 node.addEventListener('pointermove',e=>show(e,rows()));node.addEventListener('pointerleave',hide);
 node.addEventListener('focus',()=>show(focusPoint(node),rows()));node.addEventListener('blur',hide);});

function frame(host,spec){
 const W=Math.max(260,host.clientWidth||320);const H=spec.height||Math.round(Math.max(220,Math.min(320,W*0.6)));
 const m={t:26,r:14,b:42,l:56};const iw=W-m.l-m.r,ih=H-m.t-m.b;
 const svg=el('svg',{width:W,height:H,viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':spec.title||''},host);
 const[x0,x1]=spec.xDomain,[y0,y1]=spec.yDomain;
 const sx=v=>m.l+(v-x0)/((x1-x0)||1)*iw,sy=v=>m.t+ih-(v-y0)/((y1-y0)||1)*ih;
 const g=el('g',{},svg);
 for(const t of ticks(y0,y1,5)){if(t<y0-1e-12||t>y1+1e-12)continue;el('line',{x1:m.l,x2:m.l+iw,y1:sy(t),y2:sy(t),class:'grid'},g);
  const tx=el('text',{x:m.l-8,y:sy(t),'text-anchor':'end','dominant-baseline':'middle',class:'tick'},g);tx.textContent=f(spec.yFmt,t);}
 const xt=spec.xFmt==='time'?timeTicks(x0,x1,Math.max(2,Math.min(6,Math.floor(iw/110)))):ticks(x0,x1,Math.max(3,Math.floor(iw/80)));
 for(const t of xt){if(t<x0-1e-9*Math.abs(x0||1)||t>x1+1e-9*Math.abs(x1||1))continue;
  const tx=el('text',{x:sx(t),y:m.t+ih+18,'text-anchor':'middle',class:'tick'},g);tx.textContent=f(spec.xFmt,t,x1-x0);}
 el('line',{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,class:'baseline'},g);
 const xl=el('text',{x:m.l+iw/2,y:H-4,'text-anchor':'middle',class:'axis-label'},g);xl.textContent=spec.xLabel||'';
 const yl=el('text',{x:m.l-50,y:12,class:'axis-label'},g);yl.textContent=spec.yLabel||'';
 return{svg,sx,sy,m,iw,ih,W,H};}

function drawLine(host,spec){
 const series=spec.series;
 if(series.length>1){const lg=document.createElement('div');lg.className='legend';for(const s of series){const it=document.createElement('span');
  it.className='legend-item';const k=document.createElement('span');k.className='legend-key '+(s.role||'accent');const t=document.createElement('span');
  t.textContent=s.name||'';it.append(k,t);lg.appendChild(it);}host.appendChild(lg);}
 const F=frame(host,spec);const span=spec.xDomain[1]-spec.xDomain[0];
 if(spec.ref==='diagonal')el('line',{x1:F.sx(spec.xDomain[0]),y1:F.sy(spec.yDomain[0]),x2:F.sx(spec.xDomain[1]),y2:F.sy(spec.yDomain[1]),class:'ref'},F.svg);
 if(spec.refY!=null)el('line',{x1:F.m.l,x2:F.m.l+F.iw,y1:F.sy(spec.refY),y2:F.sy(spec.refY),class:'ref'},F.svg);
 if(spec.band){const b=spec.band;const top=b.x.map((x,i)=>`${F.sx(x)},${F.sy(b.hi[i])}`);const bot=b.x.map((x,i)=>`${F.sx(x)},${F.sy(b.lo[i])}`).reverse();
  el('path',{d:'M'+top.join('L')+'L'+bot.join('L')+'Z',class:'band'},F.svg);}
 series.forEach(s=>{const pts=s.x.map((x,i)=>[F.sx(x),F.sy(s.y[i])]);if(!pts.length)return;
  if(spec.area&&series.length===1){const base=F.sy(Math.max(spec.yDomain[0],Math.min(0,spec.yDomain[1])));
   el('path',{d:`M${pts[0][0]},${base}L`+pts.map(p=>p.join(',')).join('L')+`L${pts[pts.length-1][0]},${base}Z`,class:'area'},F.svg);}
  el('path',{d:'M'+pts.map(p=>p.join(',')).join('L'),class:'line '+(s.role||'accent')},F.svg);
  if(spec.markers)pts.forEach(p=>el('circle',{cx:p[0],cy:p[1],r:4,class:'dot '+(s.role||'accent')},F.svg));});
 const cross=el('line',{class:'crosshair',y1:F.m.t,y2:F.m.t+F.ih,visibility:'hidden'},F.svg);
 const dots=series.map(s=>el('circle',{r:4.5,class:'dot '+(s.role||'accent'),visibility:'hidden'},F.svg));
 const xs=[...new Set(series.flatMap(s=>s.x))].sort((a,b)=>a-b);
 const hit=el('rect',{x:F.m.l,y:F.m.t,width:F.iw,height:F.ih,fill:'transparent'},F.svg);
 hit.addEventListener('pointermove',e=>{const r=F.svg.getBoundingClientRect();const px=e.clientX-r.left;let X=xs[0],bd=Infinity;
  for(const x of xs){const d=Math.abs(F.sx(x)-px);if(d<bd){bd=d;X=x;}}
  cross.setAttribute('x1',F.sx(X));cross.setAttribute('x2',F.sx(X));cross.setAttribute('visibility','visible');
  const rows=[];series.forEach((s,j)=>{const i=s.x.indexOf(X);if(i<0){dots[j].setAttribute('visibility','hidden');return;}
   dots[j].setAttribute('cx',F.sx(X));dots[j].setAttribute('cy',F.sy(s.y[i]));dots[j].setAttribute('visibility','visible');
   rows.push({value:f(spec.yFmt,s.y[i]),label:s.name||spec.yLabel,key:true});});
  if(spec.band){const i=spec.band.x.indexOf(X);if(i>=0)rows.push({value:`${f(spec.yFmt,spec.band.lo[i])} to ${f(spec.yFmt,spec.band.hi[i])}`,label:spec.band.name||'range'});}
  rows.push({value:f(spec.xFmt,X,span),label:spec.xLabel});show(e,rows);});
 hit.addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'));hide();});}

function drawScatter(host,spec){
 const F=frame(host,spec);
 if(spec.identity){const a=Math.max(spec.xDomain[0],spec.yDomain[0]),b=Math.min(spec.xDomain[1],spec.yDomain[1]);
  el('line',{x1:F.sx(a),y1:F.sy(a),x2:F.sx(b),y2:F.sy(b),class:'ref'},F.svg);}
 if(spec.refY!=null)el('line',{x1:F.m.l,x2:F.m.l+F.iw,y1:F.sy(spec.refY),y2:F.sy(spec.refY),class:'ref'},F.svg);
 const g=el('g',{},F.svg);const px=spec.x.map(F.sx),py=spec.y.map(F.sy);const hl=spec.highlight;
 const nodes=new Array(spec.x.length);const order=spec.x.map((_,i)=>i);if(hl)order.sort((a,b)=>(hl[a]?1:0)-(hl[b]?1:0));
 for(const i of order)nodes[i]=el('circle',{cx:px[i],cy:py[i],r:hl?3:4,class:hl&&!hl[i]?'pt bg':'pt'},g);
 let on=null;const hit=el('rect',{x:F.m.l,y:F.m.t,width:F.iw,height:F.ih,fill:'transparent'},F.svg);
 hit.addEventListener('pointermove',e=>{const r=F.svg.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=-1,bd=24*24;for(let i=0;i<px.length;i++){const d=(px[i]-mx)**2+(py[i]-my)**2;if(d<bd){bd=d;best=i;}}
  if(on)on.classList.remove('on');if(best<0){hide();on=null;return;}on=nodes[best];on.classList.add('on');g.appendChild(on);
  const rows=[{value:f(spec.yFmt,spec.y[best]),label:spec.yLabel,key:true},{value:f(spec.xFmt,spec.x[best]),label:spec.xLabel}];
  if(spec.labels)rows.unshift({value:spec.labels[best],label:''});show(e,rows);});
 hit.addEventListener('pointerleave',()=>{if(on)on.classList.remove('on');on=null;hide();});}

function drawColumns(host,spec){
 const bins=spec.bins;const F=frame(host,spec);const base=F.sy(0);
 bins.forEach(b=>{const x=F.sx(b.x0)+1,w=Math.max(1,F.sx(b.x1)-F.sx(b.x0)-2),y=F.sy(b.count),h=base-y;
  if(h>0){const r=Math.min(4,w/2,h);el('path',{d:`M${x},${base}V${y+r}Q${x},${y} ${x+r},${y}H${x+w-r}Q${x+w},${y} ${x+w},${y+r}V${base}Z`,class:'col'},F.svg);}
  const hit=el('rect',{x:x-1,y:F.m.t,width:w+2,height:F.ih,fill:'transparent'},F.svg);
  hit.addEventListener('pointermove',e=>show(e,[{value:b.count.toLocaleString()+' rows',label:`${fmt(b.x0)} to ${fmt(b.x1)}`}]));
  hit.addEventListener('pointerleave',hide);});
 if(spec.refX!=null)el('line',{x1:F.sx(spec.refX),x2:F.sx(spec.refX),y1:F.m.t,y2:F.m.t+F.ih,class:'threshold'},F.svg);}

const drawers={line:drawLine,scatter:drawScatter,columns:drawColumns};
function renderAll(){document.querySelectorAll('[data-chart]').forEach(host=>{const spec=(data.charts||{})[host.dataset.chart];
 if(!spec||!drawers[spec.type])return;host.replaceChildren();try{drawers[spec.type](host,spec);}catch(err){host.textContent='Chart unavailable';}});}
let pending=null;const ro=new ResizeObserver(()=>{if(pending)cancelAnimationFrame(pending);pending=requestAnimationFrame(renderAll);});
document.querySelectorAll('[data-chart]').forEach(h=>ro.observe(h));renderAll();

const btn=document.querySelector('.theme-toggle');
function applyTheme(t){if(t)document.documentElement.dataset.theme=t;else delete document.documentElement.dataset.theme;
 if(btn)btn.textContent=t==='dark'?'Dark':t==='light'?'Light':'Auto';}
let theme=null;try{theme=localStorage.getItem('plainml-theme');}catch(e){}applyTheme(theme);
if(btn)btn.addEventListener('click',()=>{theme=theme===null?'light':theme==='light'?'dark':null;applyTheme(theme);
 try{theme?localStorage.setItem('plainml-theme',theme):localStorage.removeItem('plainml-theme');}catch(e){}});
})();
"""


def page(
    title: str,
    subtitle: str,
    body: str,
    charts: dict[str, Any],
    toc: list[tuple[str, str]] | None = None,
) -> str:
    from plainml.runs import _clean_nan

    payload = json.dumps(_clean_nan({"charts": charts}), default=_json_default, allow_nan=False)
    payload = payload.replace("</", "<\\/")  # user text can't close the <script> tag
    nav = ""
    if toc:
        nav = (
            '<nav class="toc">'
            + "".join(f'<a href="#{esc(a)}">{esc(t)}</a>' for a, t in toc)
            + "</nav>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header class="top"><div><h1>{esc(title)}</h1><p class="subtitle">{subtitle}</p>{nav}</div>
<button class="theme-toggle" type="button" title="Switch light / dark / auto">Auto</button></header>
{body}
<footer>Made with plainml {esc(__version__)} · {esc(datetime.now().strftime("%Y-%m-%d %H:%M"))}</footer>
</main>
<div id="tip" role="status" hidden></div>
<script id="report-data" type="application/json">{payload}</script>
<script>{JS}</script>
</body>
</html>
"""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _finite(values: list[Any]) -> list[float]:
    return [float(v) if v is not None and np.isfinite(float(v)) else 0.0 for v in values]


def _pad_domain(lo: float, hi: float, pad: float = 0.05) -> list[float]:
    if hi == lo:
        return [lo - 1, hi + 1]
    span = hi - lo
    return [lo - span * pad, hi + span * pad]


# --- sections shared by several reports -------------------------------------------------------


def profile_section(profile: dict[str, Any], schema: dict[str, Any] | None = None) -> str:
    used = set()
    if schema:
        for kind in ("numeric", "categorical", "datetime", "text"):
            used.update(schema.get(kind, []))
    rows = []
    for column in profile.get("columns", []):
        detail = column.get("note") or ", ".join(column.get("examples", []))
        stats = column.get("stats") or {}
        if column["kind"] == "numeric" and stats:
            detail = (
                f"{fmt_num(stats.get('min'))} to {fmt_num(stats.get('max'))}, median {fmt_num(stats.get('median'))}"
                + (f" · {column['note']}" if column.get("note") else "")
            )
        in_use = column["name"] in used if schema else None
        rows.append(
            [
                column["name"],
                column["kind"],
                fmt_pct(column["missing_pct"], 0),
                column["n_unique"],
                "yes" if in_use else ("no" if in_use is not None else "—"),
                detail,
            ]
        )
    columns_table = table(
        ["Column", "Type", "Missing", "Unique", "Used", "Details"], rows, numeric={2, 3}
    )
    return grid(
        card(issues_list(profile.get("issues", [])), "Checks"),
        card(columns_table, "Columns", wide=True),
    )


def _kv(items: list[tuple[str, Any]]) -> str:
    inner = "".join(
        f"<dt>{esc(k)}</dt><dd>{esc(fmt(v) if not isinstance(v, str) else v)}</dd>"
        for k, v in items
    )
    return f'<dl class="kv">{inner}</dl>'


def importance_section(
    importance: pd.DataFrame,
    effects: list[dict[str, Any]],
    sentences: list[str],
    charts: dict[str, Any],
    task: str,
    target: str,
    metric_label: str,
) -> str:
    if importance.empty:
        return ""
    top = importance.head(20)
    items = [
        {
            "label": r.feature,
            "value": max(float(r.importance), 0.0),
            "tip": f"{fmt_pct(float(r.share), 0)} of total importance",
        }
        for r in top.itertuples()
    ]
    bars = bar_list(items, emphasis=set(range(len(items))))
    table_rows = [
        [r.feature, float(r.importance), float(r.std), fmt_pct(float(r.share), 1)]
        for r in importance.itertuples()
    ]
    importance_card = card(
        bars
        + as_table_view(
            table(["Column", "Importance", "± (std)", "Share"], table_rows, numeric={1, 2, 3})
        ),
        "Importance",
        note=f"How much {esc(metric_label)} drops when a column's values are shuffled. Bigger means the model leans on it more.",
        wide=True,
    )
    effect_cards = []
    for i, effect in enumerate(effects[:6]):
        is_prob = task == "classification"
        y_label = (
            f"P({effect['class']})" if is_prob and effect.get("class") else f"predicted {target}"
        )
        if effect["kind"] == "numeric":
            chart_id = f"effect{i}"
            ys = _finite(effect["response"])
            lo, hi = min(ys), max(ys)
            domain = [0.0, 1.0] if is_prob and hi - lo > 0.3 else _pad_domain(lo, hi, 0.15)
            charts[chart_id] = {
                "type": "line",
                "series": [{"x": effect["grid"], "y": ys}],
                "xDomain": [effect["grid"][0], effect["grid"][-1]],
                "yDomain": domain,
                "xLabel": effect["feature"],
                "yLabel": y_label,
                "yFmt": "pct" if is_prob else "num",
                "height": 220,
                "title": f"Effect of {effect['feature']}",
            }
            data_rows = [
                [fmt_num(x), fmt_pct(y, 1) if is_prob else fmt_num(y)]
                for x, y in zip(effect["grid"], ys, strict=False)
            ]
            body = chart(chart_id, 220) + as_table_view(
                table([effect["feature"], y_label], data_rows, numeric={0, 1})
            )
        else:
            values = effect["values"]
            ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
            items = [{"label": k, "value": v} for k, v in ordered]
            body = bar_list(items, value_format="pct" if is_prob else "num", emphasis={0})
        effect_cards.append(
            card(
                body,
                f"{effect['feature']}",
                note="Average prediction when every row gets this value; other columns unchanged.",
            )
        )
    body = ""
    if sentences:
        body += (
            f'<div class="summary">{sentences_list(sentences)}</div><div style="height:16px"></div>'
        )
    body += grid(importance_card, *effect_cards)
    return body


# --- training report ------------------------------------------------------------------------


def render_training_report(
    info: dict[str, Any],
    leaderboard: pd.DataFrame,
    evaluation: dict[str, Any],
    importance: pd.DataFrame,
) -> str:
    charts: dict[str, Any] = {}
    task = info["task"]
    target = info["target"] if isinstance(info["target"], str) else ", ".join(info["target"])
    metric = info["metric"]
    metric_label = info.get("metric_label", metric)
    best = info["best"]
    greater = info.get("metric_greater_is_better", True)
    baseline = (info.get("baseline") or {}).get("cv_score")
    data = info["data"]
    title = f"Predicting {target}"
    subtitle = (
        f"{esc(info.get('task_label', task))} · {esc(Path(str(data['source'])).name)} · "
        f"{data['rows']:,} rows · {esc(info['created'][:16].replace('T', ' '))}"
    )

    tiles = [
        (
            f"{metric_label} (cross-validated)",
            fmt_num(best.get("cv_score")),
            f"± {fmt_num(best.get('cv_std'))}"
            + (f" · baseline {fmt_num(baseline)}" if baseline is not None else ""),
        ),
        ("Best model", best["name"], f"of {int((leaderboard['status'] == 'ok').sum())} tried"),
        (
            f"{metric_label} on unseen test rows",
            fmt_num((best.get("holdout") or {}).get(metric)),
            f"{info['split']['test_rows']:,} rows held out",
        ),
        (
            "Trained on",
            f"{best.get('refit_rows', data['rows']):,} rows",
            f"{len(info['schema'].get('order', []))} columns, {sum(len(info['schema'].get(k, [])) for k in ('numeric', 'categorical', 'datetime', 'text'))} used",
        ),
    ]
    summary_points = [f"{metric_label}: {info.get('metric_explain', '')}."]
    summary_points += evaluation.get("sentences", [])[:3]
    warnings = [i for i in info.get("profile", {}).get("issues", []) if i.get("level") == "warn"]
    body = stat_tiles(tiles) + f'<div class="summary">{sentences_list(summary_points)}</div>'
    if warnings:
        body += f'<div class="summary">{issues_list(warnings)}</div>'

    # leaderboard
    ok = leaderboard[leaderboard["status"] == "ok"].reset_index(drop=True)
    items = []
    emphasis, muted = set(), set()
    for i, row in ok.iterrows():
        items.append(
            {
                "label": row["model"],
                "value": row[metric],
                "err": row.get(f"{metric}_std"),
                "tip": row.get("note") or None,
            }
        )
        if row["key"] == best["key"]:
            emphasis.add(i)
        if row["key"] == "baseline":
            muted.add(i)
    metric_cols = [
        c
        for c in leaderboard.columns
        if c not in ("rank", "model", "key", "seconds", "status", "note")
        and not c.endswith("_std")
        and not c.startswith("train_")
    ]
    rows = []
    highlight = None
    for i, row in leaderboard.iterrows():
        if row["key"] == best["key"]:
            highlight = i
        status = "" if row["status"] == "ok" else f" ({row['status']})"
        rows.append(
            [
                str(row["model"]) + status,
                *[row[c] for c in metric_cols],
                row.get(f"train_{metric}"),
                fmt_duration(float(row["seconds"])) if pd.notna(row["seconds"]) else "—",
                row.get("note") if isinstance(row.get("note"), str) else "",
            ]
        )
    headers = [
        "Model",
        *[c.replace("_", " ") for c in metric_cols],
        f"train {metric}",
        "time",
        "notes",
    ]
    board = table(headers, rows, numeric=set(range(1, len(metric_cols) + 3)), highlight=highlight)
    direction = "higher" if greater else "lower"
    leaderboard_html = grid(
        card(
            bar_list(items, emphasis=emphasis, muted=muted),
            f"{metric_label} by model ({direction} is better)",
            note=f"Average over {info['split']['cv_folds']} cross-validation folds; whiskers show ± one standard deviation. The grey bar is the do-nothing baseline.",
            wide=True,
        ),
        card(
            board,
            "All scores",
            note="'train' is the score on the rows each model was fitted on; a much better train score means memorising.",
            wide=True,
        ),
    )
    toc = [
        ("results", "Results"),
        ("models", "Models"),
        ("test", "Test set"),
        ("drivers", "Drivers"),
        ("data", "Data"),
        ("reproduce", "Reproduce"),
    ]
    body = section("Results", body, anchor="results") + section(
        "Model comparison",
        leaderboard_html,
        intro=f"Each model was trained and scored {info['split']['cv_folds']} times on different slices of {info['split']['train_rows']:,} training rows (cross-validation).",
        anchor="models",
    )
    body += section(
        "How the best model does on unseen data",
        _holdout_html(evaluation, task, target, charts, info),
        intro=f"{info['split']['test_rows']:,} rows were set aside before any model was trained. These scores are the most honest estimate of real-world performance.",
        anchor="test",
    )
    body += section(
        "What drives the predictions",
        importance_section(
            importance,
            evaluation.get("effects", []),
            evaluation.get("sentences", [])[3:],
            charts,
            task,
            target,
            metric_label,
        ),
        anchor="drivers",
    )
    if info.get("tuning"):
        body += section("Tuning", _tuning_html(info["tuning"], metric_label), anchor="tuning")
        toc.insert(4, ("tuning", "Tuning"))
    body += section(
        "The data", profile_section(info.get("profile", {}), info.get("schema")), anchor="data"
    )
    body += section("Reproduce this run", _reproduce_html(info), anchor="reproduce")
    return page(title, subtitle, body, charts, toc)


def _calibration_card(calibration: dict[str, Any], charts: dict[str, Any]) -> str:
    from plainml.evaluation import calibration_verdict

    points = calibration["points"]
    binary = calibration.get("kind") == "binary"
    charts["calibration"] = {
        "type": "line",
        "series": [{"x": [p["predicted"] for p in points], "y": [p["observed"] for p in points]}],
        "xDomain": [0, 1],
        "yDomain": [0, 1],
        "xLabel": "predicted probability" if binary else "confidence",
        "yLabel": "actually happened" if binary else "actually right",
        "xFmt": "pct",
        "yFmt": "pct",
        "ref": "diagonal",
        "markers": True,
        "title": "Calibration",
    }
    rows = [[fmt_pct(p["predicted"], 0), fmt_pct(p["observed"], 0), p["rows"]] for p in points]
    ece = calibration["ece"]
    what = f"'{esc(calibration.get('positive', ''))}'" if binary else "the predicted class"
    middle = min(points, key=lambda p: abs(p["predicted"] - 0.7))
    example = (
        f"Rows given about {fmt_pct(middle['predicted'], 0)} turned out to be {what} "
        f"{fmt_pct(middle['observed'], 0)} of the time."
    )
    return card(
        chart("calibration")
        + as_table_view(table(["Predicted", "Happened", "Rows"], rows, numeric={0, 1, 2})),
        f"Calibration · {calibration_verdict(ece)}",
        note=f"Points on the diagonal mean the probabilities can be taken at face value. {example} "
        f"Average gap (ECE) {fmt_pct(ece, 1)}; Brier score {fmt_num(calibration['brier'])} (lower is better)."
        + (" Retrain with --calibrate to fix it." if ece > 0.08 else ""),
    )


def _holdout_html(
    evaluation: dict[str, Any], task: str, target: str, charts: dict[str, Any], info: dict[str, Any]
) -> str:
    scores = evaluation.get("scores", {})
    labels = {
        "f1": "F1",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
        "log_loss": "Log loss",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "mape": "MAPE",
        "f1_macro": "F1 (macro)",
        "hamming": "Hamming loss",
    }
    score_rows = [[labels.get(k, k), v] for k, v in scores.items()]
    rows: list[list[Any]]
    cards = [card(table(["Metric", "Score"], score_rows, numeric={1}), "Scores")]
    if evaluation.get("threshold") is not None:
        cards[0] = card(
            table(["Metric", "Score"], score_rows, numeric={1}),
            "Scores",
            note=f"Decision threshold tuned to {evaluation['threshold']:.2f} instead of the default 0.5.",
        )
    if task == "classification" and evaluation.get("confusion"):
        confusion = evaluation["confusion"]
        cards.append(
            card(
                heatmap(confusion["labels"], confusion["matrix"]),
                "Confusion matrix",
                note="Each row is a real class; the shading shows what share of it was predicted as each class. A strong diagonal is good.",
            )
        )
        per_class = evaluation.get("per_class", [])
        class_rows = [
            [p["label"], p["precision"], p["recall"], p["f1"], p["support"]] for p in per_class
        ]
        cards.append(
            card(
                table(
                    ["Class", "Precision", "Recall", "F1", "Rows"], class_rows, numeric={1, 2, 3, 4}
                ),
                "Per class",
            )
        )
        roc = evaluation.get("roc")
        if roc:
            charts["roc"] = {
                "type": "line",
                "series": [{"x": roc["fpr"], "y": roc["tpr"]}],
                "xDomain": [0, 1],
                "yDomain": [0, 1],
                "xLabel": "false positive rate",
                "yLabel": "true positive rate",
                "xFmt": "pct",
                "yFmt": "pct",
                "ref": "diagonal",
                "area": True,
                "title": "ROC curve",
            }
            rows = [
                [fmt_pct(x, 1), fmt_pct(y, 1)] for x, y in zip(roc["fpr"], roc["tpr"], strict=False)
            ][:: max(1, len(roc["fpr"]) // 25)]
            cards.append(
                card(
                    chart("roc")
                    + as_table_view(
                        table(["False positive rate", "True positive rate"], rows, numeric={0, 1})
                    ),
                    f"ROC curve · AUC {fmt_num(roc['auc'])}",
                    note=f"How many real '{esc(roc['positive'])}' rows are caught (up) for each false alarm rate (right). The diagonal is random guessing.",
                )
            )
        pr = evaluation.get("pr")
        if pr:
            charts["pr"] = {
                "type": "line",
                "series": [{"x": pr["recall"], "y": pr["precision"]}],
                "xDomain": [0, 1],
                "yDomain": [0, 1],
                "xLabel": "recall",
                "yLabel": "precision",
                "xFmt": "pct",
                "yFmt": "pct",
                "refY": pr["base_rate"],
                "title": "Precision-recall curve",
            }
            rows = [
                [fmt_pct(x, 1), fmt_pct(y, 1)]
                for x, y in zip(pr["recall"], pr["precision"], strict=False)
            ][:: max(1, len(pr["recall"]) // 25)]
            cards.append(
                card(
                    chart("pr")
                    + as_table_view(table(["Recall", "Precision"], rows, numeric={0, 1})),
                    "Precision vs recall",
                    note=f"Finding more '{esc(pr['positive'])}' rows (right) usually costs precision (down). The flat line is the base rate ({fmt_pct(pr['base_rate'], 0)}).",
                )
            )
        calibration = evaluation.get("calibration")
        if calibration and calibration.get("points"):
            cards.append(_calibration_card(calibration, charts))
    elif task == "regression" and evaluation.get("scatter"):
        scatter = evaluation["scatter"]
        xs, ys = _finite(scatter["actual"]), _finite(scatter["predicted"])
        lo, hi = min(xs + ys), max(xs + ys)
        domain = _pad_domain(lo, hi)
        charts["scatter"] = {
            "type": "scatter",
            "x": xs,
            "y": ys,
            "xDomain": domain,
            "yDomain": domain,
            "xLabel": f"actual {target}",
            "yLabel": f"predicted {target}",
            "identity": True,
            "height": 320,
            "title": "Predicted vs actual",
        }
        rows = [[a, p, p - a] for a, p in list(zip(xs, ys, strict=False))[:200]]
        cards.append(
            card(
                chart("scatter", 320)
                + as_table_view(
                    table(
                        [f"Actual {target}", f"Predicted {target}", "Error"],
                        rows,
                        numeric={0, 1, 2},
                    )
                ),
                "Predicted vs actual",
                note="Perfect predictions sit on the diagonal line.",
                wide=False,
            )
        )
        residuals = evaluation.get("residuals", {})
        bins = histogram_bins(residuals.get("values", []))
        if bins:
            charts["residuals"] = {
                "type": "columns",
                "bins": bins,
                "xDomain": [bins[0]["x0"], bins[-1]["x1"]],
                "yDomain": [0, max(b["count"] for b in bins) * 1.1],
                "xLabel": "prediction error (predicted − actual)",
                "yLabel": "rows",
                "title": "Error distribution",
            }
            bin_rows = [[f"{fmt_num(b['x0'])} to {fmt_num(b['x1'])}", b["count"]] for b in bins]
            cards.append(
                card(
                    chart("residuals")
                    + as_table_view(table(["Error range", "Rows"], bin_rows, numeric={1})),
                    "How big the errors are",
                    note=f"90% of errors fall between {fmt_num(residuals.get('p05'))} and {fmt_num(residuals.get('p95'))}. A peak centred on 0 means no systematic over- or under-prediction.",
                )
            )
    elif task == "multilabel" and evaluation.get("per_label"):
        per = evaluation["per_label"]
        items = [
            {"label": p["label"], "value": p["f1"]} for p in sorted(per, key=lambda p: -p["f1"])
        ]
        cards.append(card(bar_list(items, emphasis=set(range(len(items)))), "F1 per label"))
        rows = [[p["label"], p["precision"], p["recall"], p["f1"], p["support"]] for p in per]
        cards.append(
            card(
                table(
                    ["Label", "Precision", "Recall", "F1", "Positive rows"],
                    rows,
                    numeric={1, 2, 3, 4},
                ),
                "Per label",
            )
        )
    elif task == "multi_regression" and evaluation.get("per_target"):
        per = evaluation["per_target"]
        rows = [[p["target"], p["r2"], p["rmse"], p["mae"]] for p in per]
        cards.append(
            card(table(["Target", "R²", "RMSE", "MAE"], rows, numeric={1, 2, 3}), "Per target")
        )
    return grid(*cards)


def _tuning_html(tuning: dict[str, Any], metric_label: str) -> str:
    rows = []
    for entry in tuning.get("models", []):
        rows.append(
            [
                entry["name"],
                entry.get("before"),
                entry.get("after"),
                entry.get("trials"),
                json.dumps(entry.get("params", {}), default=str)[:160],
            ]
        )
    body = table(
        ["Model", f"{metric_label} before", f"{metric_label} after", "Trials", "Best settings"],
        rows,
        numeric={1, 2, 3},
    )
    return card(body, f"Hyperparameter search ({esc(tuning.get('method', ''))})", wide=True)


def _reproduce_html(info: dict[str, Any]) -> str:
    env = info.get("environment", {})
    packages = ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items())
    details = _kv(
        [
            ("Data", str(info["data"]["source"])),
            ("Data fingerprint", info["data"].get("fingerprint", "")),
            (
                "Rows used",
                f"{info['data']['rows']:,} of {info['data'].get('raw_rows', info['data']['rows']):,}",
            ),
            (
                "Split",
                f"{info['split']['train_rows']:,} train / {info['split']['test_rows']:,} test, {info['split']['cv_folds']}-fold CV",
            ),
            ("Seed", str(info.get("options", {}).get("seed"))),
            ("Class balancing", info.get("balance", "none")),
            ("Time taken", f"{info.get('timings', {}).get('total_seconds', 0):.1f}s"),
            ("Python", f"{env.get('python', '')} · {env.get('platform', '')}"),
            ("Packages", packages),
        ]
    )
    command = info.get("command", "")
    return grid(
        card(details, "Details"),
        card(
            f"<pre>{esc(command)}</pre><p class='note'>Or rerun with the saved settings:</p><pre>plainml train --config {esc(info.get('run_dir', '<run folder>'))}/config.yaml</pre>",
            "Command",
        ),
    )


def write_report(run_dir: str | Path) -> Path:
    """(Re)build report.html for a training run from the files in its folder."""
    run = load_run(run_dir)
    info = dict(run.info)
    info["run_dir"] = str(run.path)
    kind = info.get("kind", "train")
    if kind in ("train", "tune"):
        html_text = render_training_report(info, run.leaderboard, run.evaluation, run.importance)
    elif kind == "cluster":
        from plainml.clustering import render_cluster_report

        html_text = render_cluster_report(run)
    elif kind == "anomaly":
        from plainml.anomaly import render_anomaly_report

        html_text = render_anomaly_report(run)
    elif kind == "drift":
        from plainml.drift import render_drift_report

        html_text = render_drift_report(run)
    elif kind == "importance":
        from plainml.importance import render_importance_report

        html_text = render_importance_report(run)
    elif kind == "forecast":
        from plainml.forecasting import render_forecast_report

        html_text = render_forecast_report(run)
    else:  # pragma: no cover
        raise ValueError(f"Unknown run kind {kind}")
    path = run.path / REPORT_FILE
    path.write_text(html_text, encoding="utf-8")
    return path


def write_profile_report(path: str | Path, profile: Any) -> Path:
    data = profile.to_dict()
    target = data.get("target")
    tiles = [
        ("Rows", f"{data['n_rows']:,}", None),
        ("Columns", f"{data['n_cols']}", f"{len(profile.schema.features)} usable as inputs"),
        ("Duplicate rows", f"{data['n_duplicates']:,}", None),
        ("Warnings", str(len(profile.warnings)), None),
    ]
    body = stat_tiles(tiles, hero_first=False)
    if target:
        extra = ""
        if target.get("classes"):
            total = sum(target["classes"].values())
            items = [{"label": k, "value": v / total} for k, v in target["classes"].items()]
            extra = bar_list(items, value_format="pct", emphasis=set(range(len(items))))
        body += section(
            f"Target: {target['name']}",
            card(f"<p>{esc(target['task'])}: {esc(target['reason'])}</p>{extra}"),
        )
    body += section("Columns and checks", profile_section(data, data.get("schema")))
    html_text = page(
        f"Data profile: {Path(str(data['source'])).name}", esc(str(data["source"])), body, {}
    )
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out


def write_evaluation_report(
    path: str | Path, meta: dict[str, Any], details: dict[str, Any], metrics: dict[str, Any]
) -> Path:
    charts: dict[str, Any] = {}
    target = ", ".join(meta.get("targets", []))
    before = meta.get("holdout_scores", {})
    rows = []
    for key, metric in metrics.items():
        rows.append([metric.label, details["scores"].get(key), before.get(key)])
    body = section(
        "Scores",
        card(
            table(["Metric", "Now", "At training (test set)"], rows, numeric={1, 2}),
            note="Compare with the training-time test scores to spot drift.",
        ),
    )
    body += section("Details", _holdout_html(details, meta["task"], target, charts, {}))
    html_text = page(
        f"Evaluation: {meta.get('model', 'model')} predicting {target}",
        f"{details['n']:,} rows",
        body,
        charts,
    )
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out
