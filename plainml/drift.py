"""``plainml drift``: has the data changed since the model was trained?

No labels needed. Every model stores a compact summary of its training data (deciles
for numbers, value shares for categories, blank rates; aggregates only, never rows).
New data is compared column by column with the Population Stability Index (PSI), the
standard measure used in credit scoring and model monitoring:

    PSI < 0.1      stable
    0.1 – 0.25     moderate change: keep an eye on it
    > 0.25         major change: predictions may be unreliable; consider retraining

It also flags values the model never saw (new categories, numbers outside the training
range) and changes in how often values are blank. Columns the model relies on more are
weighted more heavily in the overall verdict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.table import Table

from plainml import __version__
from plainml.console import console, esc, fmt_num, fmt_pct, heading, info, note, quiet, warn
from plainml.errors import PlainMLError, find_column
from plainml.io import describe_source, load_data, source_stem
from plainml.preprocessing import category_strings
from plainml.runs import (
    EVALUATION_FILE,
    REPORT_FILE,
    RUN_FILE,
    create_run_dir,
    environment,
    write_json,
)
from plainml.schema import Schema, coerce_numeric, infer_schema, parse_dates

MODERATE, MAJOR = 0.1, 0.25
TOP_CATEGORIES = 50
EPSILON = 1e-4
DRIFT_FILE = "drift.csv"
MIN_ROWS_FOR_WARNING = 30


def describe_distribution(X: pd.DataFrame, schema: Schema) -> dict[str, Any]:
    """Summarise each input column's distribution (aggregates only) for later comparison."""
    summary: dict[str, Any] = {}
    for column in schema.features:
        values = X[column]
        missing = float(values.isna().mean())
        if column in schema.numeric:
            numbers = coerce_numeric(values).dropna()
            if numbers.empty:
                continue
            edges = np.unique(np.quantile(numbers, np.linspace(0.1, 0.9, 9)))
            counts = np.bincount(
                np.searchsorted(edges, numbers, side="right"), minlength=len(edges) + 1
            )
            summary[column] = {
                "kind": "numeric",
                "edges": edges.tolist(),
                "shares": (counts / counts.sum()).tolist(),
                "min": float(numbers.min()),
                "max": float(numbers.max()),
                "mean": float(numbers.mean()),
                "missing": missing,
            }
        elif column in schema.categorical:
            shares = category_strings(values).dropna().value_counts(normalize=True)
            top = shares.head(TOP_CATEGORIES)
            summary[column] = {
                "kind": "categorical",
                "shares": {str(k): float(v) for k, v in top.items()},
                "other": float(max(0.0, 1 - top.sum())),
                "missing": missing,
            }
        elif column in schema.datetime:
            dates = parse_dates(values).dropna()
            summary[column] = {
                "kind": "datetime",
                "min": str(dates.min()) if len(dates) else None,
                "max": str(dates.max()) if len(dates) else None,
                "missing": missing,
            }
        else:
            summary[column] = {
                "kind": "text",
                "missing": float((values.fillna("").astype(str).str.strip() == "").mean()),
            }
    return summary


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    expected, actual = expected / expected.sum(), actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compare_column(reference: dict[str, Any], values: pd.Series) -> dict[str, Any]:
    kind = reference["kind"]
    row: dict[str, Any] = {"kind": kind, "missing_before": reference.get("missing", 0.0)}
    if kind == "text":
        row["missing_now"] = float((values.fillna("").astype(str).str.strip() == "").mean())
        row["psi"] = psi(
            [1 - row["missing_before"], row["missing_before"]],
            [1 - row["missing_now"], row["missing_now"]],
        )
        return row
    row["missing_now"] = float(values.isna().mean())
    if kind == "numeric":
        numbers = coerce_numeric(values).dropna()
        edges = np.asarray(reference["edges"])
        if numbers.empty:
            row.update(psi=np.nan, outside=np.nan, shift=None)
            return row
        counts = np.bincount(
            np.searchsorted(edges, numbers, side="right"), minlength=len(edges) + 1
        )
        row["psi"] = psi(reference["shares"], counts / counts.sum())
        row["outside"] = float(((numbers < reference["min"]) | (numbers > reference["max"])).mean())
        spread = max(reference["max"] - reference["min"], 1e-12)
        row["shift"] = float((numbers.mean() - reference["mean"]) / spread)
        row["mean_before"], row["mean_now"] = reference["mean"], float(numbers.mean())
    elif kind == "categorical":
        now = category_strings(values).dropna().value_counts(normalize=True)
        known = reference["shares"]
        categories = list(known)
        expected = [known[c] for c in categories] + [reference.get("other", 0.0)]
        actual = [float(now.get(c, 0.0)) for c in categories] + [
            float(now[~now.index.isin(categories)].sum())
        ]
        row["psi"] = psi(expected, actual)
        unseen = (
            now[~now.index.isin(categories)]
            if reference.get("other", 0.0) < 0.01
            else pd.Series(dtype=float)
        )
        row["outside"] = float(unseen.sum())
        row["new_values"] = [str(v) for v in unseen.head(5).index]
    elif kind == "datetime":
        dates = parse_dates(values).dropna()
        row["psi"] = np.nan
        if len(dates) and reference.get("max"):
            later = dates > pd.Timestamp(reference["max"])
            earlier = dates < pd.Timestamp(reference["min"])
            row["outside"] = float((later | earlier).mean())
            row["latest_now"] = str(dates.max())
    missing_psi = psi(
        [1 - row["missing_before"], row["missing_before"]],
        [1 - row["missing_now"], row["missing_now"]],
    )
    row["psi"] = float(np.nanmax([row.get("psi", np.nan), missing_psi]))
    return row


def status_of(row: dict[str, Any]) -> str:
    value = _value(row, "psi")
    outside = _value(row, "outside") or 0.0
    if (value is not None and value > MAJOR) or outside > 0.2:
        return "major"
    if (value is not None and value > MODERATE) or outside > 0.05:
        return "moderate"
    return "stable"


def _value(row: dict[str, Any], key: str) -> Any:
    """A field from a table row, treating NaN (pandas' 'not set') as None."""
    value = row.get(key)
    return None if isinstance(value, float) and np.isnan(value) else value


def _explain(column: str, row: dict[str, Any]) -> str:
    parts = []
    before, now = _value(row, "mean_before"), _value(row, "mean_now")
    if (
        row.get("kind") == "numeric"
        and before is not None
        and now is not None
        and row["status"] != "stable"
    ):
        parts.append(f"average {fmt_num(before)} → {fmt_num(now)}")
    outside = _value(row, "outside")
    if outside:
        label = {
            "categorical": "values never seen in training",
            "datetime": "dates outside the training period",
        }.get(row["kind"], "values outside the training range")
        examples = _value(row, "new_values")
        example = (
            f" (e.g. {', '.join(examples[:3])})" if isinstance(examples, list) and examples else ""
        )
        parts.append(f"{fmt_pct(outside, 0)} {label}{example}")
    if abs((_value(row, "missing_now") or 0) - (_value(row, "missing_before") or 0)) > 0.05:
        parts.append(
            f"blanks {fmt_pct(row['missing_before'], 0)} → {fmt_pct(row['missing_now'], 0)}"
        )
    if not parts and row["status"] != "stable":
        parts.append("the mix of values has shifted")
    return f"{column}: " + "; ".join(parts) if parts else ""


@dataclass
class DriftResult:
    table: pd.DataFrame
    verdict: str
    sentences: list[str]
    run_dir: Path | None = None

    @property
    def report_path(self) -> Path | None:
        return self.run_dir / REPORT_FILE if self.run_dir else None

    @property
    def drifted(self) -> list[str]:
        return self.table.loc[self.table["status"] != "stable", "column"].tolist()

    def __repr__(self) -> str:
        return (
            f"DriftResult({self.verdict}: {len(self.drifted)} of {len(self.table)} columns changed)"
        )


def compare(
    reference: dict[str, Any], frame: pd.DataFrame, importance: dict[str, float] | None = None
) -> tuple[pd.DataFrame, str, list[str]]:
    """Compare new data against a stored summary. Returns (per-column table, verdict, sentences)."""
    rows = []
    for column, summary in reference.items():
        if column not in frame.columns:
            rows.append(
                {
                    "column": column,
                    "kind": summary["kind"],
                    "psi": np.nan,
                    "status": "missing",
                    "missing_now": 1.0,
                }
            )
            continue
        row = compare_column(summary, frame[column])
        row["column"] = column
        row["status"] = status_of(row)
        rows.append(row)
    table = pd.DataFrame(rows)
    weights = importance or {}
    table["importance"] = table["column"].map(lambda c: float(weights.get(c, 0.0)))
    order = {"missing": 0, "major": 1, "moderate": 2, "stable": 3}
    table = table.sort_values(
        by=["status", "importance", "psi"],
        key=lambda s: s.map(order) if s.name == "status" else -s.fillna(0),
    ).reset_index(drop=True)
    changed = table[table["status"].isin(["major", "moderate", "missing"])]
    important = changed[changed["importance"] >= 0.05] if weights else changed
    if (table["status"] == "missing").any() or (important["status"] == "major").any():
        verdict = "major"
    elif len(important) or (changed["status"] == "major").any():
        verdict = "moderate"
    else:
        verdict = "stable"
    sentences = []
    if verdict == "stable":
        sentences.append(
            "The new data looks like the training data. Predictions should be as reliable as at training time."
        )
    elif verdict == "moderate":
        sentences.append(
            "Some columns have shifted. Predictions are probably fine, but keep an eye on these:"
        )
    else:
        sentences.append(
            "The new data differs noticeably from the training data, in columns the model relies on. Predictions may be unreliable; consider retraining on recent data."
        )
    for _, row in changed.head(6).iterrows():
        if row["status"] == "missing":
            sentences.append(f"{row['column']}: missing from the new data (treated as blank).")
        else:
            text = _explain(row["column"], row.to_dict())
            if text:
                sentences.append(text + ".")
    return table, verdict, sentences


def quick_check(meta: dict[str, Any], frame: pd.DataFrame) -> str | None:
    """A one-line warning for `plainml predict` when inputs have clearly drifted (else None)."""
    reference = meta.get("drift_profile")
    if not reference or len(frame) < MIN_ROWS_FOR_WARNING:
        return None
    table, verdict, _ = compare(reference, frame, meta.get("importance"))
    if verdict != "major":
        return None
    names = ", ".join(table.loc[table["status"].isin(["major", "missing"]), "column"].head(4))
    return f"This data looks different from the training data (biggest changes: {names}). Predictions may be less reliable; see: plainml drift MODEL DATA"


def check_drift(
    reference: Any,
    data: Any,
    *,
    target: str | None = None,
    out_dir: str | Path = "runs",
    name: str | None = None,
    report: bool = True,
    save: bool = True,
    verbose: bool = True,
    **load_options: Any,
) -> DriftResult:
    """Compare ``data`` against a model's training data, or against an older data file.

    ``reference`` is a model ('latest', a run folder, a .joblib file) or a data file/DataFrame.
    """
    started = time.time()
    with quiet(not verbose):
        new = load_data(data, **load_options)
        summary, importance, source_label = _reference(reference, target, out_dir)
        table, verdict, sentences = compare(summary, new, importance)
    result = DriftResult(table=table, verdict=verdict, sentences=sentences)
    if save:
        run_dir = create_run_dir(out_dir, name or f"{source_stem(data)}-drift")
        table.to_csv(run_dir / DRIFT_FILE, index=False)
        write_json(run_dir / EVALUATION_FILE, {"sentences": sentences, "verdict": verdict})
        write_json(
            run_dir / RUN_FILE,
            {
                "kind": "drift",
                "plainml_version": __version__,
                "environment": environment(),
                "created": datetime.now().isoformat(timespec="seconds"),
                "task": "drift",
                "task_label": "data drift",
                "target": None,
                "reference": source_label,
                "data": {
                    "source": describe_source(data),
                    "rows": len(new),
                    "columns": new.shape[1],
                },
                "best": {
                    "key": verdict,
                    "name": verdict,
                    "score": float(table["psi"].max(skipna=True))
                    if table["psi"].notna().any()
                    else None,
                },
                "timings": {"total_seconds": round(time.time() - started, 2)},
                "files": {"drift": DRIFT_FILE, "report": REPORT_FILE if report else None},
            },
        )
        result.run_dir = run_dir
        if report:
            try:
                from plainml.report import write_report

                write_report(run_dir)
            except Exception as exc:
                warn(f"Couldn't write the HTML report: {esc(str(exc)[:120])}")
    if verbose:
        _render(result, source_label)
    return result


def _reference(
    reference: Any, target: str | None, out_dir: str | Path
) -> tuple[dict, dict | None, str]:
    """The stored summary to compare against: from a model, or computed from a data file."""
    as_path = Path(str(reference)) if not isinstance(reference, pd.DataFrame) else None
    looks_like_data = isinstance(reference, pd.DataFrame) or (
        as_path is not None
        and as_path.is_file()
        and as_path.suffix.lower() not in (".joblib", ".pkl", ".pickle")
    )
    if looks_like_data:
        old = load_data(reference)
        exclude = [find_column(target, old.columns, "Target column")] if target else []
        schema, _ = infer_schema(old, exclude=exclude)
        return describe_distribution(old, schema), None, describe_source(reference)
    from plainml.predicting import load_model

    _, meta, path = load_model(reference, out_dir=out_dir, with_path=True)
    summary = meta.get("drift_profile")
    if not summary:
        raise PlainMLError(
            "This model doesn't store a summary of its training data.",
            hint="Compare two data files instead: plainml drift OLD.csv NEW.csv",
        )
    return (
        summary,
        meta.get("importance"),
        f"model {meta.get('model', '')} ({Path(str(path)).parent.name if path else ''})",
    )


def _render(result: DriftResult, source_label: str) -> None:
    heading(f"Drift against {source_label}")
    grid = Table(box=None, header_style="muted", pad_edge=False)
    for column, justify in (
        ("Column", "left"),
        ("Change", "left"),
        ("PSI", "right"),
        ("Unseen / out of range", "right"),
        ("Blanks", "right"),
        ("Importance", "right"),
    ):
        grid.add_column(column, justify=justify)
    styles = {
        "major": "[error]major[/]",
        "moderate": "[warn]moderate[/]",
        "stable": "[good]stable[/]",
        "missing": "[error]missing[/]",
    }
    for _, row in result.table.head(30).iterrows():
        blanks = f"{fmt_pct(row.get('missing_before', 0) or 0, 0)} → {fmt_pct(row.get('missing_now', 0) or 0, 0)}"
        grid.add_row(
            esc(row["column"]),
            styles.get(row["status"], row["status"]),
            fmt_num(row.get("psi")),
            fmt_pct(row["outside"], 1) if pd.notna(row.get("outside")) else "—",
            blanks,
            fmt_pct(row["importance"], 0) if row["importance"] else "—",
        )
    console.print(grid)
    note(
        "PSI (population stability index): under 0.1 is stable, 0.1–0.25 a moderate change, over 0.25 a major change."
    )
    heading("Verdict")
    style = {"stable": "good", "moderate": "warn", "major": "error"}[result.verdict]
    console.print(f"[{style}]{result.verdict.upper()}[/]  {esc(result.sentences[0])}")
    for sentence in result.sentences[1:]:
        console.print(f"• {esc(sentence)}")
    if result.run_dir:
        info(f"Saved {esc(result.run_dir)} (drift.csv, report.html)")


def render_drift_report(run: Any) -> str:
    from plainml.report import (
        bar_list,
        card,
        grid,
        page,
        section,
        sentences_list,
        stat_tiles,
        table,
    )

    info_json, evaluation = run.info, run.evaluation
    frame = pd.read_csv(run.path / DRIFT_FILE)
    verdict = evaluation.get("verdict", "stable")
    changed = frame[frame["status"] != "stable"]
    tiles = [
        (
            "Verdict",
            verdict.capitalize(),
            {
                "stable": "looks like the training data",
                "moderate": "keep an eye on it",
                "major": "consider retraining",
            }[verdict],
        ),
        ("Columns changed", f"{len(changed)} of {len(frame)}", None),
        ("Rows compared", f"{info_json['data']['rows']:,}", None),
    ]
    body = section(
        "Results",
        stat_tiles(tiles)
        + f'<div class="summary">{sentences_list(evaluation.get("sentences", []))}</div>',
        anchor="results",
    )
    items = [
        {
            "label": r["column"],
            "value": float(r["psi"]) if pd.notna(r["psi"]) else 0.0,
            "tip": r["status"],
        }
        for _, r in frame.head(25).iterrows()
    ]
    emphasis = {
        i for i, r in enumerate(frame.head(25).itertuples()) if r.status in ("major", "missing")
    }
    rows = [
        [
            r["column"],
            r["status"],
            r["psi"],
            r.get("outside"),
            r.get("missing_before"),
            r.get("missing_now"),
            r.get("importance"),
        ]
        for _, r in frame.iterrows()
    ]
    body += section(
        "By column",
        grid(
            card(
                bar_list(items, emphasis=emphasis, muted=set(range(len(items))) - emphasis),
                "How much each column changed (PSI)",
                note="Blue: a major change. Under 0.1 is stable; over 0.25 is a major change.",
                wide=True,
            ),
            card(
                table(
                    [
                        "Column",
                        "Change",
                        "PSI",
                        "Unseen / outside",
                        "Blanks before",
                        "Blanks now",
                        "Importance",
                    ],
                    rows,
                    numeric={2, 3, 4, 5, 6},
                ),
                "Details",
                wide=True,
            ),
        ),
        anchor="columns",
    )
    title = f"Data drift: {Path(str(info_json['data']['source'])).name}"
    subtitle = f"compared with {info_json.get('reference', '')} · {info_json['created'][:16].replace('T', ' ')}"
    return page(title, subtitle, body, {}, [("results", "Results"), ("columns", "By column")])
