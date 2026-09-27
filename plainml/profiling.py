"""Looking at a dataset before training: column types, gaps, and likely problems."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.table import Table

from plainml.console import console, esc, fmt_num, fmt_pct, heading
from plainml.errors import find_column
from plainml.io import describe_source, load_data
from plainml.schema import HIGH_CARDINALITY, ColumnInfo, Kind, Schema, infer_schema
from plainml.tasks import CLASSIFICATION, REGRESSION, TargetInfo, detect_task

TINY_DATASET = 50
HIGH_MISSING = 0.4
LEAK_CORRELATION = 0.98


@dataclass
class Issue:
    level: str  # "warn" or "info"
    message: str
    hint: str | None = None
    code: str = ""


@dataclass
class Profile:
    source: str
    n_rows: int
    n_cols: int
    n_duplicates: int
    memory_mb: float
    columns: list[ColumnInfo]
    schema: Schema
    target: dict[str, Any] | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warn"]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = self.schema.to_dict()
        return data

    def columns_frame(self) -> pd.DataFrame:
        rows = []
        for c in self.columns:
            rows.append(
                {
                    "column": c.name,
                    "type": c.kind,
                    "missing": c.missing_pct,
                    "unique": c.n_unique,
                    "used": c.name in self.schema.features,
                    "note": c.note,
                    "examples": ", ".join(c.examples),
                }
            )
        return pd.DataFrame(rows)

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        return self.columns_frame().to_html(index=False)


def _target_summary(y: pd.Series | pd.DataFrame, info: TargetInfo) -> dict[str, Any]:
    summary: dict[str, Any] = {"name": info.display, "task": info.task, "reason": info.reason}
    if isinstance(y, pd.DataFrame):
        summary["positive_rate"] = (
            {c: float(y[c].mean()) for c in y} if info.task == "multilabel" else None
        )
        return summary
    if info.task == CLASSIFICATION:
        counts = y.value_counts()
        summary["classes"] = {str(k): int(v) for k, v in counts.items()}
        summary["minority_share"] = float(counts.min() / counts.sum())
        summary["imbalance_ratio"] = float(counts.max() / counts.min())
    else:
        values = y.astype(float)
        summary.update(
            mean=float(values.mean()),
            std=float(values.std()),
            min=float(values.min()),
            median=float(values.median()),
            max=float(values.max()),
            skew=float(values.skew()) if len(values) > 2 else 0.0,
        )
    return summary


def redact_columns(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Column summaries without any raw values (for --private runs)."""
    redacted = []
    for column in columns:
        column = {**column, "examples": []}
        column["stats"] = {k: v for k, v in (column.get("stats") or {}).items() if k != "top"}
        redacted.append(column)
    return redacted


def shown_source(source: str, private: bool) -> str:
    """A --private run records only the file name, not where it lives."""
    from pathlib import Path

    return Path(str(source)).name if private else source


def is_imbalanced(y: pd.Series) -> bool:
    counts = y.value_counts()
    if len(counts) < 2:
        return False
    return bool(counts.min() / counts.sum() < 0.2 and counts.max() / counts.min() > 3) or bool(
        counts.max() / counts.min() > 10
    )


def _leak_checks(X: pd.DataFrame, y: pd.Series, schema: Schema, task: str) -> list[Issue]:
    issues: list[Issue] = []
    n = len(y)
    if n < 10:
        return issues
    target_text = y.astype(str)
    for column in schema.features:
        values = X[column]
        if values.astype(str).equals(target_text):
            issues.append(
                Issue(
                    "warn",
                    f"'{column}' is identical to the target.",
                    hint=f"Drop it with --drop {column}.",
                    code="leak",
                )
            )
            continue
        if column in schema.numeric:
            numeric = pd.to_numeric(values, errors="coerce")
            if task == REGRESSION or y.nunique() == 2:
                encoded = (
                    y.astype(float)
                    if task == REGRESSION
                    else (y == sorted(y.unique(), key=str)[-1]).astype(float)
                )
                mask = numeric.notna()
                if mask.sum() > 10 and numeric[mask].std() > 0 and encoded[mask].std() > 0:
                    corr = float(np.corrcoef(numeric[mask], encoded[mask])[0, 1])
                    if abs(corr) >= LEAK_CORRELATION:
                        issues.append(
                            Issue(
                                "warn",
                                f"'{column}' tracks the target almost perfectly (correlation {corr:.3f}).",
                                hint=f"If it's calculated from the target or only known afterwards, drop it with --drop {column}.",
                                code="leak",
                            )
                        )
        if task == CLASSIFICATION and column in schema.categorical:
            n_values = values.nunique()
            if 2 <= n_values <= max(2, n // 2):
                groups = (
                    pd.DataFrame({"x": values.astype(str), "y": target_text})
                    .groupby("x")["y"]
                    .nunique()
                )
                if groups.max() == 1 and n_values < n * 0.5 and n_values >= y.nunique():
                    issues.append(
                        Issue(
                            "warn",
                            f"'{column}' determines the target exactly (each of its values maps to one class).",
                            hint=f"That's often a leak. If so, drop it with --drop {column}.",
                            code="leak",
                        )
                    )
    return issues


def build_profile(
    df: pd.DataFrame,
    schema: Schema,
    infos: list[ColumnInfo],
    source: str,
    target: TargetInfo | None = None,
    y: pd.Series | pd.DataFrame | None = None,
    X: pd.DataFrame | None = None,
) -> Profile:
    n_rows = len(df)
    try:
        n_duplicates = int(df.duplicated().sum())
    except TypeError:  # unhashable cells
        n_duplicates = 0
    prof = Profile(
        source=source,
        n_rows=n_rows,
        n_cols=df.shape[1],
        n_duplicates=n_duplicates,
        memory_mb=float(df.memory_usage(deep=True).sum() / 1e6),
        columns=infos,
        schema=schema,
    )
    issues = prof.issues
    if n_rows < TINY_DATASET:
        issues.append(
            Issue(
                "warn",
                f"Only {n_rows} rows. Scores will be noisy; more data helps more than anything else.",
                code="tiny",
            )
        )
    if n_duplicates:
        issues.append(
            Issue(
                "info",
                f"{n_duplicates} duplicate rows ({fmt_pct(n_duplicates / n_rows)}).",
                hint="plainml clean can remove them.",
                code="duplicates",
            )
        )
    if schema.dropped:
        listed = "; ".join(f"{c} ({why})" for c, why in list(schema.dropped.items())[:6])
        more = f" and {len(schema.dropped) - 6} more" if len(schema.dropped) > 6 else ""
        issues.append(
            Issue(
                "info",
                f"Not used as inputs: {listed}{more}.",
                hint="Force a column in with --keep NAME.",
                code="dropped",
            )
        )
    for info in infos:
        if info.name in schema.dropped:
            continue
        if info.missing_pct >= HIGH_MISSING:
            issues.append(
                Issue(
                    "warn",
                    f"'{info.name}' is {fmt_pct(info.missing_pct, 0)} blank; blanks are filled in automatically.",
                    code="missing",
                )
            )
        if info.stored_as_text:
            issues.append(
                Issue(
                    "info",
                    f"'{info.name}' holds numbers stored as text; they're converted.",
                    code="numeric_text",
                )
            )
        if info.kind == Kind.CATEGORICAL and info.n_unique > HIGH_CARDINALITY:
            issues.append(
                Issue(
                    "info",
                    f"'{info.name}' has {info.n_unique} different values; the 24 most common are used and the rest grouped.",
                    code="cardinality",
                )
            )
    if target is not None and y is not None:
        prof.target = _target_summary(y, target)
        if target.dropped_rows:
            issues.append(
                Issue(
                    "warn",
                    f"Dropped {target.dropped_rows} rows with no value for the target.",
                    code="target_missing",
                )
            )
        for note in target.notes:
            issues.append(Issue("warn", note + ".", code="target"))
        if isinstance(y, pd.Series) and target.task == CLASSIFICATION:
            counts = y.value_counts()
            if is_imbalanced(y):
                shares = ", ".join(f"{k}: {fmt_pct(v / counts.sum())}" for k, v in counts.items())
                issues.append(
                    Issue(
                        "warn",
                        f"The classes are imbalanced ({shares}).",
                        hint="plainml weights the rare classes automatically; see --balance for options.",
                        code="imbalance",
                    )
                )
            if len(counts) > 20:
                issues.append(
                    Issue(
                        "warn",
                        f"{len(counts)} classes is a lot; each needs plenty of examples.",
                        code="many_classes",
                    )
                )
        if (
            isinstance(y, pd.Series)
            and target.task == REGRESSION
            and abs(prof.target.get("skew", 0)) > 2
        ):
            issues.append(
                Issue(
                    "info",
                    "The target is heavily skewed (a few very large values).",
                    hint="If it's always positive, try --log-target.",
                    code="skew",
                )
            )
        if X is not None and isinstance(y, pd.Series):
            issues.extend(_leak_checks(X, y, schema, target.task))
    return prof


def profile(
    data: str | Path | pd.DataFrame,
    target: str | None = None,
    *,
    drop: list[str] | None = None,
    keep: list[str] | None = None,
    verbose: bool = True,
    **load_options: Any,
) -> Profile:
    """Profile a dataset (and optionally a target column) without training anything."""
    df = load_data(data, **load_options)
    source = describe_source(data)
    drop = [find_column(c, df.columns) for c in drop or []]
    keep = [find_column(c, df.columns) for c in keep or []]
    if target:
        from plainml.tasks import prepare_target

        targets = [
            find_column(t.strip(), df.columns, "Target column") for t in str(target).split(",")
        ]
        kept, y, info = prepare_target(df, targets)
        X = kept.drop(columns=targets)
        schema, infos = infer_schema(X, drop=drop, keep=keep)
        prof = build_profile(df, schema, infos, source, info, y, X)
    else:
        schema, infos = infer_schema(df, drop=drop, keep=keep)
        prof = build_profile(df, schema, infos, source)
    if verbose:
        render_profile(prof)
    return prof


_KIND_STYLE = {
    Kind.NUMERIC: "cyan",
    Kind.CATEGORICAL: "magenta",
    Kind.DATETIME: "blue",
    Kind.TEXT: "green",
    Kind.ID: "dim",
    Kind.CONSTANT: "dim",
    Kind.EMPTY: "dim",
}


def _details(info: ColumnInfo) -> str:
    stats = info.stats
    if info.kind == Kind.NUMERIC and stats:
        return f"{fmt_num(stats['min'])} … {fmt_num(stats['max'])}  (median {fmt_num(stats['median'])})"
    if info.kind == Kind.CATEGORICAL and stats.get("top"):
        return ", ".join(list(stats["top"])[:4]) + (" …" if info.n_unique > 4 else "")
    if info.kind == Kind.DATETIME and stats:
        return f"{stats['min'][:10]} → {stats['max'][:10]}"
    return info.note or ", ".join(info.examples)


def render_issues(issues: list[Issue]) -> None:
    for issue in issues:
        style = "warn" if issue.level == "warn" else "info"
        marker = "!" if issue.level == "warn" else "•"
        console.print(f"[{style}]{marker}[/] {esc(issue.message)}")
        if issue.hint:
            console.print(f"  [muted]{esc(issue.hint)}[/]")


def _short_source(source: str) -> str:
    """Show paths relative to the current folder when that's shorter."""
    import os

    try:
        relative = os.path.relpath(source)
    except ValueError:  # different drive on Windows
        return source
    return relative if len(relative) < len(source) else source


def render_profile(prof: Profile, *, columns: bool = True) -> None:
    heading("Data")
    console.print(
        f"[head]{prof.n_rows:,}[/] rows × [head]{prof.n_cols}[/] columns  "
        f"[muted]{esc(_short_source(prof.source))} · {prof.memory_mb:.1f} MB[/]"
    )
    if columns:
        table = Table(show_edge=False, pad_edge=False, box=None, header_style="muted")
        table.add_column("Column", overflow="fold", max_width=28)
        table.add_column("Type")
        table.add_column("Missing", justify="right")
        table.add_column("Unique", justify="right")
        table.add_column("Values", overflow="ellipsis", max_width=48, no_wrap=True)
        for info in prof.columns[:60]:
            used = info.name in prof.schema.features
            kind = (
                info.kind
                if used or info.kind not in (Kind.NUMERIC, Kind.CATEGORICAL)
                else f"{info.kind} (unused)"
            )
            name = esc(info.name) if used else f"[dim]{esc(info.name)}[/]"
            missing = fmt_pct(info.missing_pct, 0) if info.n_missing else "[dim]0%[/]"
            table.add_row(
                name,
                f"[{_KIND_STYLE.get(info.kind, 'white')}]{kind}[/]",
                missing,
                f"{info.n_unique:,}",
                esc(_details(info)),
            )
        console.print(table)
        if len(prof.columns) > 60:
            console.print(f"[muted]… and {len(prof.columns) - 60} more columns[/]")
    if prof.target:
        target = prof.target
        console.print()
        from plainml.tasks import TASK_LABELS

        task = TASK_LABELS.get(target["task"], target["task"])
        line = f"Target [head]{esc(target['name'])}[/] → [bold]{task}[/] [muted]({esc(target['reason'])})[/]"
        console.print(line)
        if target.get("classes"):
            total = sum(target["classes"].values())
            shares = ", ".join(
                f"{k} {fmt_pct(v / total, 0)}" for k, v in list(target["classes"].items())[:8]
            )
            console.print(f"  [muted]{esc(shares)}[/]")
        elif "mean" in target:
            console.print(
                f"  [muted]mean {fmt_num(target['mean'])}, range {fmt_num(target['min'])} … {fmt_num(target['max'])}[/]"
            )
    if prof.issues:
        console.print()
        render_issues(prof.issues)


def detect_target_task(df: pd.DataFrame, column: str) -> tuple[str, str]:
    """Convenience wrapper used by the wizard and the UI."""
    return detect_task(df[column])
