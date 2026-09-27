"""``plainml anomaly``: score every row by how unusual it is, and say why.

Several detectors look for different kinds of odd: Isolation Forest (easy to isolate),
Local Outlier Factor (far from its neighbours), One-Class SVM (outside the usual region)
and a robust distance from the median. Their scores are turned into percentiles and
averaged. With a label column of known anomalies, each detector is scored and the best
one is kept.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from plainml import __version__
from plainml.console import (
    console,
    esc,
    fmt_num,
    fmt_pct,
    fmt_value,
    heading,
    info,
    note,
    quiet,
    warn,
)
from plainml.errors import PlainMLError, did_you_mean, find_column
from plainml.io import describe_source, fingerprint, load_data, save_table, source_stem
from plainml.preprocessing import build_preprocessor
from plainml.profiling import build_profile, redact_columns, shown_source
from plainml.runs import (
    EVALUATION_FILE,
    LEADERBOARD_FILE,
    MODEL_FILE,
    REPORT_FILE,
    RUN_FILE,
    create_run_dir,
    environment,
    write_json,
)
from plainml.schema import Schema, coerce_numeric, infer_schema
from plainml.tasks import ANOMALY, _clean_labels

DETECTORS = {
    "iforest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "ocsvm": "One-Class SVM",
    "robust": "Robust distance",
}
OCSVM_MAX_ROWS = 10_000
DEFAULT_CONTAMINATION = 0.05
SCORED_FILE = "scored.csv"


class RobustDistance(BaseEstimator):
    """How far a row sits from the median, in robust spread units, on its most extreme column."""

    def fit(self, X: np.ndarray, y: Any = None) -> RobustDistance:
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0) * 1.4826
        std = X.std(axis=0)
        self.scale_ = np.where(mad > 1e-9, mad, np.where(std > 1e-9, std, 1.0))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((np.asarray(X, dtype=float) - self.median_) / self.scale_)
        return -(0.5 * z.max(axis=1) + 0.5 * np.sqrt((z**2).mean(axis=1)))


class AnomalyModel(BaseEstimator):
    """Scores new rows on the same 0-1 scale as training (1 = more unusual than every training row)."""

    def __init__(
        self,
        preprocess: Any = None,
        detectors: dict[str, Any] | None = None,
        reference: dict[str, np.ndarray] | None = None,
        threshold: float = 0.95,
    ):
        self.preprocess = preprocess
        self.detectors = detectors
        self.reference = reference
        self.threshold = threshold

    def raw_scores(self, X: Any) -> dict[str, np.ndarray]:
        encoded = np.asarray(self.preprocess.transform(X), dtype=float)
        return {
            name: -np.asarray(detector.score_samples(encoded))
            for name, detector in (self.detectors or {}).items()
        }

    def score(self, X: Any) -> np.ndarray:
        raw = self.raw_scores(X)
        assert self.reference is not None
        percentiles = [
            np.searchsorted(self.reference[name], values, side="right") / len(self.reference[name])
            for name, values in raw.items()
        ]
        return np.mean(percentiles, axis=0)

    def flag(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores) >= self.threshold

    def predict(self, X: Any) -> np.ndarray:
        return self.flag(self.score(X)).astype(int)


@dataclass
class AnomalyResult:
    run_dir: Path
    scores: pd.Series
    flags: pd.Series
    top: pd.DataFrame
    leaderboard: pd.DataFrame
    model: AnomalyModel

    @property
    def report_path(self) -> Path:
        return self.run_dir / REPORT_FILE

    def __repr__(self) -> str:
        return f"AnomalyResult({int(self.flags.sum())} unusual rows of {len(self.flags)}, run_dir='{self.run_dir}')"


def _parse_contamination(value: Any) -> float | None:
    if value is None or str(value).strip().lower() == "auto":
        return None
    try:
        number = float(str(value).strip().rstrip("%")) / (
            100 if str(value).strip().endswith("%") else 1
        )
    except ValueError:
        raise PlainMLError(
            f"--contamination must be a share like 0.02 (or 2%), or auto; got '{value}'."
        ) from None
    if not 0 < number < 0.5:
        raise PlainMLError("--contamination must be between 0 and 0.5.")
    return number


def _fit_detector(name: str, X: np.ndarray, seed: int) -> tuple[Any, np.ndarray]:
    """Fit a detector; return it with its training scores (higher = more unusual)."""
    if name == "iforest":
        detector = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(X)
        return detector, -detector.score_samples(X)
    if name == "lof":
        detector = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), novelty=True).fit(X)
        return detector, -detector.negative_outlier_factor_
    if name == "ocsvm":
        detector = OneClassSVM(nu=0.05, gamma="scale").fit(X)
        return detector, -detector.score_samples(X)
    detector = RobustDistance().fit(X)
    return detector, -detector.score_samples(X)


def _percentiles(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.searchsorted(reference, values, side="right") / len(reference)


def _reasons(df: pd.DataFrame, schema: Schema, rows: pd.Index, limit: int = 2) -> dict[Any, str]:
    """For each row, the one or two values that are most out of the ordinary."""
    scores: dict[str, pd.Series] = {}
    texts: dict[str, pd.Series] = {}
    for column in schema.numeric:
        values = coerce_numeric(df[column])
        median = values.median()
        spread = (values - median).abs().median() * 1.4826 or values.std() or 1.0
        z = (values - median) / spread
        scores[column] = z.abs().loc[rows]
        texts[column] = pd.Series(
            [
                f"{column} = {fmt_value(v)} ({abs(zz):.0f}× the usual spread {'above' if zz > 0 else 'below'} typical)"
                for v, zz in zip(values.loc[rows], z.loc[rows], strict=False)
            ],
            index=rows,
        )
    for column in schema.categorical:
        values = df[column].astype(str)
        share = values.map(values.value_counts(normalize=True))
        scores[column] = (-np.log(share.clip(lower=1e-6)) / np.log(20)).loc[
            rows
        ]  # a 1-in-20 value scores 1
        texts[column] = pd.Series(
            [
                f"{column} = '{v}' (rare: {fmt_pct(s, 1)} of rows)"
                for v, s in zip(values.loc[rows], share.loc[rows], strict=False)
            ],
            index=rows,
        )
    if not scores:
        return {}
    matrix = pd.DataFrame(scores).fillna(0)
    reasons = {}
    for row in rows:
        best = matrix.loc[row].sort_values(ascending=False)
        chosen = [texts[c].loc[row] for c in best.index[:limit] if best[c] >= 1.5]
        reasons[row] = "; ".join(chosen) if chosen else "an unusual combination of values"
    return reasons


def detect_anomalies(
    data: Any,
    *,
    contamination: Any = "auto",
    label: str | None = None,
    algorithms: list[str] | None = None,
    drop: list[str] | None = None,
    output: str | Path | None = None,
    seed: int = 42,
    top: int = 10,
    out_dir: str | Path = "runs",
    name: str | None = None,
    report: bool = True,
    private: bool = False,
    verbose: bool = True,
    **load_options: Any,
) -> AnomalyResult:
    """Score how unusual every row is; flag the most unusual share (``contamination``)."""
    started = time.time()
    share = _parse_contamination(contamination)
    with quiet(not verbose):
        df = load_data(data, seed=seed, **load_options).reset_index(drop=True)
        n_rows = len(df)
        if n_rows < 20:
            raise PlainMLError("Anomaly detection needs at least 20 rows.")
        label_column = find_column(label, df.columns, "Label column") if label else None
        truth = None
        if label_column:
            values = _clean_labels(df[label_column])
            counts = values.value_counts()
            if len(counts) != 2:
                raise PlainMLError(
                    f"The label column '{label_column}' should have two values (anomaly / normal); it has {len(counts)}."
                )
            positive = counts.index[-1]  # the rarer value marks the anomalies
            truth = (values == positive).astype(int).to_numpy()
            note(
                f"Treating {label_column} = '{positive}' as a known anomaly ({fmt_pct(truth.mean(), 1)} of rows)."
            )
        exclude = [label_column] if label_column else []
        schema, infos = infer_schema(
            df, exclude=exclude, drop=[find_column(c, df.columns) for c in drop or []]
        )
        if not schema.features:
            raise PlainMLError("No usable columns to look for anomalies in.")
        source = describe_source(data)
        prof = build_profile(df, schema, infos, source)

        chosen = []
        for raw in algorithms or ["iforest", "lof", "robust", "ocsvm"]:
            key = raw.strip().lower()
            if key not in DETECTORS:
                raise PlainMLError(
                    f"Unknown detector '{raw}'.{did_you_mean(key, DETECTORS)}",
                    hint="Choose from: " + ", ".join(DETECTORS),
                )
            if key == "ocsvm" and n_rows > OCSVM_MAX_ROWS:
                if algorithms:
                    note(f"Skipping ocsvm: too slow for {n_rows:,} rows.")
                continue
            chosen.append(key)

        preprocess = build_preprocessor(schema)
        heading(f"Looking for unusual rows among {n_rows:,} ({len(schema.features)} columns)")
        detectors: dict[str, Any] = {}
        reference: dict[str, np.ndarray] = {}
        scores: dict[str, np.ndarray] = {}
        with console.status("Running detectors…"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = np.asarray(preprocess.fit_transform(df), dtype=float)
            for key in chosen:
                try:
                    detector, raw_scores = _fit_detector(key, X, seed)
                except Exception as exc:
                    note(f"{DETECTORS[key]} failed: {esc(str(exc).splitlines()[0][:100])}")
                    continue
                detectors[key] = detector
                reference[key] = np.sort(raw_scores)
                scores[key] = _percentiles(raw_scores, reference[key])
        if not detectors:
            raise PlainMLError("Every detector failed on this data.")
        combined = np.mean(list(scores.values()), axis=0)

        rows = []
        use = list(detectors)
        if truth is not None and 0 < truth.sum() < len(truth):
            candidates = {**scores, "ensemble": combined}
            positives = int(truth.sum())
            for key, values in candidates.items():
                order = np.argsort(-values)[:positives]
                rows.append(
                    {
                        "detector": "Average of all" if key == "ensemble" else DETECTORS[key],
                        "key": key,
                        "average_precision": float(average_precision_score(truth, values)),
                        "roc_auc": float(roc_auc_score(truth, values)),
                        f"precision_at_{positives}": float(truth[order].mean()),
                    }
                )
            board = (
                pd.DataFrame(rows)
                .sort_values("average_precision", ascending=False)
                .reset_index(drop=True)
            )
            best_key = board.iloc[0]["key"]
            if best_key != "ensemble":
                use = [best_key]
                combined = scores[best_key]
        else:
            top_n = max(1, int(round(n_rows * (share or DEFAULT_CONTAMINATION))))
            ensemble_top = set(np.argsort(-combined)[:top_n])
            for key, values in scores.items():
                overlap = len(ensemble_top & set(np.argsort(-values)[:top_n])) / top_n
                rows.append(
                    {"detector": DETECTORS[key], "key": key, "agreement_with_average": overlap}
                )
            board = pd.DataFrame(rows)

        if share is None:
            share = float(truth.mean()) if truth is not None else DEFAULT_CONTAMINATION
            if truth is None:
                note(
                    f"Flagging the most unusual {fmt_pct(share, 0)} of rows; change it with --contamination."
                )
        threshold = float(np.quantile(combined, 1 - share))
        model = AnomalyModel(
            preprocess=preprocess,
            detectors={k: detectors[k] for k in use},
            reference={k: reference[k] for k in use},
            threshold=threshold,
        )
        flags = combined >= threshold
        flagged_index = df.index[flags]
        order = np.argsort(-combined)
        reasons = _reasons(df, schema, flagged_index)
        scored = df.copy()
        scored["anomaly_score"] = combined.round(4)
        scored["is_anomaly"] = flags
        scored["why"] = [reasons.get(i, "") for i in df.index]
        top_rows = scored.iloc[order[: max(top, 1)]]

        run_dir = create_run_dir(out_dir, name or f"{source_stem(data)}-anomalies")
        created = datetime.now().isoformat(timespec="seconds")
        model.plainml_meta_ = {
            "plainml_version": __version__,
            "packages": environment()["packages"],
            "created": created,
            "run": run_dir.name,
            "task": ANOMALY,
            "targets": [],
            "features": schema.features,
            "schema": schema.to_dict(),
            "model": " + ".join(DETECTORS[k] for k in use),
            "threshold": threshold,
            "contamination": share,
        }
        joblib.dump(model, run_dir / MODEL_FILE, compress=3)
        save_table(scored, run_dir / SCORED_FILE)
        if output:
            save_table(scored, output)
        board.to_csv(run_dir / LEADERBOARD_FILE, index=False)
        from plainml.report import histogram_bins

        shown_columns = schema.features[:6]
        write_json(
            run_dir / EVALUATION_FILE,
            {
                "histogram": histogram_bins(combined.tolist(), max_bins=40),
                "threshold": threshold,
                "flagged": int(flags.sum()),
                "top": [
                    {
                        "row": int(i),
                        "score": float(scored.loc[i, "anomaly_score"]),
                        "why": "" if private else scored.loc[i, "why"] or "",
                        "values": {}
                        if private
                        else {c: _json_value(scored.loc[i, c]) for c in shown_columns},
                    }
                    for i in top_rows.index
                ],
                "columns": [] if private else shown_columns,
                "labelled": truth is not None,
            },
        )
        write_json(
            run_dir / RUN_FILE,
            {
                "kind": "anomaly",
                "plainml_version": __version__,
                "environment": environment(),
                "created": created,
                "task": ANOMALY,
                "task_label": "anomaly detection",
                "target": label_column,
                "metric": "average_precision" if truth is not None else None,
                "data": {
                    "source": shown_source(source, private),
                    "rows": n_rows,
                    "raw_rows": n_rows,
                    "columns": df.shape[1],
                    "fingerprint": fingerprint(df),
                },
                "schema": schema.to_dict(),
                "profile": {
                    "issues": [i.__dict__ for i in prof.issues],
                    "columns": redact_columns([c.to_dict() for c in prof.columns])
                    if private
                    else [c.to_dict() for c in prof.columns],
                },
                "best": {
                    "key": "+".join(use),
                    "name": model.plainml_meta_["model"],
                    "score": float(board.iloc[0].get("average_precision", np.nan))
                    if truth is not None
                    else None,
                    "flagged": int(flags.sum()),
                    "share": share,
                },
                "options": {
                    "contamination": str(contamination),
                    "algorithms": chosen,
                    "seed": seed,
                    "label": label_column,
                },
                "timings": {"total_seconds": round(time.time() - started, 2)},
                "files": {
                    "model": MODEL_FILE,
                    "scored": SCORED_FILE,
                    "report": REPORT_FILE if report else None,
                },
            },
        )
        if report:
            try:
                from plainml.report import write_report

                write_report(run_dir)
            except Exception as exc:
                warn(f"Couldn't write the HTML report: {esc(str(exc)[:120])}")

    result = AnomalyResult(
        run_dir, scored["anomaly_score"], scored["is_anomaly"], top_rows, board, model
    )
    if verbose:
        _render(result, truth is not None, output)
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value if isinstance(value, (int, float, str, bool)) or value is None else str(value)


def _render(result: AnomalyResult, labelled: bool, output: Any) -> None:
    board = result.leaderboard
    heading("Detectors")
    table = Table(box=None, header_style="muted", pad_edge=False)
    for column in board.columns:
        if column != "key":
            table.add_column(
                column.replace("_", " "), justify="left" if column == "detector" else "right"
            )
    for i, row in board.iterrows():
        style = "best" if labelled and i == 0 else ""
        table.add_row(
            *[
                str(row[c]) if c == "detector" else fmt_num(row[c])
                for c in board.columns
                if c != "key"
            ],
            style=style,
        )
    console.print(table)
    if labelled:
        note(
            "Average precision: how well known anomalies are ranked at the top (1.0 is perfect; the label rate is random)."
        )
    else:
        note("Agreement: share of the flagged rows each detector also puts in its own top list.")
    flagged = int(result.flags.sum())
    heading(f"Most unusual rows ({flagged:,} flagged, {fmt_pct(flagged / len(result.flags), 1)})")
    rows = Table(box=None, header_style="muted", pad_edge=False)
    rows.add_column("Row", justify="right", style="muted")
    rows.add_column("Score", justify="right", style="bold")
    rows.add_column("Why it stands out", overflow="fold")
    for index, row in result.top.iterrows():
        rows.add_row(
            str(index),
            f"{row['anomaly_score']:.3f}",
            esc(row["why"] or "(below the flag threshold)"),
        )
    console.print(rows)
    note("Score: share of rows this one is more unusual than (1.0 = the most unusual).")
    heading("Saved")
    console.print(f"[bold]{esc(result.run_dir)}[/]")
    console.print(f"  {SCORED_FILE:<22}[muted]every row with anomaly_score, is_anomaly and why[/]")
    if output:
        console.print(f"  [muted]Also saved to {esc(output)}[/]")
    info("Score new rows with: plainml predict latest NEW_DATA.csv")


def render_anomaly_report(run: Any) -> str:
    from plainml.report import (
        as_table_view,
        card,
        chart,
        grid,
        page,
        profile_section,
        section,
        stat_tiles,
        table,
    )

    info_json, evaluation, board = run.info, run.evaluation, run.leaderboard
    best = info_json["best"]
    charts: dict[str, Any] = {}
    bins = evaluation.get("histogram", [])
    threshold = evaluation.get("threshold")
    tiles = [
        (
            "Unusual rows flagged",
            f"{best['flagged']:,}",
            f"{fmt_pct(best['flagged'] / info_json['data']['rows'], 1)} of {info_json['data']['rows']:,}",
        ),
        ("Detector", best["name"], None),
    ]
    if best.get("score") is not None:
        tiles.append(
            (
                "Average precision (vs your labels)",
                fmt_num(best["score"]),
                "1.0 = every known anomaly ranked first",
            )
        )
    body = section("Results", stat_tiles(tiles), anchor="results")
    if bins:
        charts["scores"] = {
            "type": "columns",
            "bins": bins,
            "xDomain": [bins[0]["x0"], bins[-1]["x1"]],
            "yDomain": [0, max(b["count"] for b in bins) * 1.1],
            "xLabel": "anomaly score (1 = most unusual)",
            "yLabel": "rows",
            "refX": threshold,
        }
    columns = evaluation.get("columns", [])
    top_rows = [
        [str(t["row"]), t["score"], t["why"], *[t["values"].get(c) for c in columns]]
        for t in evaluation.get("top", [])
    ]
    top_table = table(["Row", "Score", "Why it stands out", *columns], top_rows, numeric={0, 1})
    board_rows = [[r[c] for c in board.columns if c != "key"] for _, r in board.iterrows()]
    board_headers = [c.replace("_", " ") for c in board.columns if c != "key"]
    body += section(
        "Findings",
        grid(
            card(top_table, "Most unusual rows", wide=True),
            card(
                chart("scores")
                + as_table_view(
                    table(
                        ["Score range", "Rows"],
                        [[f"{fmt_num(b['x0'])}–{fmt_num(b['x1'])}", b["count"]] for b in bins],
                        numeric={1},
                    )
                ),
                "How unusual rows are",
                note=f"The vertical line is the flag threshold ({fmt_num(threshold)}).",
            ),
            card(
                table(board_headers, board_rows, numeric=set(range(1, len(board_headers)))),
                "Detectors",
            ),
        ),
        anchor="findings",
    )
    body += section(
        "The data",
        profile_section(info_json.get("profile", {}), info_json.get("schema")),
        anchor="data",
    )
    title = f"Unusual rows in {Path(str(info_json['data']['source'])).name}"
    subtitle = f"anomaly detection · {info_json['data']['rows']:,} rows · {info_json['created'][:16].replace('T', ' ')}"
    return page(
        title,
        subtitle,
        body,
        charts,
        [("results", "Results"), ("findings", "Findings"), ("data", "Data")],
    )
