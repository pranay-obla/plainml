"""``plainml web``: a local website for everything plainml does.

Upload a file, pick what to do with it (train, cluster, find anomalies, forecast, rank
columns, check drift, profile, clean or predict), watch it run, then read the results and
download any of the files one by one.

The page is plain HTML/CSS/JS in ``static/``; this module is its JSON API:

    GET  /api/info                     version, what's installed, whether a token is needed
    POST /api/login                    {"token": ...} sets a cookie (only with --token)
    POST /api/uploads                  multipart file -> column summary and preview
    GET  /api/uploads/{id}             the same summary again
    POST /api/jobs                     {"task", "upload", "options"} -> a queued job
    GET  /api/jobs/{id}?log_from=N     status, stage, progress, new log lines, result
    GET  /api/jobs/{id}/files/{path}   a file a job wrote (predictions, cleaned data...)
    GET  /api/runs                     saved runs, newest first
    GET  /api/runs/{run}               one run: headline numbers and its files
    GET  /api/runs/{run}/files/{path}  a run's file (add ?download=1 to save it)
    GET  /api/runs/{run}/preview/{path} the first rows of a CSV, as JSON
    GET  /api/models                   runs whose model can predict

The same functions also run inside the browser (``plainml.web.browser``) for the static,
serverless version of the site, so both behave the same.
"""

# No `from __future__ import annotations` here: FastAPI reads the endpoint annotations
# at runtime, and Request/UploadFile are imported inside create_app.
import json
import math
import os
import re
import secrets
import shutil
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plainml import __version__
from plainml.console import esc, fmt_num, fmt_pct, info, note, parse_duration
from plainml.errors import PlainMLError, is_installed, require
from plainml.importance import METHODS
from plainml.io import READERS, load_data
from plainml.runs import (
    DEFAULT_RUNS_DIR,
    MODEL_FILE,
    REPORT_FILE,
    RUN_FILE,
    list_runs,
    read_json,
)
from plainml.schema import Kind, infer_column
from plainml.web.jobs import Job, JobRunner

STATIC = Path(__file__).with_name("static")
WEB_DIR = ".web"  # inside the runs folder: uploads and job outputs
COOKIE = "plainml_token"
PREVIEW_ROWS = 50
UPLOAD_SUFFIXES = sorted(READERS)
TARGET_NAMES = (
    "target", "label", "y", "class", "outcome", "churn", "churned", "price", "sales",
    "revenue", "survived", "default", "fraud", "is_fraud", "species", "quality",
)  # fmt: skip

# What each file is, in the order files are listed (unlisted files come last).
FILES = {
    REPORT_FILE: ("Report", "Charts and plain-English explanations"),
    "forecast.csv": ("Forecast", "Future values with an 80% range"),
    "predictions.csv": ("Predictions", "Your rows with the model's predictions"),
    "clustered.csv": ("Rows with clusters", "Your rows with the cluster each one belongs to"),
    "scored.csv": ("Rows with anomaly scores", "Your rows scored, with the unusual ones flagged"),
    "rankings.csv": ("Column rankings", "Every column's rank under each method"),
    "selected_columns.txt": ("Selected columns", "The columns worth keeping, one per line"),
    "drift.csv": ("Drift by column", "How much each column changed, and how"),
    "cleaned.csv": ("Cleaned data", "Your data after cleaning"),
    "profile.html": ("Data profile", "Every column summarised, with warnings"),
    MODEL_FILE: ("Model", "The trained model, for plainml predict or joblib.load"),
    "model_card.md": ("Model card", "A one-page summary to share with the model"),
    "leaderboard.csv": ("Leaderboard", "Every model's scores"),
    "holdout_predictions.csv": ("Test predictions", "Held-out rows, actual vs predicted"),
    "importance.csv": ("Column importance", "How much each column matters to the model"),
    "config.yaml": ("Config", "The settings, to rerun this training exactly"),
    "evaluation.json": ("Evaluation data", "The numbers behind the report's charts"),
    RUN_FILE: ("Run details", "Settings, scores and library versions"),
}
PREDICTABLE = ("train", "tune", "cluster", "anomaly", "forecast")
TASK_TITLES = {
    "train": "Train a model",
    "cluster": "Find groups",
    "anomaly": "Find anomalies",
    "forecast": "Forecast",
    "importance": "Rank columns",
    "drift": "Check drift",
    "profile": "Profile data",
    "clean": "Clean data",
    "predict": "Predict",
}


def _plain(value: Any) -> Any:
    """A JSON-safe version of a cell value."""
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else str(value).removesuffix(" 00:00:00")
    if isinstance(value, (int, str, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def table_json(frame: pd.DataFrame, rows: int = PREVIEW_ROWS) -> dict[str, Any]:
    head = frame.head(rows)
    return {
        "columns": [str(c) for c in head.columns],
        "rows": [[_plain(v) for v in row] for row in head.itertuples(index=False)],
        "total": int(frame.attrs.get("rows_written", len(frame))),
    }


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(name).name).strip(" .") or "data"
    return cleaned[:120]


class Workspace:
    """Where the app keeps its files: runs in ``runs_dir``, uploads and outputs in ``.web``."""

    def __init__(self, runs_dir: str | Path):
        self.runs = Path(runs_dir).expanduser().resolve()
        self.uploads = self.runs / WEB_DIR / "uploads"
        self.outputs = self.runs / WEB_DIR / "outputs"
        for folder in (self.uploads, self.outputs):
            folder.mkdir(parents=True, exist_ok=True)

    def run_dir(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise PlainMLError(f"No run called '{name}'.")
        path = self.runs / name
        if not (path / RUN_FILE).is_file():
            raise PlainMLError(f"No run called '{name}'.")
        return path

    def upload_path(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", upload_id or ""):
            raise PlainMLError("Unknown upload. Upload the file again.")
        folder = self.uploads / upload_id
        meta = folder / "upload.json"
        if not meta.is_file():
            raise PlainMLError("Unknown upload. Upload the file again.")
        return folder / read_json(meta)["file"]

    def output_dir(self, job_id: str) -> Path:
        folder = self.outputs / job_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder


def inside(folder: Path, relative: str) -> Path:
    """``folder/relative``, refusing anything that escapes the folder."""
    path = (folder / relative).resolve()
    if folder.resolve() not in path.parents or not path.is_file():
        raise PlainMLError(f"No file '{relative}'.")
    return path


def file_list(folder: Path, base_url: str) -> list[dict[str, Any]]:
    order = list(FILES)
    found = [p for p in folder.rglob("*") if p.is_file() and not p.name.startswith(".")]
    found = [p for p in found if p.name != "upload.json"]
    found.sort(key=lambda p: (order.index(p.name) if p.name in order else len(order), str(p)))
    files = []
    for path in found:
        relative = path.relative_to(folder).as_posix()
        label, description = FILES.get(path.name, (path.name, ""))
        if path.parent != folder and path.suffix == ".joblib":
            label, description = f"Model: {path.stem}", "Another trained model from the leaderboard"
        files.append(
            {
                "path": relative,
                "name": path.name,
                "label": label,
                "description": description,
                "size": path.stat().st_size,
                "type": path.suffix.lstrip(".").lower(),
                "url": f"{base_url}/files/{relative}",
                "download": f"{base_url}/files/{relative}?download=1",
                "preview": f"{base_url}/preview/{relative}"
                if path.suffix in (".csv", ".tsv")
                else None,
            }
        )
    return files


def suggest_target(columns: list[dict[str, Any]]) -> str | None:
    usable = [c for c in columns if c["kind"] in (Kind.NUMERIC, Kind.CATEGORICAL)]
    for column in usable:
        name = column["name"].lower().strip()
        if name in TARGET_NAMES or name.startswith(("is_", "has_")):
            return str(column["name"])
    return str(usable[-1]["name"]) if usable else None


def summarize_upload(path: Path, upload_id: str) -> dict[str, Any]:
    frame = load_data(path)
    sample = frame.sample(20_000, random_state=0) if len(frame) > 20_000 else frame
    columns = []
    for name in frame.columns:
        column = infer_column(sample[name])
        columns.append(
            {
                "name": str(name),
                "kind": column.kind,
                "missing_pct": round(float(frame[name].isna().mean()) * 100, 1),
                "unique": int(column.n_unique),
                "examples": [str(v) for v in column.examples[:3]],
                "note": column.note,
            }
        )
    return {
        "id": upload_id,
        "name": path.name,
        "size": path.stat().st_size,
        "rows": len(frame),
        "columns": columns,
        "preview": table_json(frame, 30),
        "suggested_target": suggest_target(columns),
        "date_columns": [c["name"] for c in columns if c["kind"] == Kind.DATETIME],
    }


# --- summaries of runs -----------------------------------------------------------------------


def _tile(label: str, value: Any, sub: str | None = None) -> dict[str, Any]:
    return {"label": label, "value": value, "sub": sub}


def run_tiles(run: dict[str, Any], evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    kind = run.get("kind", "train")
    best = run.get("best") or {}
    data = run.get("data") or {}
    rows = f"{data.get('rows', 0):,} rows"
    if kind in ("train", "tune"):
        metric = run.get("metric_label", run.get("metric", "score"))
        holdout = (best.get("holdout") or {}).get(run.get("metric"))
        tiles = [
            _tile(f"Test {metric}", fmt_num(holdout), "on rows the model never saw"),
            _tile("Best model", best.get("name")),
            _tile(
                f"Cross-validated {metric}",
                fmt_num(best.get("cv_score")),
                f"± {fmt_num(best.get('cv_std'))}" if best.get("cv_std") is not None else None,
            ),
        ]
        baseline = run.get("baseline") or {}
        if baseline.get("cv_score") is not None:
            tiles.append(
                _tile(
                    "Simple baseline", fmt_num(baseline["cv_score"]), "every model should beat this"
                )
            )
        return tiles
    if kind == "forecast":
        options = run.get("options") or {}
        tiles = [
            _tile("Average error (MAE)", fmt_num(best.get("score")), "in backtests"),
            _tile("Best model", best.get("name")),
            _tile("Horizon", f"{options.get('horizon')} {evaluation.get('unit', 'period')}s"),
        ]
        if options.get("groups"):
            tiles.append(_tile("Series", str(options["groups"]), f"one per {options.get('group')}"))
        return tiles
    if kind == "cluster":
        return [
            _tile("Groups found", str(best.get("k"))),
            _tile("Method", best.get("name")),
            _tile(
                "Silhouette", fmt_num(best.get("score")), "separation, −1 to 1 (higher is clearer)"
            ),
            _tile("Rows", f"{data.get('rows', 0):,}"),
        ]
    if kind == "anomaly":
        return [
            _tile(
                "Unusual rows",
                f"{best.get('flagged', 0):,}",
                fmt_pct(best.get("share") or 0, 1) + " of rows",
            ),
            _tile("Methods", best.get("name")),
            _tile("Rows checked", f"{data.get('rows', 0):,}"),
        ]
    if kind == "importance":
        selected = best.get("selected") or []
        return [
            _tile(
                "Columns worth keeping",
                str(len(selected)),
                ", ".join(selected[:4]) + (" …" if len(selected) > 4 else ""),
            ),
            _tile(f"{run.get('metric_label', 'Score')} with them", fmt_num(best.get("score"))),
            _tile("Methods compared", str(len(run.get("methods") or []))),
            _tile("Data", rows),
        ]
    if kind == "drift":
        verdict = str(best.get("name", "")).capitalize()
        drifted = evaluation.get("drifted") or []
        return [
            _tile("Verdict", verdict or "—"),
            _tile(
                "Columns that changed", str(len(drifted)), ", ".join(map(str, drifted[:4])) or None
            ),
            _tile("New data", rows),
        ]
    return [_tile("Rows", rows)]


def run_summary(ws: Workspace, name: str) -> dict[str, Any]:
    folder = ws.run_dir(name)
    run = read_json(folder / RUN_FILE)
    evaluation_path = folder / "evaluation.json"
    evaluation = read_json(evaluation_path) if evaluation_path.is_file() else {}
    if (folder / "drift.csv").is_file():
        by_column = pd.read_csv(folder / "drift.csv")
        evaluation["drifted"] = by_column.loc[by_column["status"] != "stable", "column"].tolist()
    base = f"/api/runs/{name}"
    kind = run.get("kind", "train")
    source = Path(str((run.get("data") or {}).get("source", ""))).name
    target = run.get("target")
    title = {
        "train": f"Predicting {target}",
        "tune": f"Predicting {target} (tuned)",
        "forecast": f"Forecasting {target}",
        "cluster": "Groups in the data",
        "anomaly": "Unusual rows",
        "importance": f"Which columns matter for {target}",
        "drift": "Has the data changed?",
    }.get(kind, name)
    sentences = evaluation.get("sentences") or evaluation.get("insights") or []
    return {
        "name": name,
        "kind": kind,
        "title": title,
        "subtitle": " · ".join(
            x
            for x in (run.get("task_label"), source, run.get("created", "")[:16].replace("T", " "))
            if x
        ),
        "tiles": run_tiles(run, evaluation),
        "sentences": [str(s) for s in sentences[:6]],
        "files": file_list(folder, base),
        "report": f"{base}/files/{REPORT_FILE}" if (folder / REPORT_FILE).is_file() else None,
        "predictable": kind in PREDICTABLE and (folder / MODEL_FILE).is_file(),
    }


# --- running tasks -------------------------------------------------------------------------------


def _columns(value: Any) -> list[str] | None:
    if not value:
        return None
    items = value if isinstance(value, list) else str(value).split(",")
    return [str(v).strip() for v in items if str(v).strip()] or None


def _number(value: Any, kind: type = float) -> Any:
    if value in (None, "", "auto"):
        return None
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise PlainMLError(f"'{value}' isn't a number.") from exc


def build_work(ws: Workspace, task: str, upload: str | None, options: dict[str, Any]) -> Any:
    """Check the request and return the function the job will run."""
    if task not in TASK_TITLES:
        raise PlainMLError(f"Unknown task '{task}'.")
    data = ws.upload_path(upload) if upload else None
    if data is None and task not in ("predict",):
        raise PlainMLError("Upload a data file first.")
    target = options.get("target") or None
    private = bool(options.get("private"))
    out = str(ws.runs)

    def ran(result: Any) -> dict[str, Any]:
        return {"kind": "run", "run": Path(result.run_dir).name}

    if task == "train":
        if not target:
            raise PlainMLError("Pick the column to predict.")
        speed = options.get("speed", "standard")
        settings: dict[str, Any] = {
            "quick": speed == "quick",
            "thorough": speed == "thorough",
            "private": private,
            "calibrate": bool(options.get("calibrate")),
            "log_target": bool(options.get("log_target")),
            "drop": _columns(options.get("drop")) or [],
            "models": _columns(options.get("models")),
            "metric": options.get("metric") or None,
            "task": options.get("task") or None,
            "time_budget": parse_duration(options.get("time_budget") or None),
            "out_dir": out,
        }

        def work(job: Job) -> dict[str, Any]:
            from plainml.training import train

            def progress(fraction: float, message: str) -> None:
                job.progress = max(0.02, min(0.97, fraction))
                job.stage = message
                job.touch()

            return ran(
                train(
                    str(data),
                    target,
                    progress=progress,
                    **{k: v for k, v in settings.items() if v is not None},
                )
            )

        return work
    if task == "cluster":

        def work(job: Job) -> dict[str, Any]:
            from plainml.clustering import cluster

            k = options.get("k") or "auto"
            return ran(
                cluster(
                    str(data),
                    k=k if k == "auto" else int(k),
                    drop=_columns(options.get("drop")),
                    private=private,
                    out_dir=out,
                )
            )

        return work
    if task == "anomaly":

        def work(job: Job) -> dict[str, Any]:
            from plainml.anomaly import detect_anomalies

            share = _number(options.get("contamination"))
            return ran(
                detect_anomalies(
                    str(data),
                    contamination=share / 100 if share else "auto",
                    label=options.get("label") or None,
                    drop=_columns(options.get("drop")),
                    private=private,
                    out_dir=out,
                )
            )

        return work
    if task == "forecast":
        if not target:
            raise PlainMLError("Pick the column to forecast.")

        def work(job: Job) -> dict[str, Any]:
            from plainml.forecasting import forecast

            return ran(
                forecast(
                    str(data),
                    target,
                    date=options.get("date") or None,
                    horizon=_number(options.get("horizon"), int),
                    freq=options.get("freq") or None,
                    agg=options.get("agg") or "sum",
                    group=options.get("group") or None,
                    inputs=_columns(options.get("inputs")),
                    country=options.get("country") or None,
                    out_dir=out,
                )
            )

        return work
    if task == "importance":
        if not target:
            raise PlainMLError("Pick the column the others should predict.")

        def work(job: Job) -> dict[str, Any]:
            from plainml.importance import feature_importance

            def progress(fraction: float, message: str) -> None:
                job.progress = max(0.02, min(0.97, fraction))
                job.stage = message
                job.touch()

            return ran(
                feature_importance(
                    str(data),
                    target,
                    methods=_columns(options.get("methods")),
                    k=_number(options.get("k"), int),
                    drop=_columns(options.get("drop")),
                    private=private,
                    out_dir=out,
                    progress=progress,
                )
            )

        return work
    if task == "drift":
        reference_run = options.get("reference_run")
        reference_upload = options.get("reference_upload")
        if reference_run:
            reference: Any = ws.run_dir(reference_run)
        elif reference_upload:
            reference = ws.upload_path(reference_upload)
        else:
            raise PlainMLError("Choose what to compare against: a trained model or another file.")

        def work(job: Job) -> dict[str, Any]:
            from plainml.drift import check_drift

            return ran(check_drift(str(reference), str(data), target=target, out_dir=out))

        return work
    if task == "profile":

        def work(job: Job) -> dict[str, Any]:
            from plainml.profiling import profile
            from plainml.report import write_profile_report

            folder = ws.output_dir(job.id)
            result = profile(str(data), target)
            write_profile_report(folder / "profile.html", result)
            summary = result.to_dict()
            return output_result(
                job,
                folder,
                "Data profile",
                [
                    _tile("Rows", f"{summary['n_rows']:,}"),
                    _tile(
                        "Columns",
                        str(summary["n_cols"]),
                        f"{len(result.schema.features)} usable as inputs",
                    ),
                    _tile("Duplicate rows", f"{summary['n_duplicates']:,}"),
                    _tile("Warnings", str(len(result.warnings))),
                ],
                report="profile.html",
            )

        return work
    if task == "clean":

        def work(job: Job) -> dict[str, Any]:
            from plainml.cleaning import clean

            folder = ws.output_dir(job.id)
            before = load_data(str(data))
            cleaned = clean(
                str(data),
                output=folder / "cleaned.csv",
                target=target,
                drop_duplicates=options.get("drop_duplicates", True),
                impute=bool(options.get("impute")),
                outliers=options.get("outliers") or "none",
                encode=bool(options.get("encode")),
            )
            return output_result(
                job,
                folder,
                "Cleaned data",
                [
                    _tile("Rows", f"{len(cleaned):,}", f"{len(before) - len(cleaned):,} removed"),
                    _tile("Columns", str(cleaned.shape[1]), f"was {before.shape[1]}"),
                    _tile(
                        "Blank cells",
                        f"{int(cleaned.isna().sum().sum()):,}",
                        f"was {int(before.isna().sum().sum()):,}",
                    ),
                ],
                preview=table_json(cleaned),
            )

        return work
    # predict
    model_run = options.get("model")
    if not model_run:
        raise PlainMLError("Choose a model to predict with.")
    model_dir = ws.run_dir(model_run)
    kind = read_json(model_dir / RUN_FILE).get("kind")
    if data is None and kind != "forecast":
        raise PlainMLError("Upload the rows to predict.")

    def predict_work(job: Job) -> dict[str, Any]:
        from plainml.predicting import predict

        folder = ws.output_dir(job.id)
        frame = predict(
            model_dir,
            str(data) if data is not None else None,
            output=folder / "predictions.csv",
            proba=bool(options.get("proba")),
            horizon=_number(options.get("horizon"), int),
        )
        if kind == "forecast":
            (folder / "predictions.csv").rename(folder / "forecast.csv")
        made = [c for c in frame.columns if str(c).startswith(("predicted_", "probability_"))]
        made += [
            c
            for c in (
                "confidence",
                "cluster",
                "anomaly_score",
                "is_anomaly",
                "forecast",
                "lower_80",
                "upper_80",
            )
            if c in frame.columns
        ]
        shown = frame[[*[c for c in frame.columns if c not in made][:4], *made]]
        tiles = [
            _tile("Rows forecast" if kind == "forecast" else "Rows predicted", f"{len(frame):,}"),
            _tile("Model", model_run),
        ]
        return output_result(job, folder, "Predictions", tiles, preview=table_json(shown))

    return predict_work


def output_result(
    job: Job,
    folder: Path,
    title: str,
    tiles: list[dict[str, Any]],
    preview: dict[str, Any] | None = None,
    report: str | None = None,
) -> dict[str, Any]:
    base = f"/api/jobs/{job.id}"
    return {
        "kind": "output",
        "title": title,
        "tiles": tiles,
        "files": file_list(folder, base),
        "preview": preview,
        "report": f"{base}/files/{report}" if report else None,
    }


# --- the API's answers, shared by the server and the in-browser version -----------------------------


def info_payload(
    ws: Workspace,
    *,
    needs_token: bool = False,
    signed_in: bool = True,
    max_upload_mb: int = 500,
    mode: str = "server",
) -> dict[str, Any]:
    return {
        "version": __version__,
        "mode": mode,
        "needs_token": needs_token,
        "signed_in": signed_in,
        "runs_dir": str(ws.runs),
        "upload_types": UPLOAD_SUFFIXES,
        "importance_methods": [
            {
                "key": m.key,
                "label": m.label,
                "about": m.about,
                "default": m.default,
                "family": m.family,
            }
            for m in METHODS
        ],
        "max_upload_mb": max_upload_mb,
        "installed": {
            name: is_installed(module)
            for name, module in (
                ("boost", "lightgbm"),
                ("torch", "torch"),
                ("holidays", "holidays"),
                ("tune", "optuna"),
                ("imbalance", "imblearn"),
            )
        },
    }


def start_upload(ws: Workspace, filename: str) -> tuple[str, Path]:
    """A new upload folder for ``filename``: returns (upload id, path to write the file to)."""
    name = _safe_name(filename or "data.csv")
    if Path(name).suffix.lower() not in READERS:
        raise PlainMLError(
            f"Can't read '{name}'.", hint="Upload one of: " + ", ".join(UPLOAD_SUFFIXES)
        )
    upload_id = uuid.uuid4().hex[:12]
    folder = ws.uploads / upload_id
    folder.mkdir(parents=True)
    return upload_id, folder / name


def finish_upload(upload_id: str, path: Path) -> dict[str, Any]:
    """Read a written upload and remember its summary (removing it if it can't be read)."""
    try:
        summary = summarize_upload(path, upload_id)
    except PlainMLError:
        shutil.rmtree(path.parent, ignore_errors=True)
        raise
    (path.parent / "upload.json").write_text(
        json.dumps({"file": path.name, **summary}), encoding="utf-8"
    )
    return summary


def upload_payload(ws: Workspace, upload_id: str) -> dict[str, Any]:
    stored = read_json(ws.upload_path(upload_id).parent / "upload.json")
    stored.pop("file", None)
    return stored


def runs_payload(ws: Workspace) -> list[dict[str, Any]]:
    frame = list_runs(ws.runs)
    if frame.empty:
        return []
    return [{k: _plain(v) for k, v in row.items()} for row in frame.iloc[::-1].to_dict("records")]


def models_payload(ws: Workspace) -> list[dict[str, Any]]:
    return [
        row
        for row in runs_payload(ws)
        if row["kind"] in PREDICTABLE and (ws.runs / row["run"] / MODEL_FILE).is_file()
    ]


def preview_payload(path: Path) -> dict[str, Any]:
    return table_json(pd.read_csv(path, nrows=PREVIEW_ROWS))


def media_type(path: Path) -> str | None:
    return {
        ".md": "text/markdown; charset=utf-8",
        ".yaml": "text/plain; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".json": "application/json",
    }.get(path.suffix)


# --- the app -----------------------------------------------------------------------------------


def create_app(
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    *,
    token: str | None = None,
    max_upload_mb: int = 500,
) -> Any:
    missing = [m for m in ("fastapi", "uvicorn", "multipart") if not is_installed(m)]
    if missing:
        raise PlainMLError(
            "The web app needs a few extra packages.",
            hint='Install them with: pip install "plainml[web]"',
        )
    import fastapi
    from fastapi import File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    ws = Workspace(runs_dir)
    runner = JobRunner()
    token = token if token is not None else os.environ.get("PLAINML_WEB_TOKEN") or None
    app = fastapi.FastAPI(title="plainml", version=__version__, docs_url=None, redoc_url=None)
    app.state.runner = runner
    app.state.workspace = ws

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        path = request.url.path
        open_paths = ("/api/info", "/api/login")
        if token and path.startswith("/api/") and path not in open_paths:
            sent = request.cookies.get(COOKIE, "")
            if not secrets.compare_digest(sent.encode(), token.encode()):
                return JSONResponse({"detail": "Sign in first."}, status_code=401)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    def fail(exc: PlainMLError, status: int = 422) -> HTTPException:
        return HTTPException(status, {"message": exc.message, "hint": exc.hint})

    @app.get("/api/info")
    def api_info(request: Request) -> dict[str, Any]:
        signed_in = not token or secrets.compare_digest(
            request.cookies.get(COOKIE, "").encode(), token.encode()
        )
        return info_payload(
            ws, needs_token=bool(token), signed_in=signed_in, max_upload_mb=max_upload_mb
        )

    @app.post("/api/login")
    def login(payload: dict[str, Any]) -> JSONResponse:
        sent = str(payload.get("token", ""))
        if not token or not secrets.compare_digest(sent.encode(), token.encode()):
            raise HTTPException(401, {"message": "That token isn't right."})
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=30 * 86400)
        return response

    @app.post("/api/uploads")
    def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            upload_id, path = start_upload(ws, file.filename or "data.csv")
        except PlainMLError as exc:
            raise fail(exc, 415) from exc
        limit = max_upload_mb * 1024 * 1024
        written = 0
        with path.open("wb") as handle:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    handle.close()
                    shutil.rmtree(path.parent, ignore_errors=True)
                    raise HTTPException(
                        413,
                        {
                            "message": f"The file is over {max_upload_mb} MB.",
                            "hint": "Start the app with a larger --max-upload-mb, or use the command line.",
                        },
                    )
                handle.write(chunk)
        try:
            return finish_upload(upload_id, path)
        except PlainMLError as exc:
            raise fail(exc) from exc

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str) -> dict[str, Any]:
        try:
            return upload_payload(ws, upload_id)
        except PlainMLError as exc:
            raise fail(exc, 404) from exc

    @app.post("/api/jobs")
    def start_job(payload: dict[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task", ""))
        try:
            work = build_work(ws, task, payload.get("upload"), payload.get("options") or {})
        except PlainMLError as exc:
            raise fail(exc) from exc
        job = runner.submit(task, TASK_TITLES.get(task, task), work)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, log_from: int = 0) -> dict[str, Any]:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, {"message": "No such job (the app may have restarted)."})
        return job.to_dict(max(0, log_from))

    def send(path: Path, download: bool) -> FileResponse:
        media = media_type(path) if path.suffix in (".md", ".yaml", ".txt") else None
        if download:
            return FileResponse(
                path, filename=path.name, media_type=media or "application/octet-stream"
            )
        return FileResponse(path, media_type=media)

    @app.get("/api/jobs/{job_id}/files/{relative:path}")
    def job_file(job_id: str, relative: str, download: bool = False) -> FileResponse:
        if runner.get(job_id) is None and not (ws.outputs / job_id).is_dir():
            raise HTTPException(404, {"message": "No such job."})
        try:
            return send(inside(ws.outputs / job_id, relative), download)
        except PlainMLError as exc:
            raise fail(exc, 404) from exc

    @app.get("/api/jobs/{job_id}/preview/{relative:path}")
    def job_preview(job_id: str, relative: str) -> dict[str, Any]:
        try:
            return preview_payload(inside(ws.outputs / job_id, relative))
        except PlainMLError as exc:
            raise fail(exc, 404) from exc

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return runs_payload(ws)

    @app.get("/api/runs/{name}")
    def run(name: str) -> dict[str, Any]:
        try:
            return run_summary(ws, name)
        except PlainMLError as exc:
            raise fail(exc, 404) from exc

    @app.get("/api/runs/{name}/files/{relative:path}")
    def run_file(name: str, relative: str, download: bool = False) -> FileResponse:
        try:
            return send(inside(ws.run_dir(name), relative), download)
        except PlainMLError as exc:
            raise fail(exc, 404) from exc

    @app.get("/api/runs/{name}/preview/{relative:path}")
    def run_preview(name: str, relative: str) -> dict[str, Any]:
        try:
            return preview_payload(inside(ws.run_dir(name), relative))
        except PlainMLError as exc:
            raise fail(exc, 404) from exc

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return models_payload(ws)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def run_web(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    token: str | None = None,
    max_upload_mb: int = 500,
    open_browser: bool = True,
) -> None:
    uvicorn = require("uvicorn", "The web app")
    app = create_app(runs_dir, token=token, max_upload_mb=max_upload_mb)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{port}"
    info(f"plainml web is running at {url}  (Ctrl-C to stop)")
    note(f"Runs are saved in {esc(app.state.workspace.runs)}")
    protected = bool(token or os.environ.get("PLAINML_WEB_TOKEN"))
    if host == "0.0.0.0" and not protected:
        note(
            "Listening on all network interfaces with no token: anyone who can reach this "
            "machine can upload data and run jobs. Add --token (or set PLAINML_WEB_TOKEN)."
        )
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
