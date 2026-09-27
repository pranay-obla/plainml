"""Run folders: where each training run's model, report and metadata are saved.

A run folder looks like::

    runs/20260924-172015_churn/
        model.joblib             the trained model (preprocessing included)
        report.html              charts and explanations
        leaderboard.csv          every model's cross-validated scores
        evaluation.json          held-out test results used by the report
        importance.csv           which columns mattered
        holdout_predictions.csv  test rows with actual vs predicted values
        run.json                 everything needed to reproduce the run
        config.yaml              the options, reusable with --config
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plainml.errors import PlainMLError

DEFAULT_RUNS_DIR = "runs"
RUN_FILE = "run.json"
MODEL_FILE = "model.joblib"
REPORT_FILE = "report.html"
LEADERBOARD_FILE = "leaderboard.csv"
EVALUATION_FILE = "evaluation.json"
IMPORTANCE_FILE = "importance.csv"
HOLDOUT_FILE = "holdout_predictions.csv"
CONFIG_FILE = "config.yaml"


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return slug[:40] or "run"


def create_run_dir(out_dir: str | Path, name: str) -> Path:
    base = Path(out_dir).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = base / f"{stamp}_{slugify(name)}"
    suffix = 2
    while path.exists():
        path = base / f"{stamp}_{slugify(name)}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _clean_nan(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _clean_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_nan(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_nan(value.item())
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_clean_nan(data), indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def environment() -> dict[str, Any]:
    """Python and library versions, saved with every run and model."""
    packages = {}
    for name in (
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
        "xgboost",
        "lightgbm",
        "catboost",
        "imbalanced-learn",
        "torch",
        "optuna",
        "shap",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _run_dirs(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    return sorted(
        (p for p in out_dir.iterdir() if (p / RUN_FILE).is_file()),
        key=lambda p: p.name,
    )


def resolve_run(ref: str | Path | None = "latest", out_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Find a run folder from 'latest', a folder path, a model file, or part of a run name."""
    out = Path(out_dir).expanduser()
    if ref is None or str(ref) in ("latest", "last", "@"):
        runs = _run_dirs(out)
        if not runs:
            raise PlainMLError(
                f"No runs found in '{out}'.",
                hint="Train a model first: plainml train DATA --target COLUMN",
            )
        return runs[-1]
    path = Path(ref).expanduser()
    if path.is_file():
        for folder in (path.parent, path.parent.parent):  # model.joblib, or models/<key>.joblib
            if (folder / RUN_FILE).is_file():
                return folder
        raise PlainMLError(f"'{ref}' is a file, not a plainml run folder.")
    if path.is_dir():
        if (path / RUN_FILE).is_file():
            return path
        raise PlainMLError(f"'{ref}' isn't a plainml run folder (no {RUN_FILE} inside).")
    text = str(ref)
    runs = _run_dirs(out)
    named = [p for p in runs if p.name.partition("_")[2] == text]
    if named:
        return named[-1]  # the newest run with exactly that name
    for matches in (
        [p for p in runs if p.name.startswith(text)],
        [p for p in runs if text in p.name],
    ):
        if len(matches) == 1:
            return matches[0]
        if matches:
            shown = ", ".join(p.name for p in matches[-5:])
            raise PlainMLError(
                f"'{ref}' matches {len(matches)} runs ({shown}{', …' if len(matches) > 5 else ''}).",
                hint="Use more of the name or the full folder name (see: plainml runs), or 'latest'.",
            )
    raise PlainMLError(
        f"Couldn't find a run called '{ref}'.",
        hint="List runs with: plainml runs",
    )


def resolve_model_path(ref: str | Path | None, out_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Find a model file from a .joblib path, a run folder, or 'latest'."""
    if ref is not None:
        path = Path(ref).expanduser()
        if path.is_file():
            if path.suffix not in (".joblib", ".pkl", ".pickle"):
                raise PlainMLError(
                    f"'{ref}' doesn't look like a model file.",
                    hint="Pass a .joblib model, a run folder, or 'latest'.",
                )
            return path
    run = resolve_run(ref, out_dir)
    model = run / MODEL_FILE
    if not model.is_file():
        raise PlainMLError(f"Run '{run.name}' has no {MODEL_FILE}.")
    return model


@dataclass
class Run:
    path: Path
    info: dict[str, Any]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def model_path(self) -> Path:
        return self.path / MODEL_FILE

    @property
    def report_path(self) -> Path:
        return self.path / REPORT_FILE

    @property
    def leaderboard(self) -> pd.DataFrame:
        file = self.path / LEADERBOARD_FILE
        return pd.read_csv(file) if file.is_file() else pd.DataFrame()

    @property
    def evaluation(self) -> dict[str, Any]:
        file = self.path / EVALUATION_FILE
        return read_json(file) if file.is_file() else {}

    @property
    def importance(self) -> pd.DataFrame:
        file = self.path / IMPORTANCE_FILE
        return pd.read_csv(file) if file.is_file() else pd.DataFrame()

    def load_model(self) -> Any:
        from plainml.predicting import load_model

        return load_model(self.model_path)[0]

    def __repr__(self) -> str:
        best = self.info.get("best", {})
        return f"Run({self.name!r}, task={self.info.get('task')!r}, best={best.get('name')!r})"


def load_run(ref: str | Path | None = "latest", out_dir: str | Path = DEFAULT_RUNS_DIR) -> Run:
    path = resolve_run(ref, out_dir)
    return Run(path=path, info=read_json(path / RUN_FILE))


def list_runs(out_dir: str | Path = DEFAULT_RUNS_DIR) -> pd.DataFrame:
    """One row per run: when, what data, which target, best model and its score."""
    rows = []
    for path in _run_dirs(Path(out_dir).expanduser()):
        try:
            info = read_json(path / RUN_FILE)
        except (OSError, json.JSONDecodeError):
            continue
        best = info.get("best") or {}
        target = info.get("target")
        rows.append(
            {
                "run": path.name,
                "created": info.get("created", "")[:16].replace("T", " "),
                "kind": info.get("kind", "train"),
                "task": info.get("task"),
                "data": Path(str(info.get("data", {}).get("source", ""))).name,
                "target": ", ".join(target) if isinstance(target, list) else target,
                "best_model": best.get("name"),
                "metric": info.get("metric"),
                "score": best.get("cv_score", best.get("score")),
                "rows": info.get("data", {}).get("rows"),
            }
        )
    return pd.DataFrame(rows)


def runs_to_prune(
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    *,
    keep: int | None = None,
    older_than: float | None = None,
    kind: str | None = None,
) -> list[Path]:
    """Run folders to delete: all but the newest ``keep``, and/or older than ``older_than`` seconds.

    With both, a run goes only if it fails both tests (it's outside the newest ``keep`` *and*
    old). ``kind`` limits pruning to one kind of run (train, forecast, importance...).
    """
    if keep is None and older_than is None:
        raise PlainMLError("Say what to prune: --keep N and/or --older-than 30d.")
    runs = _run_dirs(Path(out_dir).expanduser())
    if kind:
        runs = [p for p in runs if _kind_of(p) == kind]
    newest = set(runs[-keep:]) if keep else set()
    now = datetime.now()
    chosen = []
    for path in runs:
        if keep is not None and path in newest:
            continue
        if older_than is not None and (now - _created(path)).total_seconds() < older_than:
            continue
        chosen.append(path)
    return chosen


def _kind_of(path: Path) -> str:
    try:
        return str(read_json(path / RUN_FILE).get("kind", "train"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


def _created(path: Path) -> datetime:
    try:
        return datetime.fromisoformat(read_json(path / RUN_FILE)["created"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return datetime.fromtimestamp(path.stat().st_mtime)
