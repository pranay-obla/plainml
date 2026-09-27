"""Working out what kind of problem a target column describes, and preparing it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from plainml.errors import PlainMLError, did_you_mean
from plainml.schema import coerce_numeric, is_text_like

CLASSIFICATION = "classification"
REGRESSION = "regression"
MULTILABEL = "multilabel"
MULTI_REGRESSION = "multi_regression"
CLUSTERING = "clustering"
ANOMALY = "anomaly"
FORECAST = "forecast"

SUPERVISED = (CLASSIFICATION, REGRESSION, MULTILABEL, MULTI_REGRESSION)
CLASSIFICATION_LIKE = (CLASSIFICATION, MULTILABEL)

TASK_LABELS = {
    CLASSIFICATION: "classification",
    REGRESSION: "regression",
    MULTILABEL: "multi-label classification",
    MULTI_REGRESSION: "multi-output regression",
    CLUSTERING: "clustering",
    ANOMALY: "anomaly detection",
    FORECAST: "time-series forecasting",
}

_ALIASES = {
    "classification": CLASSIFICATION,
    "classify": CLASSIFICATION,
    "class": CLASSIFICATION,
    "categorical": CLASSIFICATION,
    "regression": REGRESSION,
    "regress": REGRESSION,
    "numeric": REGRESSION,
    "multilabel": MULTILABEL,
    "multi-label": MULTILABEL,
    "multi_label": MULTILABEL,
    "multi_regression": MULTI_REGRESSION,
    "multi-regression": MULTI_REGRESSION,
    "multioutput": MULTI_REGRESSION,
}

MAX_CLASSES_FOR_INTEGER_TARGET = 10


def normalize_task(task: str | None) -> str | None:
    if task is None or task == "auto":
        return None
    key = task.strip().lower()
    if key not in _ALIASES:
        raise PlainMLError(
            f"Unknown task '{task}'.{did_you_mean(key, _ALIASES)}",
            hint="Use classification or regression (or leave it out to auto-detect).",
        )
    return _ALIASES[key]


def detect_task(y: pd.Series) -> tuple[str, str]:
    """Guess classification vs regression for one target. Returns (task, reason)."""
    name = str(y.name)
    non_null = y.dropna()
    n_unique = int(non_null.nunique())
    if ptypes.is_datetime64_any_dtype(y):
        raise PlainMLError(
            f"The target '{name}' holds dates, which plainml can't predict directly.",
            hint="For predicting values over time, use: plainml forecast DATA --date COLUMN --target VALUE",
        )
    if ptypes.is_bool_dtype(y):
        return CLASSIFICATION, f"'{name}' is true/false"
    if is_text_like(y):
        converted = coerce_numeric(non_null)
        if converted.notna().mean() >= 0.95 and n_unique > MAX_CLASSES_FOR_INTEGER_TARGET:
            return (
                REGRESSION,
                f"'{name}' holds numbers (stored as text) with {n_unique} distinct values",
            )
        return CLASSIFICATION, f"'{name}' holds labels ({n_unique} classes)"
    values = non_null.astype(float)
    if n_unique == 2:
        a, b = sorted(values.unique())
        return CLASSIFICATION, f"'{name}' has two values ({_fmt(a)} and {_fmt(b)})"
    integer_valued = bool(np.all(np.mod(values, 1) == 0))
    if integer_valued and n_unique <= MAX_CLASSES_FOR_INTEGER_TARGET:
        return (
            CLASSIFICATION,
            f"'{name}' has only {n_unique} distinct whole-number values "
            "(use --task regression if they're amounts rather than categories)",
        )
    return REGRESSION, f"'{name}' is a number with {n_unique} distinct values"


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


@dataclass
class TargetInfo:
    names: list[str]
    task: str
    reason: str
    classes: list[Any] | None = None
    label_maps: dict[str, list[Any]] = field(default_factory=dict)
    dropped_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str | list[str]:
        return self.names[0] if len(self.names) == 1 else self.names

    @property
    def display(self) -> str:
        return ", ".join(self.names)


def to_python(value: Any) -> Any:
    """Convert numpy scalars to plain Python so they serialise to JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clean_labels(y: pd.Series) -> pd.Series:
    if ptypes.is_bool_dtype(y):
        return y.astype(bool)
    non_null = y.dropna()
    if len(non_null) and all(isinstance(v, (bool, np.bool_)) for v in non_null):
        return y.astype(bool)  # true/false that pandas stored as objects because of blanks
    if ptypes.is_numeric_dtype(y):
        values = y.astype(float)
        if bool(np.all(np.mod(values.dropna(), 1) == 0)):
            return values.astype("int64")
        return values
    return y.astype(object).map(lambda v: str(v).strip()).astype(object)


def prepare_target(
    df: pd.DataFrame, targets: list[str], task: str | None = None
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame, TargetInfo]:
    """Validate the target(s), pick the task and return (rows kept, y, info).

    Rows with a missing target are dropped, as are classes with a single example
    (they can't appear in both the training and the test split).
    """
    missing_target = df[targets].isna().any(axis=1)
    df = df.loc[~missing_target]
    dropped = int(missing_target.sum())
    if df.empty:
        raise PlainMLError(f"Every row is missing a value for {', '.join(targets)}.")

    for target in targets:
        if df[target].nunique() < 2:
            value = df[target].iloc[0]
            raise PlainMLError(
                f"The target '{target}' has the same value ({value}) in every row, "
                "so there's nothing to predict.",
                hint="Pick a column whose values vary.",
            )

    if len(targets) == 1:
        target = targets[0]
        detected, reason = detect_task(df[target])
        if task in (MULTILABEL, MULTI_REGRESSION):
            raise PlainMLError(f"Task '{task}' needs more than one target column.")
        chosen = task or detected
        if task and task != detected:
            reason = f"you chose {task} (auto-detect suggested {detected})"
        info = TargetInfo(names=[target], task=chosen, reason=reason, dropped_rows=dropped)
        if chosen == CLASSIFICATION:
            y = _clean_labels(df[target])
            counts = y.value_counts()
            if (counts >= 2).sum() < 2:
                raise PlainMLError(
                    f"The target '{target}' has {len(counts):,} different values in {len(y):,} rows, "
                    "so there are no repeated classes to learn from.",
                    hint="If it holds amounts, add --task regression. If it's an ID or a name, "
                    "choose another target column.",
                )
            rare = counts[counts < 2].index.tolist()
            if rare:
                keep = ~y.isin(rare)
                df, y = df.loc[keep], y.loc[keep]
                info.notes.append(
                    f"Dropped {len(rare)} class(es) with a single example: "
                    + ", ".join(map(str, rare[:5]))
                )
            info.classes = [to_python(c) for c in sorted(y.unique(), key=_sort_key)]
            return df, y, info
        y = coerce_numeric(df[target])
        bad = y.isna()
        if bad.all():
            raise PlainMLError(
                f"The target '{target}' isn't numeric, so it can't be used for regression.",
                hint="Leave out --task to let plainml pick classification.",
            )
        if bad.any():
            info.notes.append(f"Dropped {int(bad.sum())} rows whose target isn't a number")
            df, y = df.loc[~bad], y.loc[~bad]
        return df, y, info

    detected_tasks = {t: detect_task(df[t])[0] for t in targets}
    all_binary = all(df[t].nunique() == 2 for t in targets)
    all_numeric = all(coerce_numeric(df[t]).notna().all() for t in targets)
    if task is None and not all_binary and all_numeric:
        task = MULTI_REGRESSION  # e.g. several 0-10 scores: predict the amounts
    if task in (CLASSIFICATION, MULTILABEL) or (
        task is None and all(v == CLASSIFICATION for v in detected_tasks.values())
    ):
        if not all_binary:
            not_binary = [t for t in targets if df[t].nunique() != 2]
            raise PlainMLError(
                "Multi-target classification only supports yes/no (two-value) targets; "
                f"these have more values: {', '.join(not_binary)}.",
                hint="Train one model per target instead.",
            )
        info = TargetInfo(
            names=list(targets),
            task=MULTILABEL,
            reason=f"{len(targets)} yes/no targets, predicted together",
            dropped_rows=dropped,
        )
        columns = {}
        for t in targets:
            negative, positive = sorted(_clean_labels(df[t]).unique(), key=_sort_key)
            info.label_maps[t] = [to_python(negative), to_python(positive)]
            columns[t] = (_clean_labels(df[t]) == positive).astype(int)
        return df, pd.DataFrame(columns, index=df.index), info
    if task in (REGRESSION, MULTI_REGRESSION) or (
        task is None and all(v == REGRESSION for v in detected_tasks.values())
    ):
        Y = pd.DataFrame({t: coerce_numeric(df[t]) for t in targets}, index=df.index)
        keep = Y.notna().all(axis=1)
        info = TargetInfo(
            names=list(targets),
            task=MULTI_REGRESSION,
            reason=f"{len(targets)} numeric targets, predicted together",
            dropped_rows=dropped,
        )
        if not keep.all():
            info.notes.append(f"Dropped {int((~keep).sum())} rows with a non-numeric target")
        return df.loc[keep], Y.loc[keep], info
    raise PlainMLError(
        "Multiple targets must be all yes/no columns (multi-label) or all numbers (multi-output regression).",
        hint="Detected: " + ", ".join(f"{t} → {v}" for t, v in detected_tasks.items()),
    )


def _sort_key(value: Any) -> tuple[int, Any]:
    return (0, value) if isinstance(value, (int, float, np.number, bool)) else (1, str(value))
