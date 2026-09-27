"""Metrics: which ones apply to each task, how to rank models, and what they mean."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    get_scorer,
    hamming_loss,
    make_scorer,
    precision_score,
    recall_score,
)

from plainml.errors import PlainMLError, did_you_mean
from plainml.tasks import CLASSIFICATION, MULTI_REGRESSION, MULTILABEL, REGRESSION


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    scoring: Any  # an sklearn scoring name or scorer object
    greater_is_better: bool = True
    bounded: bool = True  # scores live in [0, 1]
    explain: str = ""

    def scorer(self) -> Any:
        return get_scorer(self.scoring) if isinstance(self.scoring, str) else self.scoring

    def display(self, raw: float) -> float:
        """sklearn negates error metrics so that bigger is better; undo that for display."""
        return raw if self.greater_is_better else -raw

    def describe(self, target: str) -> str:
        return self.explain.format(target=target)


class SafeScorer:
    """Wrap a scorer so a metric that can't be computed on one fold gives NaN, not a crash.

    (e.g. ROC-AUC on a fold that happens to contain a single class.)
    """

    def __init__(self, scorer: Any):
        self.scorer = scorer

    def __call__(self, estimator: Any, X: Any, y: Any, **kwargs: Any) -> float:
        try:
            return float(self.scorer(estimator, X, y, **kwargs))
        except Exception:
            return float("nan")


def _prf(func: Any, average: str) -> Any:
    # pos_label=None: averaged scores don't have a positive class, and scikit-learn >= 1.8
    # otherwise rejects string labels such as "yes"/"no" (it checks the default pos_label=1).
    return make_scorer(func, average=average, zero_division=0, pos_label=None)


_F1 = "balance of precision and recall{avg} (1.0 is perfect)"


def _classification_metrics(n_classes: int) -> dict[str, Metric]:
    binary = n_classes == 2
    metrics = [
        Metric(
            "f1", "F1", _prf(f1_score, "macro"), explain=_F1.format(avg=", averaged over classes")
        ),
        Metric(
            "accuracy",
            "Accuracy",
            "accuracy",
            explain="share of predictions that are exactly right",
        ),
        Metric(
            "balanced_accuracy",
            "Balanced accuracy",
            "balanced_accuracy",
            explain="accuracy averaged over classes, so rare classes count as much as common ones",
        ),
        Metric(
            "precision",
            "Precision",
            _prf(precision_score, "macro"),
            explain="when the model predicts a class, how often it's right",
        ),
        Metric(
            "recall",
            "Recall",
            _prf(recall_score, "macro"),
            explain="how many of each class's real cases the model finds",
        ),
        Metric(
            "roc_auc",
            "ROC-AUC",
            "roc_auc" if binary else "roc_auc_ovr_weighted",
            explain="how well the model ranks each class above the others (0.5 is random, 1.0 is perfect)",
        ),
        Metric(
            "log_loss",
            "Log loss",
            "neg_log_loss",
            greater_is_better=False,
            bounded=False,
            explain="how confident and correct the predicted probabilities are (lower is better)",
        ),
    ]
    return {m.key: m for m in metrics}


def _regression_metrics(y: pd.Series | pd.DataFrame | None, multi: bool) -> dict[str, Metric]:
    metrics = [
        Metric(
            "rmse",
            "RMSE",
            "neg_root_mean_squared_error",
            greater_is_better=False,
            bounded=False,
            explain="typical size of a prediction error, in the units of {target}; big misses count extra (lower is better)",
        ),
        Metric(
            "mae",
            "MAE",
            "neg_mean_absolute_error",
            greater_is_better=False,
            bounded=False,
            explain="average size of a prediction error, in the units of {target} (lower is better)",
        ),
        Metric(
            "r2",
            "R²",
            "r2",
            bounded=False,
            explain="share of the variation in {target} the model explains (1.0 is perfect, 0 is no better than always guessing the average)",
        ),
    ]
    values = None if y is None else np.abs(np.asarray(y, dtype=float))
    if not multi and values is not None and values.size and values.min() > 1e-9:
        metrics.append(
            Metric(
                "mape",
                "MAPE",
                "neg_mean_absolute_percentage_error",
                greater_is_better=False,
                bounded=False,
                explain="average error as a fraction of the true value (0.1 means 10% off; lower is better)",
            )
        )
    return {m.key: m for m in metrics}


def _multilabel_metrics() -> dict[str, Metric]:
    metrics = [
        Metric(
            "f1",
            "F1 (micro)",
            _prf(f1_score, "micro"),
            explain=_F1.format(avg=" over every label decision"),
        ),
        Metric(
            "f1_macro",
            "F1 (macro)",
            _prf(f1_score, "macro"),
            explain=_F1.format(avg=", averaged over labels"),
        ),
        Metric(
            "accuracy",
            "Exact-match accuracy",
            "accuracy",
            explain="share of rows where every label is right",
        ),
        Metric(
            "hamming",
            "Hamming loss",
            make_scorer(hamming_loss, greater_is_better=False),
            greater_is_better=False,
            explain="share of individual label decisions that are wrong (lower is better)",
        ),
        Metric(
            "precision",
            "Precision (micro)",
            _prf(precision_score, "micro"),
            explain="when the model says a label applies, how often it's right",
        ),
        Metric(
            "recall",
            "Recall (micro)",
            _prf(recall_score, "micro"),
            explain="share of the labels that really apply that the model finds",
        ),
    ]
    return {m.key: m for m in metrics}


def metrics_for(task: str, y: Any = None, n_classes: int | None = None) -> dict[str, Metric]:
    if task == CLASSIFICATION:
        if n_classes is None:
            n_classes = int(pd.Series(np.asarray(y)).nunique()) if y is not None else 2
        return _classification_metrics(n_classes)
    if task == REGRESSION:
        return _regression_metrics(y, multi=False)
    if task == MULTILABEL:
        return _multilabel_metrics()
    if task == MULTI_REGRESSION:
        return _regression_metrics(y, multi=True)
    raise ValueError(task)


DEFAULT_METRIC = {
    CLASSIFICATION: "f1",
    REGRESSION: "rmse",
    MULTILABEL: "f1",
    MULTI_REGRESSION: "r2",
}

_ALIASES = {
    "f1_macro": "f1",
    "f1_micro": "f1",
    "f1score": "f1",
    "auc": "roc_auc",
    "roc": "roc_auc",
    "rocauc": "roc_auc",
    "acc": "accuracy",
    "balanced": "balanced_accuracy",
    "bacc": "balanced_accuracy",
    "logloss": "log_loss",
    "r2_score": "r2",
    "r_squared": "r2",
    "root_mean_squared_error": "rmse",
    "mean_absolute_error": "mae",
    "mean_absolute_percentage_error": "mape",
}


def resolve_metric(name: str | None, task: str, available: dict[str, Metric]) -> Metric:
    if not name:
        return available[DEFAULT_METRIC[task]]
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = key if key in available else _ALIASES.get(key, key)
    if key in available:
        return available[key]
    extra = ""
    if key == "mape":
        extra = " MAPE isn't available because the target contains zeros (it would divide by zero)."
    elif key in ("mse", "msle", "rmsle"):
        extra = " Use rmse instead."
    raise PlainMLError(
        f"Metric '{name}' isn't available for {task}.{did_you_mean(key, available)}{extra}",
        hint="Available: " + ", ".join(available),
    )
