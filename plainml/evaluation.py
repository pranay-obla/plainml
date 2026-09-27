"""Scoring a fitted model on labelled data and collecting what the report needs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

from plainml.metrics import Metric, SafeScorer
from plainml.tasks import CLASSIFICATION, MULTI_REGRESSION, MULTILABEL, REGRESSION, to_python

MAX_CURVE_POINTS = 150
MAX_SCATTER_POINTS = 1000


def _downsample_curve(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    if len(x) <= MAX_CURVE_POINTS:
        return [float(v) for v in x], [float(v) for v in y]
    index = np.unique(np.linspace(0, len(x) - 1, MAX_CURVE_POINTS).round().astype(int))
    return [float(v) for v in x[index]], [float(v) for v in y[index]]


def calibration_summary(
    truth: np.ndarray, probability: np.ndarray, bins: int = 10
) -> dict[str, Any]:
    """How honest the probabilities are: a reliability curve, Brier score and ECE.

    ``truth`` is 1 where the event happened, ``probability`` the predicted chance of it.
    Bins hold equal numbers of rows, so sparse ends don't produce wild points.
    ECE (expected calibration error) is the row-weighted gap between what the model said
    and what happened, e.g. 0.04 means probabilities are off by 4 points on average.
    """
    truth = np.asarray(truth, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, bins + 1)))
    which = np.clip(
        np.searchsorted(edges, probability, side="right") - 1, 0, max(len(edges) - 2, 0)
    )
    points, ece = [], 0.0
    for b in np.unique(which):
        mask = which == b
        said, happened = float(probability[mask].mean()), float(truth[mask].mean())
        ece += mask.mean() * abs(said - happened)
        points.append({"predicted": said, "observed": happened, "rows": int(mask.sum())})
    return {
        "points": points,
        "brier": float(np.mean((probability - truth) ** 2)),
        "ece": float(ece),
    }


def calibration_verdict(ece: float) -> str:
    if ece <= 0.03:
        return "well calibrated"
    if ece <= 0.08:
        return "roughly calibrated"
    return "poorly calibrated"


def predict_proba_safe(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)
    except Exception:
        return None
    return proba if isinstance(proba, np.ndarray) else None


def model_classes(model: Any) -> list[Any] | None:
    classes = getattr(model, "classes_", None)
    return [to_python(c) for c in classes] if classes is not None else None


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    task: str,
    metrics: dict[str, Metric],
    *,
    label_maps: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score ``model`` on (X, y). Returns (details for the report, per-row predictions).

    Per-row columns are named after the target (``actual_price``, ``predicted_price``)
    so they can't collide with the dataset's own columns.
    """
    name = str(y.name) if isinstance(y, pd.Series) else ""
    scores = {key: m.display(SafeScorer(m.scorer())(model, X, y)) for key, m in metrics.items()}
    y_pred = model.predict(X)
    result: dict[str, Any] = {"n": len(X), "scores": scores}
    rows = pd.DataFrame(index=X.index)

    if task == CLASSIFICATION:
        classes = model_classes(model) or sorted(pd.unique(np.asarray(y)).tolist(), key=str)
        proba = predict_proba_safe(model, X)
        rows[f"actual_{name}"] = np.asarray(y)
        rows[f"predicted_{name}"] = y_pred
        if proba is not None:
            rows["confidence"] = proba.max(axis=1).round(4)
        matrix = confusion_matrix(y, y_pred, labels=classes)
        result["confusion"] = {"labels": [str(c) for c in classes], "matrix": matrix.tolist()}
        per_class = []
        precision = precision_score(y, y_pred, labels=classes, average=None, zero_division=0)
        recall = recall_score(y, y_pred, labels=classes, average=None, zero_division=0)
        f1 = f1_score(y, y_pred, labels=classes, average=None, zero_division=0)
        for i, label in enumerate(classes):
            per_class.append(
                {
                    "label": str(label),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(matrix[i].sum()),
                }
            )
        result["per_class"] = per_class
        if proba is not None and len(classes) == 2:
            positive = classes[1]
            truth = (np.asarray(y) == positive).astype(int)
            if 0 < truth.sum() < len(truth):
                fpr, tpr, _ = roc_curve(truth, proba[:, 1])
                precision_c, recall_c, _ = precision_recall_curve(truth, proba[:, 1])
                fx, fy = _downsample_curve(fpr, tpr)
                rx, py = _downsample_curve(recall_c[::-1], precision_c[::-1])
                result["roc"] = {
                    "fpr": fx,
                    "tpr": fy,
                    "auc": float(roc_auc_score(truth, proba[:, 1])),
                    "positive": str(positive),
                }
                result["pr"] = {
                    "recall": rx,
                    "precision": py,
                    "base_rate": float(truth.mean()),
                    "positive": str(positive),
                }
        if proba is not None and len(classes) == 2:
            truth = (np.asarray(y) == classes[1]).astype(int)
            result["calibration"] = {
                **calibration_summary(truth, proba[:, 1]),
                "kind": "binary",
                "positive": str(classes[1]),
            }
        elif proba is not None:  # several classes: is the top class's confidence honest?
            correct = np.asarray(classes, dtype=object)[proba.argmax(axis=1)] == np.asarray(
                y, dtype=object
            )
            result["calibration"] = {
                **calibration_summary(correct, proba.max(axis=1)),
                "kind": "top",
            }
        threshold = getattr(model, "best_threshold_", None)
        if threshold is not None:
            result["threshold"] = float(threshold)

    elif task == REGRESSION:
        actual = np.asarray(y, dtype=float)
        predicted = np.asarray(y_pred, dtype=float).ravel()
        rows[f"actual_{name}"] = actual
        rows[f"predicted_{name}"] = predicted
        rows["error"] = predicted - actual
        residuals = predicted - actual
        rng = np.random.default_rng(seed)
        index = np.arange(len(actual))
        if len(index) > MAX_SCATTER_POINTS:
            index = np.sort(rng.choice(index, MAX_SCATTER_POINTS, replace=False))
        result["scatter"] = {
            "actual": actual[index].tolist(),
            "predicted": predicted[index].tolist(),
        }
        result["residuals"] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "p05": float(np.percentile(residuals, 5)),
            "p95": float(np.percentile(residuals, 95)),
            "values": residuals[index].tolist(),
        }

    elif task == MULTILABEL:
        Y = np.asarray(y)
        P = np.asarray(y_pred)
        per_label = []
        for i, name in enumerate(y.columns):
            negative, positive = (label_maps or {}).get(name, [0, 1])
            rows[f"actual_{name}"] = np.where(Y[:, i] == 1, positive, negative)
            rows[f"predicted_{name}"] = np.where(P[:, i] == 1, positive, negative)
            per_label.append(
                {
                    "label": str(name),
                    "precision": float(precision_score(Y[:, i], P[:, i], zero_division=0)),
                    "recall": float(recall_score(Y[:, i], P[:, i], zero_division=0)),
                    "f1": float(f1_score(Y[:, i], P[:, i], zero_division=0)),
                    "support": int(Y[:, i].sum()),
                }
            )
        result["per_label"] = per_label

    elif task == MULTI_REGRESSION:
        Y = np.asarray(y, dtype=float)
        P = np.asarray(y_pred, dtype=float)
        per_target = []
        for i, name in enumerate(y.columns):
            rows[f"actual_{name}"] = Y[:, i]
            rows[f"predicted_{name}"] = P[:, i]
            per_target.append(
                {
                    "target": str(name),
                    "r2": float(r2_score(Y[:, i], P[:, i])),
                    "rmse": float(root_mean_squared_error(Y[:, i], P[:, i])),
                    "mae": float(mean_absolute_error(Y[:, i], P[:, i])),
                }
            )
        result["per_target"] = per_target

    return result, rows
