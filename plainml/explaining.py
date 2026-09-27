"""Explaining models in plain English: which columns matter, and how.

- Permutation importance: shuffle one column and see how much the score drops.
  Works for every model and is measured in the metric's own units.
- Effects: sweep a column across its typical range and watch the prediction move
  ("higher income → higher predicted price").
- SHAP (optional): per-feature contributions, when the ``shap`` package is installed.
- Row explanations: why the model made one particular prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.inspection import permutation_importance

from plainml.console import console, esc, fmt_num, fmt_pct, heading, note
from plainml.errors import PlainMLError, require
from plainml.metrics import Metric, SafeScorer
from plainml.schema import Schema, coerce_numeric
from plainml.tasks import CLASSIFICATION, REGRESSION

MAX_IMPORTANCE_ROWS = 2000
EFFECT_ROWS = 300


def _sample(X: pd.DataFrame, y: Any, n: int, seed: int) -> tuple[pd.DataFrame, Any]:
    if len(X) <= n:
        return X, y
    index = X.sample(n=n, random_state=seed).index
    return X.loc[index], (y.loc[index] if y is not None else None)


def permutation_table(
    model: Any,
    X: pd.DataFrame,
    y: Any,
    metric: Metric,
    *,
    seed: int = 42,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """How much the score drops when each column is shuffled (bigger = more important)."""
    X, y = _sample(X, y, MAX_IMPORTANCE_ROWS, seed)
    if X.shape[1] > 100:
        n_repeats = 3
    result = permutation_importance(
        model,
        X,
        y,
        scoring=SafeScorer(metric.scorer()),
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return _finish_table(table)


def _response(model: Any, X: pd.DataFrame, task: str) -> np.ndarray:
    """Model output averaged over rows: mean prediction or mean class probabilities."""
    if task == CLASSIFICATION and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)).mean(axis=0)
    return np.atleast_1d(np.asarray(model.predict(X), dtype=float).mean(axis=0))


def sensitivity_table(model: Any, X: pd.DataFrame, task: str, *, seed: int = 42) -> pd.DataFrame:
    """Label-free importance: how much predictions change when a column is shuffled."""
    X, _ = _sample(X, None, MAX_IMPORTANCE_ROWS, seed)
    rng = np.random.default_rng(seed)

    def outputs(frame: pd.DataFrame) -> np.ndarray:
        if task == CLASSIFICATION and hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(frame))
        return np.asarray(model.predict(frame), dtype=float).reshape(len(frame), -1)

    base = outputs(X)
    scale = float(np.std(base)) or 1.0
    rows = []
    for column in X.columns:
        shuffled = X.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        change = np.abs(outputs(shuffled) - base).mean() / scale
        rows.append({"feature": column, "importance": float(change), "std": 0.0})
    return _finish_table(pd.DataFrame(rows))


def _finish_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    positive = table["importance"].clip(lower=0)
    total = positive.sum()
    table["share"] = positive / total if total > 0 else 0.0
    return table


def feature_effects(
    model: Any,
    X: pd.DataFrame,
    schema: Schema,
    features: list[str],
    task: str,
    classes: list[Any] | None = None,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sweep each feature across its typical values and record the average prediction."""
    if task not in (CLASSIFICATION, REGRESSION):
        return []
    sample, _ = _sample(X, None, EFFECT_ROWS, seed)
    effects = []
    for column in features:
        try:
            if column in schema.numeric:
                values = coerce_numeric(X[column]).dropna()
                if values.nunique() < 2:
                    continue
                grid = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 9)))
                if len(grid) < 2:
                    continue
                curve = []
                for value in grid:
                    frame = sample.copy()
                    frame[column] = value
                    curve.append(_response(model, frame, task))
                effect: dict[str, Any] | None = _numeric_effect(
                    column, grid, np.array(curve), task, classes
                )
            elif column in schema.categorical:
                top = X[column].dropna().value_counts().head(8).index.tolist()
                if len(top) < 2:
                    continue
                responses: dict[str, np.ndarray] = {}
                for value in top:
                    frame = sample.copy()
                    frame[column] = value
                    responses[str(value)] = _response(model, frame, task)
                effect = _categorical_effect(column, responses, task, classes)
            else:
                continue
        except Exception:  # an odd column shouldn't stop the report
            continue
        if effect:
            effects.append(effect)
    return effects


def _pick_class(matrix: np.ndarray, classes: list[Any] | None) -> tuple[int, str | None]:
    """For probability outputs, follow the class whose probability moves the most."""
    if matrix.ndim == 1 or matrix.shape[1] == 1:
        return 0, None
    if matrix.shape[1] == 2:
        return 1, str(classes[1]) if classes else None
    spread = matrix.max(axis=0) - matrix.min(axis=0)
    index = int(spread.argmax())
    return index, str(classes[index]) if classes else None


def _numeric_effect(
    column: str, grid: np.ndarray, responses: np.ndarray, task: str, classes: list[Any] | None
) -> dict[str, Any]:
    index, label = _pick_class(responses, classes) if task == CLASSIFICATION else (0, None)
    curve = responses[:, index] if responses.ndim == 2 else responses
    correlation = pd.Series(grid).corr(pd.Series(curve), method="spearman")
    direction = "up" if correlation > 0.7 else "down" if correlation < -0.7 else "mixed"
    if np.ptp(curve) < 1e-9:
        direction = "flat"
    return {
        "feature": column,
        "kind": "numeric",
        "grid": [float(v) for v in grid],
        "response": [float(v) for v in curve],
        "direction": direction,
        "class": label,
    }


def _categorical_effect(
    column: str, responses: dict[str, np.ndarray], task: str, classes: list[Any] | None
) -> dict[str, Any] | None:
    matrix = np.array(list(responses.values()))
    index, label = _pick_class(matrix, classes) if task == CLASSIFICATION else (0, None)
    values = {k: float(v[index] if np.ndim(v) else v) for k, v in responses.items()}
    if max(values.values()) - min(values.values()) < 1e-9:
        return None
    return {"feature": column, "kind": "categorical", "values": values, "class": label}


def describe_effect(effect: dict[str, Any], target: str, task: str) -> str:
    column = effect["feature"]
    label = effect.get("class")
    is_prob = task == CLASSIFICATION

    def show(value: float) -> str:
        return fmt_pct(value, 0) if is_prob else fmt_num(value)

    outcome = f"'{label}' more likely" if is_prob else f"higher predicted {target}"
    if effect["kind"] == "numeric":
        low, high = effect["response"][0], effect["response"][-1]
        span = f"({show(low)} → {show(high)} across its typical range)"
        if effect["direction"] == "up":
            return f"Higher {column} → {outcome} {span}."
        if effect["direction"] == "down":
            lower = f"'{label}' less likely" if is_prob else f"lower predicted {target}"
            return f"Higher {column} → {lower} {span}."
        if effect["direction"] == "flat":
            return f"{column} barely changes the prediction on its own."
        return f"{column} matters, but not in a simple 'more is more' way {span}."
    values = effect["values"]
    best = max(values, key=values.get)
    worst = min(values, key=values.get)
    if is_prob:
        return (
            f"{column} = '{best}' makes '{label}' most likely ({show(values[best])}); "
            f"'{worst}' least likely ({show(values[worst])})."
        )
    return (
        f"{column} = '{best}' gives the highest predicted {target} ({show(values[best])}); "
        f"'{worst}' the lowest ({show(values[worst])})."
    )


def summarize_importance(table: pd.DataFrame) -> list[str]:
    """A few plain-English sentences about an importance table."""
    if table.empty:
        return []
    sentences = []
    useful = table[table["importance"] > 1e-6]
    top = useful["feature"].head(3).tolist()
    if not top:
        return ["No single column makes a noticeable difference on its own."]
    if len(top) == 1:
        sentences.append(f"The model relies mostly on {top[0]}.")
    else:
        sentences.append(f"The model relies most on {', '.join(top[:-1])} and {top[-1]}.")
    first_share = float(table["share"].iloc[0])
    if first_share > 0.6 and len(table) > 2:
        sentences.append(
            f"{table['feature'].iloc[0]} alone accounts for {fmt_pct(first_share, 0)} of the total importance. "
            "If that value is only known after the outcome, it's a leak and the scores are too optimistic."
        )
    idle = table[table["importance"] <= 1e-6]["feature"].tolist()
    if idle and len(idle) < len(table):
        shown = ", ".join(idle[:6]) + (f" and {len(idle) - 6} more" if len(idle) > 6 else "")
        sentences.append(
            f"Shuffling these on their own makes no measurable difference: {shown}. "
            "They're either unhelpful or duplicate information other columns already carry."
        )
    return sentences


def typical_values(X: pd.DataFrame, schema: Schema) -> dict[str, Any]:
    """A 'typical' value per column (median or most common), used for row explanations."""
    typical: dict[str, Any] = {}
    for column in schema.features:
        series = X[column].dropna()
        if series.empty:
            continue
        if column in schema.numeric:
            typical[column] = float(coerce_numeric(series).median())
        elif column in schema.text:
            typical[column] = ""
        else:
            mode = series.mode()
            typical[column] = mode.iloc[0].item() if hasattr(mode.iloc[0], "item") else mode.iloc[0]
    return typical


def explain_row(
    model: Any, row: pd.DataFrame, typical: dict[str, Any], task: str, classes: list[Any] | None
) -> tuple[str, pd.DataFrame]:
    """Why one prediction: swap each column to its typical value and see what changes."""
    prediction = model.predict(row)[0]
    base = _response(model, row, task)
    index, _ = _pick_class(base.reshape(1, -1), classes) if task == CLASSIFICATION else (0, None)
    if task == CLASSIFICATION and classes is not None and base.size > 1 and prediction in classes:
        index = classes.index(prediction)
    rows = []
    for column, value in typical.items():
        if column not in row.columns:
            continue
        actual = row[column].iloc[0]
        if pd.isna(actual) or str(actual) == str(value):
            continue
        changed = row.copy()
        changed[column] = value
        delta = float(base[index] - _response(model, changed, task)[index])
        rows.append({"feature": column, "value": actual, "typical": value, "effect": delta})
    table = pd.DataFrame(rows, columns=["feature", "value", "typical", "effect"])
    if not table.empty:
        table = table.reindex(table["effect"].abs().sort_values(ascending=False).index).reset_index(
            drop=True
        )
    if task == CLASSIFICATION and base.size > 1:
        headline = f"Predicted '{prediction}' with probability {fmt_pct(float(base[index]), 0)}."
    else:
        headline = f"Predicted {fmt_num(float(np.ravel(prediction)[0]))}."
    return headline, table


def shap_table(model: Any, X: pd.DataFrame, *, seed: int = 42, max_rows: int = 300) -> pd.DataFrame:
    """Mean |SHAP value| per raw column (encoded features summed back to their column)."""
    shap = require("shap", "SHAP explanations")
    from plainml.preprocessing import describe_features, inner_pipeline, preprocess_step
    from plainml.registry import LabelEncodedClassifier

    pipeline = inner_pipeline(model)
    if pipeline is None:
        raise PlainMLError(
            "SHAP isn't available for ensemble models; use the permutation importance instead."
        )
    estimator = pipeline.named_steps["model"]
    if isinstance(estimator, LabelEncodedClassifier):
        estimator = estimator.estimator_
    if type(estimator).__name__.startswith("MultiOutput"):
        raise PlainMLError("SHAP isn't available for multi-target models.")
    sample, _ = _sample(X, None, max_rows, seed)
    preprocess = preprocess_step(pipeline)
    encoded = np.asarray(preprocess.transform(sample), dtype=float)
    origins = [origin for origin, _ in describe_features(preprocess)]
    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(encoded, check_additivity=False)
    except Exception:
        predict = (
            estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
        )
        background = encoded[: min(100, len(encoded))]
        explainer = shap.Explainer(predict, background, seed=seed)
        values = explainer(encoded[: min(150, len(encoded))]).values
    values = np.abs(
        np.asarray(values if not isinstance(values, list) else np.stack(values, axis=-1))
    )
    if values.ndim == 3:  # (rows, features, classes) or (classes, rows, features)
        values = values.mean(axis=2) if values.shape[1] == len(origins) else values.mean(axis=0)
    per_feature = values.mean(axis=0)
    table = (
        pd.DataFrame({"feature": origins, "importance": per_feature})
        .groupby("feature", sort=False)
        .sum()
    )
    table = table.reset_index()
    table["std"] = 0.0
    return _finish_table(table)


@dataclass
class Explanation:
    importance: pd.DataFrame
    sentences: list[str] = field(default_factory=list)
    effects: list[dict[str, Any]] = field(default_factory=list)
    shap: pd.DataFrame | None = None
    row_headline: str | None = None
    row: pd.DataFrame | None = None
    labelled: bool = True


def explain(
    model: Any = "latest",
    data: Any = None,
    *,
    top: int = 15,
    use_shap: bool = False,
    row: int | None = None,
    output: str | Path | None = None,
    verbose: bool = True,
    out_dir: str | Path = "runs",
    **load_options: Any,
) -> Explanation:
    """Explain a trained model. ``data`` defaults to the run's held-out test rows."""
    from plainml.console import quiet
    from plainml.io import load_data, save_table
    from plainml.metrics import metrics_for, resolve_metric
    from plainml.predicting import load_model
    from plainml.runs import HOLDOUT_FILE

    fitted, meta, model_path = load_model(model, out_dir=out_dir, with_path=True)
    task = meta["task"]
    if task not in (CLASSIFICATION, REGRESSION, "multilabel", "multi_regression"):
        raise PlainMLError(f"explain works for supervised models; this one is a {task} model.")
    schema = Schema.from_dict(meta["schema"])
    targets = meta["targets"]
    if data is None:
        holdout = Path(model_path).parent / HOLDOUT_FILE
        if not holdout.is_file():
            raise PlainMLError(
                "No data to explain with.", hint="Pass a data file: plainml explain MODEL DATA"
            )
        frame = pd.read_csv(holdout)
        rename = {f"actual_{t}": t for t in targets}
        frame = frame.rename(columns=rename)
        if meta.get("private") and not any(c in frame.columns for c in schema.features):
            raise PlainMLError(
                "This model was trained with --private, so its test rows weren't saved.",
                hint="Pass the data to explain with: plainml explain MODEL DATA",
            )
    else:
        frame = load_data(data, **load_options)
    missing = [c for c in schema.features if c not in frame.columns]
    if len(missing) == len(schema.features):
        raise PlainMLError("None of the model's input columns are in this data.")
    X = frame[[c for c in schema.features if c in frame.columns]]
    labelled = all(t in frame.columns for t in targets)

    with quiet(not verbose):
        if labelled:
            y = frame[targets[0]] if len(targets) == 1 else frame[targets]
            if task == "multilabel":
                y = pd.DataFrame(
                    {
                        t: (frame[t].astype(str) == str(meta["label_maps"][t][1])).astype(int)
                        for t in targets
                    }
                )
            metric = resolve_metric(
                meta.get("metric"),
                task,
                metrics_for(task, y, n_classes=len(meta.get("classes") or []) or None),
            )
            importance = permutation_table(fitted, X, y, metric)
        else:
            metric = None
            importance = sensitivity_table(fitted, X, task)
        sentences = summarize_importance(importance)
        top_features = importance["feature"].head(min(top, 6)).tolist()
        effects = feature_effects(fitted, X, schema, top_features, task, meta.get("classes"))
        target_label = ", ".join(targets)
        sentences += [describe_effect(e, target_label, task) for e in effects]
        shap_values = shap_table(fitted, X) if use_shap else None
        headline, row_table = (None, None)
        if row is not None:
            if not 0 <= row < len(X):
                raise PlainMLError(f"Row {row} is out of range (0-{len(X) - 1}).")
            headline, row_table = explain_row(
                fitted, X.iloc[[row]], meta.get("typical", {}), task, meta.get("classes")
            )

    result = Explanation(
        importance=importance,
        sentences=sentences,
        effects=effects,
        shap=shap_values,
        row_headline=headline,
        row=row_table,
        labelled=labelled,
    )
    if output:
        save_table(importance, output)
    if verbose:
        render_explanation(result, metric, top)
    return result


def importance_bar(share: float, width: int = 24) -> str:
    filled = max(0, min(width, round(share * width)))
    return "█" * filled + "[muted]" + "·" * (width - filled) + "[/]"


def render_importance(table: pd.DataFrame, top: int = 15, unit: str | None = None) -> None:
    grid = Table(box=None, show_edge=False, header_style="muted", pad_edge=False)
    grid.add_column("Column", max_width=30, overflow="fold")
    grid.add_column("", no_wrap=True)
    grid.add_column(f"Importance{f' ({unit})' if unit else ''}", justify="right")
    peak = max(float(table["importance"].max()), 1e-12)
    for _, r in table.head(top).iterrows():
        share = max(float(r["importance"]), 0.0) / peak
        grid.add_row(esc(r["feature"]), importance_bar(share), fmt_num(float(r["importance"])))
    console.print(grid)
    if len(table) > top:
        note(f"… and {len(table) - top} more columns")


def _show(value: Any) -> str:
    return (
        fmt_num(float(value))
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        else str(value)
    )


def render_explanation(result: Explanation, metric: Metric | None, top: int) -> None:
    heading("What the model relies on")
    if metric is not None:
        note(f"How much {metric.label} drops when each column is shuffled")
    else:
        note(
            "No target column in this data, so this shows how much predictions change when each column is shuffled"
        )
    render_importance(result.importance, top)
    if result.sentences:
        heading("In plain English")
        for sentence in result.sentences:
            console.print(f"• {esc(sentence)}")
    if result.shap is not None:
        heading("SHAP (average contribution per column)")
        render_importance(result.shap, top)
    if result.row_headline:
        heading("This prediction")
        console.print(esc(result.row_headline))
        if result.row is not None and not result.row.empty:
            grid = Table(box=None, header_style="muted", pad_edge=False)
            for name in ("Column", "Value", "Typical value", "Effect"):
                grid.add_column(name, justify="right" if name == "Effect" else "left")
            for _, r in result.row.head(8).iterrows():
                sign = "[good]+" if r["effect"] > 0 else "[error]"
                grid.add_row(
                    esc(r["feature"]),
                    esc(_show(r["value"])),
                    esc(_show(r["typical"])),
                    f"{sign}{fmt_num(float(r['effect']))}[/]",
                )
            console.print(grid)
            note("Effect: how much this value moves the prediction compared with a typical value")
