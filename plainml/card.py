"""Model cards: a one-page Markdown summary of a trained model, saved with every run.

A model card says what the model predicts, what it was trained on, how well it did,
what drives it, where it may fall short, and how to use it. It's meant to be read by
someone who wasn't there when the model was trained, and edited (the "Intended use"
section is left for a person to fill in).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plainml.console import fmt_num, fmt_pct
from plainml.runs import EVALUATION_FILE, RUN_FILE, read_json

CARD_FILE = "model_card.md"

_METRIC_LABELS = {
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


def _cell(value: Any) -> str:
    text = fmt_num(value) if isinstance(value, float) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _columns(names: list[str], limit: int = 25) -> str:
    shown = ", ".join(f"`{n}`" for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _usage(run: str, info: dict[str, Any]) -> str:
    forecast = info.get("task") == "forecast"
    predict = (
        f"plainml predict {run} --horizon 30 -o forecast.csv"
        if forecast
        else f"plainml predict {run} new_data.csv -o predictions.csv"
    )
    python = (
        f'forecast = plainml.predict("runs/{run}", horizon=30)'
        if forecast
        else f'predictions = plainml.predict("runs/{run}", "new_data.csv")'
    )
    return (
        "```bash\n"
        f"{predict}\n"
        f"plainml serve {run}           # a REST API with docs at /docs\n"
        f"plainml deploy {run}          # a Docker image for the API\n"
        + (
            ""
            if forecast
            else f"plainml drift {run} new_data.csv   # is new data still like the training data?\n"
        )
        + "```\n\n"
        "```python\n"
        "import plainml\n"
        f"{python}\n"
        "```"
    )


def _environment(info: dict[str, Any]) -> str:
    env = info.get("environment") or {}
    packages = env.get("packages") or {}
    pins = ", ".join(f"{name} {version}" for name, version in packages.items())
    lines = [f"- plainml {info.get('plainml_version', '?')}, Python {env.get('python', '?')}"]
    if pins:
        lines.append(f"- {pins}")
    lines.append(
        "- Load the model with the same major.minor versions of scikit-learn and the model's "
        "library; pickled models don't reliably move between versions."
    )
    return "\n".join(lines)


def _train_card(info: dict[str, Any], evaluation: dict[str, Any]) -> str:
    best = info["best"]
    task = info["task"]
    target = info["target"] if isinstance(info["target"], str) else ", ".join(info["target"])
    data = info.get("data") or {}
    split = info.get("split") or {}
    metric_label = info.get("metric_label", info.get("metric", ""))
    classes = info.get("classes") or []
    details = []
    if info.get("options", {}).get("calibrate"):
        details.append("probabilities calibrated")
    if info.get("threshold") is not None:
        details.append(f"decision threshold {info['threshold']:.2f}")
    if best.get("members"):
        details.append("combines " + ", ".join(best["members"]))
    task_text = info.get("task_label", task)
    if classes:
        shown = ", ".join(str(c) for c in classes[:8]) + (" …" if len(classes) > 8 else "")
        task_text += f" ({len(classes)} classes: {shown})"
    holdout = best.get("holdout") or {}
    headline = holdout.get(info.get("metric"))
    at_a_glance = [
        ["Predicts", f"`{target}`"],
        ["Task", task_text],
        ["Model", best["name"] + (f" ({'; '.join(details)})" if details else "")],
        [
            "Trained on",
            f"{data.get('rows', 0):,} rows × {data.get('columns', 0)} columns from "
            f"`{Path(str(data.get('source', 'a DataFrame'))).name}`",
        ],
        [
            "Test score",
            f"{metric_label} {fmt_num(headline)} on {split.get('test_rows', 0):,} held-out rows "
            f"(cross-validation {fmt_num(best.get('cv_score'))} ± {fmt_num(best.get('cv_std'))})",
        ],
    ]
    if info.get("baseline"):
        at_a_glance.append(
            [
                "Simple baseline",
                f"{metric_label} {fmt_num(info['baseline']['cv_score'])} "
                "(what guessing the most common answer or the average scores)",
            ]
        )
    parts = [
        f"# Model card: {best['name']} predicting {target}",
        f"*Made by plainml on {info['created'][:16].replace('T', ' ')} · run `{info.get('run', '')}`*",
        "## At a glance",
        _table(["", ""], at_a_glance),
        "## Intended use",
        "_Fill this in: who will use the model, for which decision, and what it must not "
        "be used for. Note anyone who could be affected by its mistakes._",
    ]

    schema = info.get("schema") or {}
    used = [
        c for kind in ("numeric", "categorical", "datetime", "text") for c in schema.get(kind, [])
    ]
    inputs = [f"The model reads {len(used)} columns: {_columns(used)}."]
    for kind, label in (
        ("categorical", "Categories"),
        ("datetime", "Dates"),
        ("text", "Free text"),
    ):
        if schema.get(kind):
            inputs.append(f"- {label}: {_columns(schema[kind], 12)}")
    dropped = schema.get("dropped") or {}
    if dropped:
        inputs.append(
            "- Not used: " + "; ".join(f"`{c}` ({why})" for c, why in list(dropped.items())[:12])
        )
    inputs.append("- Missing values are fine: they're filled in the same way as during training.")
    parts += ["## Inputs", "\n".join(inputs)]

    performance = [
        f"Scores on {split.get('test_rows', 0):,} rows the model never saw while training "
        f"(then it was retrained on all {best.get('refit_rows', data.get('rows', 0)):,} rows).",
        "",
        _table(["Metric", "Score"], [[_METRIC_LABELS.get(k, k), v] for k, v in holdout.items()]),
    ]
    per_class = evaluation.get("per_class") or []
    if per_class:
        performance += [
            "",
            "Per class:",
            "",
            _table(
                ["Class", "Precision", "Recall", "F1", "Rows"],
                [
                    [p["label"], p["precision"], p["recall"], p["f1"], p["support"]]
                    for p in per_class
                ],
            ),
        ]
        weakest = min(per_class, key=lambda p: p["f1"])
        if len(per_class) > 1 and weakest["f1"] < 0.6:
            performance.append(
                f"\nThe weakest class is `{weakest['label']}` (F1 {fmt_num(weakest['f1'])}): "
                "treat its predictions with the most care."
            )
    residuals = evaluation.get("residuals")
    if residuals:
        performance.append(
            f"\n9 in 10 predictions land between {fmt_num(residuals['p05'])} and "
            f"+{fmt_num(residuals['p95'])} of the real value."
        )
    calibration = evaluation.get("calibration")
    if calibration:
        from plainml.evaluation import calibration_verdict

        performance.append(
            f"\nProbabilities are {calibration_verdict(calibration['ece'])}: on average they're "
            f"off by {fmt_pct(calibration['ece'], 1)} (Brier score {fmt_num(calibration['brier'])})."
            + (
                " Retrain with `--calibrate` before relying on the probabilities themselves."
                if calibration["ece"] > 0.08 and not info.get("options", {}).get("calibrate")
                else ""
            )
        )
    parts += ["## Performance", "\n".join(performance)]

    sentences = evaluation.get("sentences") or []
    if sentences:
        parts += ["## What drives the predictions", "\n".join(f"- {s}" for s in sentences)]
        parts.append(
            "_These describe patterns the model learned from the data, not causes: changing a "
            "column in real life won't necessarily change the outcome._"
        )

    caveats = []
    profile = info.get("profile") or {}
    for issue in profile.get("issues") or []:
        if issue.get("level") == "warn":
            caveats.append(issue["message"])
    train_score, cv_score = best.get("train_score"), best.get("cv_score")
    if (
        isinstance(train_score, float)
        and isinstance(cv_score, float)
        and info.get("metric_greater_is_better", True)
        and train_score - cv_score > 0.1
    ):
        caveats.append(
            f"It scores much better on its training data ({fmt_num(train_score)}) than in "
            f"cross-validation ({fmt_num(cv_score)}), so it partly memorises the data."
        )
    target_profile = profile.get("target") or {}
    already = any("imbalanced" in c for c in caveats)
    if (target_profile.get("minority_share") or 1) < 0.2 and not already:
        caveats.append(
            f"The classes are imbalanced (the rarest is {fmt_pct(target_profile['minority_share'], 0)} "
            "of rows); accuracy alone would be misleading, so judge it by the scores above."
        )
    if data.get("rows", 0) < 1000:
        caveats.append(
            f"It learned from only {data.get('rows', 0):,} rows; expect scores on new data to vary."
        )
    caveats.append(
        "The scores hold for data like the training data. If the source, time period or "
        "population changes, check with `plainml drift` and retrain when it reports a shift."
    )
    caveats += list(info.get("notes") or [])
    parts += ["## Caveats and limitations", "\n".join(f"- {c}" for c in caveats)]
    parts += ["## How to use it", _usage(info.get("run", "latest"), info)]
    command = info.get("command")
    reproduce = [f"- Command: `{command}`"] if command else []
    reproduce.append(
        f"- Data fingerprint `{data.get('fingerprint', '?')}` (changes if the data changes)"
    )
    parts += ["## Reproduce", "\n".join(reproduce) + "\n" + _environment(info)]
    return "\n\n".join(parts) + "\n"


def _forecast_card(info: dict[str, Any], evaluation: dict[str, Any]) -> str:
    best = info["best"]
    options = info.get("options") or {}
    data = info.get("data") or {}
    unit = evaluation.get("unit", "period")
    rows = [
        ["Forecasts", f"`{info['target']}`, {options.get('horizon')} {unit}s ahead"],
        ["Model", best["name"]],
        [
            "History",
            f"{data.get('periods', '?')} {unit}s from `{Path(str(data.get('source', ''))).name}`",
        ],
        [
            "Backtest error",
            f"MAE {fmt_num(best['score'])} (average miss, in units of {info['target']})",
        ],
    ]
    if options.get("group"):
        rows.append(
            [
                "Series",
                f"{options.get('groups')} values of `{options['group']}`, each forecast separately",
            ]
        )
    if options.get("inputs"):
        rows.append(["Inputs known in advance", _columns(options["inputs"])])
    if options.get("country"):
        rows.append(["Holidays", options["country"]])
    parts = [
        f"# Model card: forecasting {info['target']}",
        f"*Made by plainml on {info['created'][:16].replace('T', ' ')} · run `{info.get('run', '')}`*",
        "## At a glance",
        _table(["", ""], rows),
        "## Intended use",
        "_Fill this in: which plans or decisions rely on this forecast._",
    ]
    insights = evaluation.get("insights") or []
    if insights:
        parts += ["## Patterns in the history", "\n".join(f"- {s}" for s in insights)]
    caveats = [
        "The 80% range comes from backtest errors: roughly 1 in 5 real values should fall outside it.",
        "Forecasts assume the future behaves like the past; one-off events (a new competitor, "
        "a pandemic) aren't foreseen.",
        *evaluation.get("notes", []),
    ]
    if options.get("inputs"):
        caveats.append(
            "Future input values are treated as known. If they're estimates, the forecast inherits their error."
        )
    parts += ["## Caveats and limitations", "\n".join(f"- {c}" for c in caveats)]
    parts += ["## How to use it", _usage(info.get("run", "latest"), info)]
    parts += ["## Reproduce", _environment(info)]
    return "\n\n".join(parts) + "\n"


def write_model_card(run_dir: str | Path) -> Path | None:
    """Write ``model_card.md`` for a training or forecasting run. Returns its path."""
    run_dir = Path(run_dir)
    info = read_json(run_dir / RUN_FILE)
    info.setdefault("run", run_dir.name)
    evaluation_path = run_dir / EVALUATION_FILE
    evaluation = read_json(evaluation_path) if evaluation_path.is_file() else {}
    kind = info.get("kind")
    if kind in ("train", "tune"):
        text = _train_card(info, evaluation)
    elif kind == "forecast":
        text = _forecast_card(info, evaluation)
    else:
        return None
    path = run_dir / CARD_FILE
    path.write_text(text, encoding="utf-8")
    return path
