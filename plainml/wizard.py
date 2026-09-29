"""Guided mode: running ``plainml`` with no arguments asks a few questions instead."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import questionary
from questionary import Choice

from plainml.console import console, esc, heading, info, note
from plainml.errors import PlainMLError

STYLE = questionary.Style(
    [
        ("qmark", "fg:#2a78d6 bold"),
        ("question", "bold"),
        ("pointer", "fg:#2a78d6 bold"),
        ("highlighted", "fg:#2a78d6 bold"),
        ("answer", "fg:#2a78d6"),
        ("instruction", "fg:#898781"),
    ]
)

ACTIONS = [
    Choice("Train a model to predict a column", "train"),
    Choice("Look at a dataset (profile)", "profile"),
    Choice("Predict with a trained model", "predict"),
    Choice("Clean a dataset", "clean"),
    Choice("Group similar rows (clustering)", "cluster"),
    Choice("Find unusual rows (anomaly detection)", "anomaly"),
    Choice("Forecast a value over time", "forecast"),
    Choice("Find which columns matter (feature importance)", "importance"),
    Choice("Open the website (upload, run, download in the browser)", "web"),
    Choice("Quit", "quit"),
]

DATA_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".txt")


def _ask(question: Any) -> Any:
    answer = question.ask()
    if answer is None:  # Ctrl-C / Esc
        raise KeyboardInterrupt
    return answer


def _nearby_data_files() -> list[str]:
    files = sorted(
        p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() in DATA_SUFFIXES
    )
    return [p.name for p in files][:20]


def ask_data() -> tuple[str, pd.DataFrame]:
    from plainml.io import load_data

    files = _nearby_data_files()
    while True:
        if files:
            choice = _ask(
                questionary.select(
                    "Which data file?",
                    choices=[*files, Choice("Another file…", "__other__")],
                    style=STYLE,
                )
            )
        else:
            choice = "__other__"
        path = (
            choice
            if choice != "__other__"
            else _ask(questionary.path("Path to the data file (or a URL):", style=STYLE))
        )
        try:
            with console.status("Reading…"):
                df = load_data(path)
            info(f"{len(df):,} rows × {df.shape[1]} columns")
            return path, df
        except PlainMLError as exc:
            console.print(f"[error]{esc(exc.message)}[/]")
            if exc.hint:
                note(esc(exc.hint))


def _describe(series: pd.Series) -> str:
    from plainml.schema import infer_column

    column = infer_column(series)
    detail = f"{column.n_unique:,} values"
    if column.missing_pct:
        detail += f", {column.missing_pct:.0%} blank"
    return f"{column.kind}, {detail}"


def ask_target(
    df: pd.DataFrame, question: str = "Which column should the model predict?"
) -> str | None:
    choices = [Choice(f"{name}  ({_describe(df[name])})", name) for name in df.columns]
    return questionary.select(
        question,
        choices=choices,
        style=STYLE,
        use_search_filter=len(choices) > 10,
        use_jk_keys=False,
    ).ask()


def _wizard_train(path: str, df: pd.DataFrame) -> None:
    from plainml.tasks import detect_task
    from plainml.training import train

    target = ask_target(df)
    if target is None:
        raise KeyboardInterrupt
    task, reason = detect_task(df[target])
    other = "regression" if task == "classification" else "classification"
    task = _ask(
        questionary.select(
            f"Looks like {task}: {reason}. Right?",
            choices=[Choice(f"Yes, {task}", task), Choice(f"No, it's {other}", other)],
            style=STYLE,
        )
    )
    speed = _ask(
        questionary.select(
            "How thorough?",
            choices=[
                Choice("Quick: fast models only (seconds to a minute)", "quick"),
                Choice(
                    "Standard: every regular model plus an ensemble (a few minutes)", "standard"
                ),
                Choice("Thorough: extra models and a stacked ensemble (longer)", "thorough"),
            ],
            style=STYLE,
        )
    )
    others = [c for c in df.columns if c != target]
    drop = _ask(
        questionary.checkbox(
            "Any columns to leave out? (space to tick, enter to continue)",
            choices=others,
            style=STYLE,
        )
    )
    result = train(
        path, target, task=task, quick=speed == "quick", thorough=speed == "thorough", drop=drop
    )
    if _ask(questionary.confirm("Open the report in your browser?", default=True, style=STYLE)):
        import click

        click.launch(str(result.report_path))


def _wizard_forecast(df: pd.DataFrame, target: str, date: str | None) -> None:
    from plainml.forecasting import forecast
    from plainml.schema import Kind, infer_column

    kinds = {c: infer_column(df[c]).kind for c in df.columns if c not in (target, date)}
    groups = [c for c, kind in kinds.items() if kind == Kind.CATEGORICAL]
    group = None
    if groups:
        group = _ask(
            questionary.select(
                "Forecast each value of a column separately (a store, a product…)?",
                choices=[Choice("No, one forecast for everything", ""), *groups],
                style=STYLE,
            )
        )
    numbers = [c for c, kind in kinds.items() if kind == Kind.NUMERIC and c != group]
    inputs: list[str] = []
    if numbers:
        inputs = _ask(
            questionary.checkbox(
                "Any columns known in advance that affect it, like a promotion or price? "
                "(space to tick, enter to skip)",
                choices=numbers,
                style=STYLE,
            )
        )
    horizon = _ask(
        questionary.text("How many periods ahead? (blank = automatic)", default="", style=STYLE)
    )
    forecast(
        df,
        target,
        date=date,
        group=group or None,
        inputs=inputs or None,
        horizon=int(horizon) if horizon.strip().isdigit() else None,
    )


def _wizard_predict() -> None:
    from plainml.predicting import predict, render_predictions
    from plainml.runs import list_runs

    runs = list_runs()
    if runs.empty:
        info("No trained models yet; train one first.")
        return
    choices = [
        Choice(f"{r.run}  ({r.best_model} → {r.target})", r.run)
        for r in runs.iloc[::-1].itertuples()
    ]
    run = _ask(questionary.select("Which model?", choices=choices[:20], style=STYLE))
    path, _ = ask_data()
    frame = predict(f"runs/{run}", path)
    render_predictions(frame)
    output = _ask(
        questionary.text(
            "Save predictions to (leave blank to skip):", default="predictions.csv", style=STYLE
        )
    )
    if output:
        from plainml.io import save_table

        save_table(frame, output)
        info(f"Saved {esc(output)}")


def run_wizard(start: str | None = None) -> None:
    heading("plainml guided mode")
    note(
        "Answer a few questions; Ctrl-C to quit. (Tip: every step has a one-line command too, see plainml --help)"
    )
    action = start or _ask(
        questionary.select("What would you like to do?", choices=ACTIONS, style=STYLE)
    )
    if action == "quit":
        return
    if action == "predict":
        _wizard_predict()
        return
    if action == "web":
        from plainml.web.server import run_web

        run_web()
        return
    path, df = ask_data()
    if action == "train":
        _wizard_train(path, df)
    elif action == "profile":
        from plainml.profiling import profile

        profile(df)
    elif action == "clean":
        from plainml.cleaning import clean

        output = _ask(
            questionary.text(
                "Save the cleaned data to:", default=f"{Path(path).stem}_clean.csv", style=STYLE
            )
        )
        clean(df, output=output)
    elif action == "cluster":
        from plainml.clustering import cluster

        cluster(df)
    elif action == "anomaly":
        from plainml.anomaly import detect_anomalies

        detect_anomalies(df)
    elif action == "forecast":
        target = ask_target(df, "Which column should be forecast?")
        if target is None:
            raise KeyboardInterrupt
        date = ask_target(df, "Which column holds the date?")
        _wizard_forecast(df, target, date)
    elif action == "importance":
        from plainml.importance import feature_importance

        target = ask_target(df, "Which column should the others predict?")
        if target is None:
            raise KeyboardInterrupt
        feature_importance(path, target)
