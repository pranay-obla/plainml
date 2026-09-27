"""The plainml web app. Start it with ``plainml ui`` (needs: pip install "plainml[ui]")."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from plainml import __version__
from plainml.errors import PlainMLError
from plainml.io import READERS, load_data
from plainml.runs import MODEL_FILE, REPORT_FILE, list_runs, load_run

RUNS_DIR = Path(os.environ.get("PLAINML_RUNS_DIR", "runs"))
UPLOADS = RUNS_DIR / "uploads"
UPLOAD_TYPES = sorted({suffix.lstrip(".") for suffix in READERS})

st.set_page_config(page_title="plainml", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def _read(path: str, modified: float) -> pd.DataFrame:
    return load_data(path)


def data_picker(key: str) -> tuple[str, pd.DataFrame] | None:
    """Upload a file (kept under runs/uploads so later steps can reuse it) or give a path/URL."""
    uploaded = st.file_uploader("Data file", type=UPLOAD_TYPES, key=f"{key}-upload")
    typed = st.text_input(
        "…or a path / URL", key=f"{key}-path", placeholder="data/sales.csv or https://…"
    )
    path = None
    if uploaded is not None:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        content = uploaded.getvalue()
        digest = hashlib.sha1(content).hexdigest()[:8]
        target = UPLOADS / f"{Path(uploaded.name).stem}-{digest}{Path(uploaded.name).suffix}"
        if not target.exists():
            target.write_bytes(content)
        path = str(target)
    elif typed:
        path = typed.strip()
    if not path:
        return None
    try:
        modified = Path(path).stat().st_mtime if Path(path).exists() else 0.0
        with st.spinner("Reading…"):
            return path, _read(path, modified)
    except PlainMLError as exc:
        st.error(exc.message + (f"\n\n{exc.hint}" if exc.hint else ""))
        return None


def show_report(run_dir: Path, height: int = 1400) -> None:
    report = run_dir / REPORT_FILE
    if report.is_file():
        components.html(report.read_text(encoding="utf-8"), height=height, scrolling=True)


def downloads(run_dir: Path, extra: list[str] | None = None) -> None:
    files = [MODEL_FILE, REPORT_FILE, *(extra or [])]
    columns = st.columns(len(files))
    for column, name in zip(columns, files, strict=False):
        path = run_dir / name
        if path.is_file():
            column.download_button(
                f"Download {name}",
                path.read_bytes(),
                file_name=f"{run_dir.name}_{name}",
                use_container_width=True,
            )


def run_safely(action: Any) -> Any:
    try:
        return action()
    except PlainMLError as exc:
        st.error(exc.message + (f"\n\n{exc.hint}" if exc.hint else ""))
    except Exception as exc:  # show, don't crash the app
        st.exception(exc)
    return None


# --- pages ------------------------------------------------------------------------------------


def page_train() -> None:
    from plainml.metrics import DEFAULT_METRIC, metrics_for
    from plainml.registry import MODELS
    from plainml.tasks import detect_task
    from plainml.training import train

    st.header("Train a model")
    st.caption(
        "Upload a table, choose the column to predict, and plainml compares many models for you."
    )
    picked = data_picker("train")
    if not picked:
        return
    path, df = picked
    st.write(f"**{len(df):,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(30), use_container_width=True)
    target = st.selectbox("Column to predict", list(df.columns), index=len(df.columns) - 1)
    try:
        detected, reason = detect_task(df[target].dropna())
    except PlainMLError as exc:
        st.error(exc.message)
        return
    task = st.radio(
        "Task",
        ["classification", "regression"],
        index=0 if detected == "classification" else 1,
        horizontal=True,
        help=reason,
    )
    st.caption(f"Detected **{detected}**: {reason}")
    with st.expander("Options"):
        quick = st.toggle("Quick mode (fast models only)", value=len(df) > 50_000)
        available = [s.key for s in MODELS if task in s.tasks and s.available and not s.baseline]
        chosen = st.multiselect("Only these models (leave empty for all)", available)
        metric_names = list(
            metrics_for(
                task,
                df[target].dropna() if task == "regression" else None,
                n_classes=int(df[target].nunique()),
            )
        )
        metric = st.selectbox(
            "Rank models by", metric_names, index=metric_names.index(DEFAULT_METRIC[task])
        )
        drop = st.multiselect("Columns to ignore", [c for c in df.columns if c != target])
        cv = st.slider("Cross-validation folds", 2, 10, 5)
        balance = (
            st.selectbox("Imbalanced classes", ["auto", "none", "weights", "smote"])
            if task == "classification"
            else "auto"
        )
        budget = st.text_input("Time budget (e.g. 5m, blank for none)", "")
    if not st.button("Train", type="primary"):
        return
    bar = st.progress(0.0, text="Starting…")

    def progress(fraction: float, message: str) -> None:
        bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    options: dict[str, Any] = dict(
        task=task,
        quick=quick,
        metric=metric,
        drop=drop,
        cv=cv,
        balance=balance,
        out_dir=str(RUNS_DIR),
    )
    if chosen:
        options["models"] = chosen
    if budget.strip():
        options["time_budget"] = budget.strip()
    result = run_safely(lambda: train(path, target, verbose=False, progress=progress, **options))
    bar.empty()
    if result is None:
        return
    st.session_state["last_run"] = str(result.run_dir)
    st.success(f"Best model: **{result.best_model}**, saved to `{result.run_dir}`")
    columns = st.columns(3)
    columns[0].metric(f"{metric} (cross-validated)", f"{result.cv_score:.4g}")
    columns[1].metric(
        f"{metric} on held-out rows", f"{result.holdout_scores.get(metric, float('nan')):.4g}"
    )
    columns[2].metric("Models compared", int((result.leaderboard["status"] == "ok").sum()))
    downloads(result.run_dir, ["holdout_predictions.csv"])
    show_report(result.run_dir)


def page_predict() -> None:
    from plainml.predicting import predict

    st.header("Predict")
    runs = list_runs(RUNS_DIR)
    runs = (
        runs[runs["kind"].isin(["train", "tune", "cluster", "anomaly"])] if not runs.empty else runs
    )
    if runs.empty:
        st.info("No trained models yet. Train one first.")
        return
    labels = {
        r.run: f"{r.run} · {r.best_model} → {r.target or r.task}"
        for r in runs.iloc[::-1].itertuples()
    }
    default = st.session_state.get("last_run")
    options = list(labels)
    index = options.index(Path(default).name) if default and Path(default).name in options else 0
    chosen = st.selectbox("Model", options, index=index, format_func=lambda key: labels[key])
    picked = data_picker("predict")
    if not picked:
        return
    path, _ = picked
    proba = st.toggle("Include class probabilities", value=False)
    if chosen and st.button("Predict", type="primary"):
        frame = run_safely(lambda: predict(RUNS_DIR / chosen, path, proba=proba))
        if frame is not None:
            st.dataframe(frame, use_container_width=True)
            st.download_button(
                "Download predictions (CSV)",
                frame.to_csv(index=False).encode(),
                file_name="predictions.csv",
            )


def page_explore() -> None:
    from plainml.profiling import profile
    from plainml.report import write_profile_report

    st.header("Explore data")
    picked = data_picker("explore")
    if not picked:
        return
    path, df = picked
    target = st.selectbox("Target column (optional)", ["(none)", *df.columns])
    prof = run_safely(lambda: profile(df, None if target == "(none)" else target, verbose=False))
    if prof is None:
        return
    columns = st.columns(4)
    columns[0].metric("Rows", f"{prof.n_rows:,}")
    columns[1].metric("Columns", prof.n_cols)
    columns[2].metric("Duplicate rows", f"{prof.n_duplicates:,}")
    columns[3].metric("Warnings", len(prof.warnings))
    for issue in prof.issues:
        (st.warning if issue.level == "warn" else st.info)(
            issue.message + (f" — {issue.hint}" if issue.hint else "")
        )
    st.dataframe(prof.columns_frame(), use_container_width=True)
    report = write_profile_report(RUNS_DIR / "profiles" / f"{Path(path).stem}.html", prof)
    st.download_button("Download profile (HTML)", report.read_bytes(), file_name=report.name)


def page_runs() -> None:
    st.header("Past runs")
    runs = list_runs(RUNS_DIR)
    if runs.empty:
        st.info(f"No runs in '{RUNS_DIR}' yet.")
        return
    st.dataframe(runs.iloc[::-1], use_container_width=True, hide_index=True)
    chosen = st.selectbox("Open a run", list(runs["run"].iloc[::-1]))
    if chosen:
        run = load_run(RUNS_DIR / chosen)
        downloads(run.path)
        show_report(run.path)


def page_tools() -> None:
    from plainml.anomaly import detect_anomalies
    from plainml.clustering import cluster
    from plainml.forecasting import forecast

    st.header("More tools")
    tool = st.radio(
        "What would you like to do?",
        ["Group similar rows", "Find unusual rows", "Forecast over time"],
        horizontal=True,
    )
    picked = data_picker("tools")
    if not picked:
        return
    path, df = picked
    if tool == "Group similar rows":
        k = st.text_input("Number of groups (a number, a range like 2-8, or auto)", "auto")
        drop = st.multiselect("Columns to ignore", list(df.columns))
        if st.button("Find groups", type="primary"):
            with st.spinner("Clustering…"):
                result = run_safely(
                    lambda: cluster(path, k=k, drop=drop, out_dir=RUNS_DIR, verbose=False)
                )
            if result:
                for sentence in result.descriptions:
                    st.write("• " + sentence)
                downloads(result.run_dir, ["clustered.csv"])
                show_report(result.run_dir)
    elif tool == "Find unusual rows":
        contamination = st.text_input("Share to flag (e.g. 0.02) or auto", "auto")
        label = st.selectbox("Column of known anomalies (optional)", ["(none)", *df.columns])
        if st.button("Find unusual rows", type="primary"):
            with st.spinner("Scoring rows…"):
                result = run_safely(
                    lambda: detect_anomalies(
                        path,
                        contamination=contamination,
                        label=None if label == "(none)" else label,
                        out_dir=RUNS_DIR,
                        verbose=False,
                    )
                )
            if result:
                st.dataframe(result.top, use_container_width=True)
                downloads(result.run_dir, ["scored.csv"])
                show_report(result.run_dir)
    else:
        target = st.selectbox(
            "Value to forecast",
            [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])] or list(df.columns),
        )
        date = st.selectbox("Date column", ["(detect)", *df.columns])
        horizon = st.number_input("Periods ahead (0 = automatic)", min_value=0, value=0)
        if st.button("Forecast", type="primary"):
            with st.spinner("Backtesting models…"):
                result = run_safely(
                    lambda: forecast(
                        path,
                        target,
                        date=None if date == "(detect)" else date,
                        horizon=int(horizon) or None,
                        out_dir=RUNS_DIR,
                        verbose=False,
                    )
                )
            if result:
                for sentence in result.insights:
                    st.write("• " + sentence)
                st.dataframe(result.forecast, use_container_width=True)
                downloads(result.run_dir, ["forecast.csv"])
                show_report(result.run_dir, 1100)


PAGES = {
    "Train a model": page_train,
    "Predict": page_predict,
    "Explore data": page_explore,
    "More tools": page_tools,
    "Past runs": page_runs,
}

with st.sidebar:
    st.title("plainml")
    st.caption(f"From a table to a trained, explained model · v{__version__}")
    choice = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(
        f"Runs are saved in `{RUNS_DIR.resolve()}`. Everything here is also a one-line command: see `plainml --help`."
    )

PAGES[choice]()
