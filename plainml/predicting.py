"""Using trained models: predictions on new data, and scoring on fresh labelled data."""

from __future__ import annotations

import importlib.metadata
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.table import Table

from plainml import __version__
from plainml.console import console, esc, fmt_num, fmt_value, heading, info, note, success, warn
from plainml.errors import PlainMLError, did_you_mean
from plainml.evaluation import evaluate_model, predict_proba_safe, predicted_confidence
from plainml.io import load_data, save_table
from plainml.runs import DEFAULT_RUNS_DIR, RUN_FILE, read_json, resolve_model_path
from plainml.tasks import (
    ANOMALY,
    CLASSIFICATION,
    CLUSTERING,
    FORECAST,
    MULTI_REGRESSION,
    MULTILABEL,
    REGRESSION,
    _clean_labels,
)

# Libraries whose pickled objects can change between versions.
_PICKLE_SENSITIVE = (
    "scikit-learn",
    "numpy",
    "joblib",
    "xgboost",
    "lightgbm",
    "catboost",
    "imbalanced-learn",
    "torch",
)


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _minor(version: str | None) -> tuple[str, ...]:
    return tuple((version or "").split(".")[:2])


def version_mismatches(saved: dict[str, str]) -> list[tuple[str, str, str | None]]:
    """(library, version it was saved with, version installed now) where major.minor differ."""
    return [
        (name, version, _installed(name))
        for name, version in saved.items()
        if name in _PICKLE_SENSITIVE and _minor(version) != _minor(_installed(name))
    ]


def _saved_packages(path: Path) -> dict[str, str]:
    """Library versions from the run's run.json, readable even when the model file isn't."""
    for folder in (path.parent, path.parent.parent):
        run_file = folder / RUN_FILE
        if run_file.is_file():
            try:
                return dict(read_json(run_file).get("environment", {}).get("packages", {}))
            except (OSError, ValueError):
                return {}
    return {}


def _describe_mismatches(mismatches: list[tuple[str, str, str | None]]) -> tuple[str, str]:
    details = ", ".join(
        f"{name} {saved} (you have {now or 'none'})" for name, saved, now in mismatches
    )
    pins = " ".join(f'"{name}=={saved}"' for name, saved, _ in mismatches)
    return details, pins


def load_model(
    ref: Any = "latest", *, out_dir: str | Path = DEFAULT_RUNS_DIR, with_path: bool = False
) -> Any:
    """Load a plainml model. Returns (model, metadata) or (model, metadata, path).

    ``ref`` can be a .joblib file, a run folder, part of a run name, 'latest', or an
    already-loaded model. Only load model files you trust: loading runs code from the file.
    """
    path: Path | None = None
    caught: list[warnings.WarningMessage] = []
    if hasattr(ref, "predict") or hasattr(ref, "forecast"):
        model = ref
    else:
        path = resolve_model_path(ref, out_dir)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = joblib.load(path)
        except ModuleNotFoundError as exc:
            raise PlainMLError(
                f"Loading '{path}' needs the package '{exc.name}', which isn't installed.",
                hint='Install the same extras used for training, e.g. pip install "plainml[boost]".',
            ) from exc
        except Exception as exc:
            mismatches = version_mismatches(_saved_packages(path))
            if mismatches:
                details, pins = _describe_mismatches(mismatches)
                raise PlainMLError(
                    f"Couldn't load '{path}': it was saved with {details}. Model files only "
                    "reliably load with the library versions they were made with.",
                    hint=f"Retrain it, or load it in an environment with: pip install {pins}",
                ) from exc
            raise PlainMLError(f"Couldn't load the model '{path}': {exc}") from exc
    meta = getattr(model, "plainml_meta_", None)
    if meta is None:
        meta = _meta_for_foreign_model(model)
    saved_version = str(meta.get("plainml_version", "0"))
    if saved_version.split(".")[:2] != __version__.split(".")[:2] and saved_version != "0":
        warn(f"This model was made with plainml {saved_version}; you have {__version__}.")
    mismatches = version_mismatches(meta.get("packages") or _saved_packages(path)) if path else []
    inconsistent = any(type(w.message).__name__ == "InconsistentVersionWarning" for w in caught)
    if mismatches or inconsistent:
        details, pins = (
            _describe_mismatches(mismatches) if mismatches else ("a different scikit-learn", "")
        )
        warn(
            esc(
                f"This model was saved with {details}. It loaded, but its predictions may not "
                "match training. Retrain it to be safe"
                + (f", or use: pip install {pins}" if pins else ".")
            )
        )
    return (model, meta, path) if with_path else (model, meta)


def _meta_for_foreign_model(model: Any) -> dict[str, Any]:
    """Minimal metadata so plain scikit-learn models can still be used for predictions."""
    from sklearn.base import is_classifier

    features = getattr(model, "feature_names_in_", None)
    if features is None:
        raise PlainMLError(
            "This model wasn't made by plainml and doesn't record its input columns.",
            hint="Train models with plainml train, or pass a scikit-learn model fitted on a DataFrame.",
        )
    task = CLASSIFICATION if is_classifier(model) else REGRESSION
    return {
        "task": task,
        "targets": ["target"],
        "features": [str(f) for f in features],
        "classes": [c.item() if hasattr(c, "item") else c for c in getattr(model, "classes_", [])]
        or None,
        "schema": {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
            "order": list(features),
        },
    }


def _check_columns(frame: pd.DataFrame, features: list[str], strict: bool) -> list[str]:
    missing = [c for c in features if c not in frame.columns]
    if not missing:
        return []
    if len(missing) == len(features):
        hints = [f"'{c}'{did_you_mean(c, frame.columns)}" for c in features[:5]]
        raise PlainMLError(
            "None of the model's input columns are in this data.",
            hint="The model expects: " + ", ".join(hints),
        )
    listed = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
    if strict:
        raise PlainMLError(
            f"Missing columns: {listed}.", hint="Leave out --strict to fill them in as blanks."
        )
    warn(
        f"{len(missing)} input column(s) missing, treated as blank: {esc(listed)}. Predictions may be less accurate."
    )
    return missing


def predict(
    model: Any = "latest",
    data: Any = None,
    *,
    output: str | Path | None = None,
    proba: bool = False,
    strict: bool = False,
    include_input: bool = True,
    horizon: int | None = None,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    verbose: bool = False,
    check_drift: bool = True,
    chunk_size: int | None = None,
    **load_options: Any,
) -> pd.DataFrame:
    """Predict with a trained model. Returns the input rows plus prediction columns.

    With ``check_drift`` (the default), warns when the new rows look clearly different
    from the training data. ``chunk_size`` streams a big CSV/Parquet file to ``output``
    that many rows at a time, and returns only the first chunk.
    """
    fitted, meta = load_model(model, out_dir=out_dir)
    task = meta["task"]
    if task == FORECAST:
        from plainml.forecasting import forecast_with_model

        frame = load_data(data, **load_options) if data is not None else None
        result = forecast_with_model(fitted, frame, horizon=horizon)
        if output:
            save_table(result, output)
        return result
    if data is None:
        raise PlainMLError("Pass the data to predict on.")
    if chunk_size:
        if not output:
            raise PlainMLError("Predicting in chunks needs an output file (-o predictions.csv).")
        return predict_in_chunks(
            fitted,
            data,
            output,
            meta=meta,
            chunk_size=chunk_size,
            proba=proba,
            strict=strict,
            include_input=include_input,
            check_drift=check_drift,
            verbose=verbose,
        )
    frame = load_data(data, **load_options)
    _check_columns(frame, meta["features"], strict)
    if check_drift:
        from plainml.drift import quick_check

        message = quick_check(meta, frame)
        if message:
            warn(esc(message))
    out = predict_frame(fitted, meta, frame, proba=proba, include_input=include_input)

    if output:
        path = save_table(out, output)
        if verbose:
            success(f"Saved {len(out):,} predictions to {esc(path)}")
    return out


def predict_frame(
    fitted: Any,
    meta: dict[str, Any],
    frame: pd.DataFrame,
    *,
    proba: bool = False,
    include_input: bool = True,
) -> pd.DataFrame:
    """Prediction columns for ``frame`` (plus the input columns with ``include_input``)."""
    task = meta["task"]
    out = frame.copy() if include_input else pd.DataFrame(index=frame.index)
    targets = meta.get("targets") or ["target"]

    if task == CLASSIFICATION:
        target = targets[0]
        predicted = fitted.predict(frame)
        out[f"predicted_{target}"] = predicted
        probabilities = predict_proba_safe(fitted, frame)
        if probabilities is not None:
            classes = getattr(fitted, "classes_", meta.get("classes") or [])
            out["confidence"] = predicted_confidence(probabilities, predicted, classes).round(4)
            if proba:
                for i, label in enumerate(classes):
                    out[f"probability_{label}"] = probabilities[:, i].round(4)
    elif task == REGRESSION:
        out[f"predicted_{targets[0]}"] = np.asarray(fitted.predict(frame), dtype=float).ravel()
    elif task == MULTILABEL:
        predictions = np.asarray(fitted.predict(frame))
        for i, target in enumerate(targets):
            negative, positive = meta["label_maps"][target]
            out[f"predicted_{target}"] = np.where(predictions[:, i] == 1, positive, negative)
    elif task == MULTI_REGRESSION:
        predictions = np.asarray(fitted.predict(frame), dtype=float)
        for i, target in enumerate(targets):
            out[f"predicted_{target}"] = predictions[:, i]
    elif task == CLUSTERING:
        out["cluster"] = fitted.predict(frame)
    elif task == ANOMALY:
        scores = fitted.score(frame)
        out["anomaly_score"] = scores.round(4)
        out["is_anomaly"] = fitted.flag(scores)
    else:  # pragma: no cover
        raise PlainMLError(f"Don't know how to predict for task '{task}'.")
    return out


CHUNKABLE = (".csv", ".tsv", ".txt", ".parquet")


def _chunks(path: Path, chunk_size: int) -> Any:
    """Read a CSV/TSV or Parquet file a slice at a time."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        from plainml.errors import require

        parquet = require("pyarrow.parquet", "Reading Parquet in chunks")
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
        return
    from plainml.io import ENCODINGS, _sniff_delimiter

    separator = "\t" if suffix == ".tsv" else _sniff_delimiter(str(path))
    for encoding in ENCODINGS:  # settle the encoding on the first slice, then stream
        try:
            pd.read_csv(path, sep=separator, encoding=encoding, nrows=5000)
            break
        except UnicodeDecodeError:
            continue
    yield from pd.read_csv(path, sep=separator, encoding=encoding, chunksize=chunk_size)


def predict_in_chunks(
    model: Any,
    data: str | Path,
    output: str | Path,
    *,
    meta: dict[str, Any] | None = None,
    chunk_size: int = 100_000,
    proba: bool = False,
    strict: bool = False,
    include_input: bool = True,
    check_drift: bool = True,
    verbose: bool = True,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
) -> pd.DataFrame:
    """Predict a file too big for memory: read, predict and append ``chunk_size`` rows at a time.

    Writes CSV/TSV (or Parquet, by the output's extension) and returns the first chunk's
    predictions as a preview.
    """
    if meta is None:
        model, meta = load_model(model, out_dir=out_dir)
    source = Path(data)
    if source.suffix.lower() not in CHUNKABLE or not source.is_file():
        raise PlainMLError(
            "Chunked prediction works on local CSV, TSV or Parquet files.",
            hint="Leave out --chunk-size to load the whole file at once.",
        )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    out_format = target.suffix.lower()
    if out_format not in (".csv", ".tsv", ".parquet"):
        raise PlainMLError("Chunked predictions are saved as .csv, .tsv or .parquet.")
    writer = None
    preview: pd.DataFrame | None = None
    total = 0
    offset = 0
    try:
        with console.status("Predicting…") as status:
            for chunk in _chunks(source, chunk_size):
                chunk.columns = [str(c) for c in chunk.columns]
                chunk.index = pd.RangeIndex(offset, offset + len(chunk))
                offset += len(chunk)
                if preview is None:
                    _check_columns(chunk, meta["features"], strict)
                    if check_drift:
                        from plainml.drift import quick_check

                        message = quick_check(meta, chunk)
                        if message:
                            warn(esc(message))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = predict_frame(
                        model, meta, chunk, proba=proba, include_input=include_input
                    )
                if out_format == ".parquet":
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    table = pa.Table.from_pandas(out, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(target, table.schema)
                    writer.write_table(table.cast(writer.schema))
                else:
                    out.to_csv(
                        target,
                        sep="\t" if out_format == ".tsv" else ",",
                        index=False,
                        mode="w" if preview is None else "a",
                        header=preview is None,
                    )
                if preview is None:
                    preview = out
                total += len(out)
                status.update(f"Predicting… {total:,} rows done")
    finally:
        if writer is not None:
            writer.close()
    if preview is None:
        raise PlainMLError(f"'{data}' has no rows.")
    if verbose:
        success(f"Saved {total:,} predictions to {esc(target)} ({chunk_size:,} rows at a time)")
    preview.attrs["rows_written"] = total
    return preview


def _cell(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return esc(str(value).removesuffix(" 00:00:00"))
    return esc(fmt_value(value)) if isinstance(value, (float, np.floating)) else esc(value)


def render_predictions(frame: pd.DataFrame, n: int = 10, saved: bool = False) -> None:
    prediction_columns = [
        c
        for c in frame.columns
        if c.startswith(("predicted_", "probability_"))
        or c in ("confidence", "cluster", "anomaly_score", "is_anomaly")
        or c in ("forecast", "lower_80", "upper_80")
    ]
    context = [c for c in frame.columns if c not in prediction_columns][:4]
    table = Table(header_style="muted", box=None, pad_edge=False)
    for column in [*context, *prediction_columns]:
        style = "bold" if column in prediction_columns else ""
        table.add_column(esc(column), style=style, max_width=24, overflow="ellipsis", no_wrap=True)
    for _, row in frame.head(n).iterrows():
        table.add_row(*[_cell(v) for v in row[[*context, *prediction_columns]]])
    console.print(table)
    total = frame.attrs.get("rows_written", len(frame))
    if total > n:
        note(
            f"… {total - n:,} more rows"
            + ("" if saved else " (save them all with -o predictions.csv)")
        )


def evaluate(
    model: Any = "latest",
    data: Any = None,
    *,
    report: str | Path | None = None,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    verbose: bool = True,
    **load_options: Any,
) -> dict[str, Any]:
    """Score a trained model on new labelled data and compare with its training-time scores."""
    from plainml.console import quiet
    from plainml.metrics import metrics_for

    fitted, meta = load_model(model, out_dir=out_dir)
    task = meta["task"]
    if task not in (CLASSIFICATION, REGRESSION, MULTILABEL, MULTI_REGRESSION):
        raise PlainMLError(f"evaluate works for supervised models; this one is a {task} model.")
    if data is None:
        raise PlainMLError("Pass labelled data to evaluate on.")
    frame = load_data(data, **load_options)
    targets = meta["targets"]
    absent = [t for t in targets if t not in frame.columns]
    if absent:
        raise PlainMLError(
            f"The data needs the target column(s) {', '.join(absent)} to score the model.",
            hint="To predict on unlabelled data use: plainml predict",
        )
    frame = frame.dropna(subset=targets)
    _check_columns(frame, meta["features"], strict=False)
    if task == CLASSIFICATION:
        y: Any = _clean_labels(frame[targets[0]])
    elif task == REGRESSION:
        from plainml.schema import coerce_numeric

        y = coerce_numeric(frame[targets[0]])
    elif task == MULTILABEL:
        y = pd.DataFrame(
            {
                t: (_clean_labels(frame[t]).astype(str) == str(meta["label_maps"][t][1])).astype(
                    int
                )
                for t in targets
            },
            index=frame.index,
        )
    else:
        from plainml.schema import coerce_numeric

        y = pd.DataFrame({t: coerce_numeric(frame[t]) for t in targets}, index=frame.index)
    metrics = metrics_for(task, y, n_classes=len(meta.get("classes") or []) or None)
    with quiet(not verbose):
        details, rows = evaluate_model(
            fitted, frame, y, task, metrics, label_maps=meta.get("label_maps")
        )
    before = meta.get("holdout_scores", {})
    if verbose:
        heading(f"Scores on {len(frame):,} rows")
        table = Table(header_style="muted", box=None, pad_edge=False)
        table.add_column("Metric")
        table.add_column("Now", justify="right", style="bold")
        table.add_column("At training (test set)", justify="right")
        table.add_column("")
        for key, metric in metrics.items():
            now = details["scores"].get(key)
            then = before.get(key)
            change = ""
            if now is not None and then is not None and not np.isnan(now):
                better = (now > then) == metric.greater_is_better
                delta = abs(now - then)
                scale = 1 if metric.bounded else max(abs(then), 1e-9)
                if delta / scale > 0.05:
                    change = "[good]better[/]" if better else "[warn]worse[/]"
            table.add_row(metric.label, fmt_num(now), fmt_num(then), change)
        console.print(table)
        primary = meta.get("metric")
        if primary in before and primary in details["scores"]:
            metric = metrics[primary]
            now, then = details["scores"][primary], before[primary]
            scale = 1 if metric.bounded else max(abs(then), 1e-9)
            worse = (now < then) if metric.greater_is_better else (now > then)
            if worse and abs(now - then) / scale > 0.1:
                warn(
                    f"{metric.label} is noticeably worse than at training time. The data may have changed "
                    "(drift); consider retraining on recent data."
                )
            else:
                info(f"{metric.label} is in line with training time.")
    if report:
        from plainml.report import write_evaluation_report

        path = write_evaluation_report(report, meta, details, metrics)
        if verbose:
            success(f"Report saved to {esc(path)}")
    return {
        "scores": details["scores"],
        "training_scores": before,
        "details": details,
        "predictions": rows,
    }
