"""MLflow: log plainml runs to a tracking server, and save models in MLflow's format.

Both need the ``mlflow`` extra (``pip install "plainml[mlflow]"``, which installs the
lightweight ``mlflow-skinny``). The tracking server comes from ``MLFLOW_TRACKING_URI``
as usual; without it, MLflow uses its local default store.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from plainml import __version__
from plainml.console import esc, success
from plainml.errors import PlainMLError, require
from plainml.runs import DEFAULT_RUNS_DIR, MODEL_FILE, RUN_FILE, read_json, resolve_model_path
from plainml.tasks import FORECAST

DEFAULT_EXPERIMENT = "plainml"


def _mlflow() -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep plainml's output clean
    module = require("mlflow", "MLflow tracking")
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    return module


def _flat(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flat(f"{prefix}{key}." if prefix or key else prefix, inner, out)
        return
    if value is None or value == [] or value == {}:
        return
    key = prefix.rstrip(".")
    out[key] = ",".join(map(str, value)) if isinstance(value, list) else value


def _signature(mlflow: Any, model: Any, meta: dict[str, Any]) -> tuple[Any, Any]:
    example = meta.get("example") or {}
    if not example:
        return None, None
    frame = pd.DataFrame([example])
    # integers can't hold blanks, so declare them as floats or blanks later fail the schema
    frame = frame.astype(dict.fromkeys(frame.select_dtypes("integer").columns, float))
    try:
        from mlflow.models import ModelSignature, infer_signature
        from mlflow.types.schema import ColSpec, Schema

        inferred = infer_signature(frame, model.predict(frame))
        # plainml fills missing columns in as blanks, so no input column is required
        optional = Schema([ColSpec(c.type, c.name, required=False) for c in inferred.inputs.inputs])
        return ModelSignature(inputs=optional, outputs=inferred.outputs), frame
    except Exception:
        return None, None


def log_run(
    run_dir: str | Path,
    *,
    experiment: str | None = None,
    log_model: bool = True,
    verbose: bool = True,
) -> str:
    """Log a finished plainml run (params, scores, files, and the model) to MLflow.

    Returns the MLflow run id.
    """
    mlflow = _mlflow()
    run_dir = Path(run_dir)
    info = read_json(run_dir / RUN_FILE)
    mlflow.set_experiment(
        experiment or os.environ.get("PLAINML_MLFLOW_EXPERIMENT") or DEFAULT_EXPERIMENT
    )
    best = info.get("best") or {}
    params: dict[str, Any] = {}
    _flat("", info.get("options") or {}, params)
    _flat("model.", best.get("params") or {}, params)
    params = {k: str(v)[:500] for k, v in params.items() if k not in ("out_dir", "name")}
    metrics: dict[str, float] = {}
    for key in ("cv_score", "cv_std", "train_score", "score"):
        if isinstance(best.get(key), (int, float)):
            metrics[key] = float(best[key])
    for key, value in (best.get("holdout") or {}).items():
        if isinstance(value, (int, float)):
            metrics[f"test_{key}"] = float(value)
    tags = {
        "plainml.kind": str(info.get("kind", "train")),
        "plainml.task": str(info.get("task", "")),
        "plainml.target": str(info.get("target", "")),
        "plainml.model": str(best.get("name", "")),
        "plainml.version": __version__,
        "plainml.run": run_dir.name,
    }
    with mlflow.start_run(run_name=run_dir.name) as active, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name != MODEL_FILE:
                mlflow.log_artifact(str(path))
        if log_model and (run_dir / MODEL_FILE).is_file() and info.get("task") != FORECAST:
            import joblib

            model = joblib.load(run_dir / MODEL_FILE)
            signature, example = _signature(mlflow, model, getattr(model, "plainml_meta_", {}))
            from mlflow import sklearn as mlflow_sklearn

            mlflow_sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=example,
                extra_pip_requirements=[f"plainml=={__version__}"],
                serialization_format="cloudpickle",  # skops can't vouch for plainml's classes
            )
        run_id = str(active.info.run_id)
    if verbose:
        success(
            f"Logged to MLflow (experiment '{esc(mlflow.get_experiment(active.info.experiment_id).name)}', run {run_id[:8]})"
        )
    return run_id


def export_mlflow(
    model: Any = "latest",
    *,
    output: str | Path | None = None,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    verbose: bool = True,
) -> Path:
    """Save a model as an MLflow model folder (loadable with mlflow.pyfunc.load_model)."""
    mlflow = _mlflow()
    import joblib
    from mlflow import sklearn as mlflow_sklearn

    path = resolve_model_path(model, out_dir)
    fitted = joblib.load(path)
    meta = getattr(fitted, "plainml_meta_", {}) or {}
    if meta.get("task") == FORECAST:
        raise PlainMLError(
            "MLflow export covers models that predict from rows, not forecast models.",
            hint="Serve a forecast model with: plainml serve",
        )
    target = Path(output) if output else path.parent / "mlflow_model"
    if target.exists():
        raise PlainMLError(f"'{target}' already exists.", hint="Pass another folder with -o.")
    signature, example = _signature(mlflow, fitted, meta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlflow_sklearn.save_model(
            fitted,
            str(target),
            signature=signature,
            input_example=example,
            extra_pip_requirements=[f"plainml=={__version__}"],
            serialization_format="cloudpickle",  # skops can't vouch for plainml's classes
        )
    if verbose:
        success(f"Saved an MLflow model to {esc(target)}")
        from plainml.console import info

        info("Load it with: mlflow.pyfunc.load_model(" + repr(str(target)) + ")")
    return target
