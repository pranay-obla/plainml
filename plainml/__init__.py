"""plainml: machine learning in plain English.

From a spreadsheet to a trained, explained, deployable model in one command.

The command line tool is the main interface (run ``plainml --help``), but everything
is also available from Python::

    import plainml

    result = plainml.train("churn.csv", target="churned")
    preds = plainml.predict(result.run_dir, "new_customers.csv")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public API, imported lazily so `plainml --help` doesn't pay for importing scikit-learn.
_API = {
    "train": "plainml.training",
    "TrainResult": "plainml.training",
    "predict": "plainml.predicting",
    "evaluate": "plainml.predicting",
    "load_model": "plainml.predicting",
    "profile": "plainml.profiling",
    "load_data": "plainml.io",
    "load_run": "plainml.runs",
    "list_runs": "plainml.runs",
    "clean": "plainml.cleaning",
    "select_features": "plainml.importance",
    "feature_importance": "plainml.importance",
    "explain": "plainml.explaining",
    "tune": "plainml.tuning",
    "cluster": "plainml.clustering",
    "detect_anomalies": "plainml.anomaly",
    "forecast": "plainml.forecasting",
    "check_drift": "plainml.drift",
    "PlainMLError": "plainml.errors",
}

__all__ = ["__version__", *_API]


def __getattr__(name: str) -> Any:
    if name in _API:
        import importlib

        module = importlib.import_module(_API[name])
        return getattr(module, name)
    raise AttributeError(f"module 'plainml' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from plainml.anomaly import detect_anomalies
    from plainml.cleaning import clean
    from plainml.clustering import cluster
    from plainml.drift import check_drift
    from plainml.errors import PlainMLError
    from plainml.explaining import explain
    from plainml.forecasting import forecast
    from plainml.importance import feature_importance, select_features
    from plainml.io import load_data
    from plainml.predicting import evaluate, load_model, predict
    from plainml.profiling import profile
    from plainml.runs import list_runs, load_run
    from plainml.training import TrainResult, train
    from plainml.tuning import tune
