"""User-facing errors and helpers for optional dependencies."""

from __future__ import annotations

import difflib
import importlib
import importlib.util
from collections.abc import Iterable
from types import ModuleType


class PlainMLError(Exception):
    """An error caused by user input, reported without a traceback.

    ``hint`` is an optional next step shown under the message.
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message if not self.hint else f"{self.message}\n{self.hint}"


def did_you_mean(name: str, options: Iterable[str]) -> str:
    """Return a " Did you mean 'x'?" suffix, or an empty string if nothing is close."""
    options = [str(o) for o in options]
    lowered = {o.lower(): o for o in options}
    if name.lower() in lowered:
        return f" Did you mean '{lowered[name.lower()]}'?"
    matches = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
    if not matches:
        matches = difflib.get_close_matches(name.lower(), list(lowered), n=1, cutoff=0.6)
        matches = [lowered[m] for m in matches]
    return f" Did you mean '{matches[0]}'?" if matches else ""


def find_column(name: str, columns: Iterable[str], what: str = "Column") -> str:
    """Resolve a column name, forgiving case and surrounding whitespace.

    Raises a PlainMLError listing the available columns when nothing matches.
    """
    columns = [str(c) for c in columns]
    if name in columns:
        return name
    loose = [c for c in columns if c.strip().lower() == name.strip().lower()]
    if len(loose) == 1:
        return loose[0]
    shown = ", ".join(repr(c) for c in columns[:25])
    more = f" (and {len(columns) - 25} more)" if len(columns) > 25 else ""
    raise PlainMLError(
        f"{what} '{name}' not found.{did_you_mean(name, columns)}",
        hint=f"Available columns: {shown}{more}",
    )


# Maps an importable module to the pip extra that provides it.
EXTRAS = {
    "xgboost": "boost",
    "lightgbm": "boost",
    "catboost": "boost",
    "optuna": "tune",
    "shap": "explain",
    "imblearn": "imbalance",
    "fastapi": "serve",
    "uvicorn": "serve",
    "skl2onnx": "onnx",
    "onnxruntime": "onnx",
    "pyarrow": "formats",
    "sqlalchemy": "formats",
    "polars": "fast",
    "sklearn_extra": "cluster",
    "torch": "torch",
    "holidays": "forecast",
    "mlflow": "mlflow",
    "multipart": "web",
}


def install_hint(module: str) -> str:
    extra = EXTRAS.get(module)
    if extra:
        return f'Install it with: pip install "plainml[{extra}]"'
    return f"Install it with: pip install {module}"


def is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def require(module: str, feature: str) -> ModuleType:
    """Import an optional dependency or raise a PlainMLError explaining how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise PlainMLError(
            f"{feature} needs the optional package '{module}', which isn't installed.",
            hint=install_hint(module),
        ) from exc
