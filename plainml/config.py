"""YAML config files, so a training run can be written down and repeated exactly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plainml.errors import PlainMLError, did_you_mean

CONFIG_KEYS = {
    "data",
    "target",
    "task",
    "metric",
    "models",
    "exclude",
    "quick",
    "thorough",
    "cv",
    "test_size",
    "seed",
    "time_budget",
    "balance",
    "threshold",
    "log_target",
    "calibrate",
    "mlflow",
    "ensemble",
    "refit",
    "save_all",
    "zip",
    "drop",
    "keep",
    "sample",
    "n_jobs",
    "out_dir",
    "name",
    "report",
    "sheet",
    "query",
    "table",
    "engine",
    "private",
}

TEMPLATE = """\
# plainml training config. Run it with:  plainml train --config {name}
# Any option given on the command line overrides the value here.

data: {data}            # file path, URL, or database URL
target: {target}        # column to predict (a list for multi-target)

# task: classification  # classification | regression (auto-detected if left out)
# metric: f1            # ranking metric, e.g. f1, accuracy, roc_auc, rmse, r2, mae
# models: [rf, lightgbm, xgboost]   # only these models (see: plainml models)
# exclude: [svm]        # skip these models
quick: false            # true = only fast models
cv: 5                   # cross-validation folds
test_size: 0.2          # share of rows held out for the final test
seed: 42                # random seed, for reproducible results
# time_budget: 10m      # stop starting new models after this long
balance: auto           # auto | none | weights | smote (classification only)
# drop: [customer_id]   # columns to ignore
# keep: [zip_code]      # columns to use even if they look like IDs
ensemble: true          # also try averaging the top 3 models
report: true            # write report.html
out_dir: runs
"""


def load_config(path: str | Path) -> dict[str, Any]:
    file = Path(path).expanduser()
    if not file.is_file():
        raise PlainMLError(
            f"Config file '{path}' does not exist.", hint="Create one with: plainml init"
        )
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PlainMLError(f"Config file '{path}' isn't valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PlainMLError(f"Config file '{path}' should contain 'key: value' lines.")
    normalized = {str(k).replace("-", "_"): v for k, v in data.items()}
    for key in normalized:
        if key not in CONFIG_KEYS:
            raise PlainMLError(
                f"Unknown setting '{key}' in {path}.{did_you_mean(key, CONFIG_KEYS)}"
            )
    data_path = normalized.get("data")
    if isinstance(data_path, str) and "://" not in data_path and not Path(data_path).is_absolute():
        # relative data paths are relative to the config file, not the current directory
        candidate = (file.parent / data_path).resolve()
        if candidate.exists():
            normalized["data"] = str(candidate)
    return normalized


def write_template(path: str | Path, data: str = "data.csv", target: str = "target") -> Path:
    file = Path(path).expanduser()
    if file.exists():
        raise PlainMLError(f"'{file}' already exists; not overwriting it.")
    file.write_text(TEMPLATE.format(name=file.name, data=data, target=target), encoding="utf-8")
    return file
