# Changelog

## 0.1.0 (unreleased)

First release of plainml, which grew out of [curdrice](https://github.com/pranay-obla/curdrice-v2).

- `plainml train`: detects classification, regression, multi-label and multi-output tasks; profiles the data
  (types, blanks, IDs, leaks, imbalance); compares up to 15 models plus an ensemble with cross-validation
  against a do-nothing baseline; evaluates the winner on a held-out test set; explains it in plain English;
  and saves a run folder with the model, an HTML report and everything needed to reproduce it.
- Data handling: CSV/TSV/Excel/Parquet/JSON/Feather/URLs/databases, numbers stored as text, day-first and
  month-first dates, free text (TF-IDF), with the same preparation applied automatically at prediction time.
- Imbalanced classes: class weights, SMOTE and decision-threshold tuning.
- Commands: `profile`, `clean`, `predict`, `evaluate` (with a drift check), `explain` (importance, effects,
  SHAP, single-row explanations), `report`, `tune` (Optuna or random search), `select`, `cluster`,
  `anomaly`, `forecast`, `serve` (FastAPI), `export` (ONNX), `ui` (Streamlit web app), `runs`, `compare`,
  `models`, `init`.
- Guided mode (`plainml` with no arguments), grouped help with examples, "did you mean" suggestions, friendly
  errors, `--time-budget`, `--quick`, `--config` YAML files, and Ctrl-C to stop early and keep what's trained.
- A Python API (`plainml.train`, `predict`, `explain`, `cluster`, `forecast`, ...).
- Tests, CI on Python 3.10–3.13 across Linux/macOS/Windows (including the oldest supported and pre-release
  library versions), ruff, mypy, pre-commit, docs and example datasets.
