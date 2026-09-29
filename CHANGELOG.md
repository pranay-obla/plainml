# Changelog

## 0.1.0 (unreleased)

First release of plainml, which grew out of [curdrice](https://github.com/pranay-obla/curdrice-v2).

- **Website** (`plainml web`): upload a file, pick any task, watch a live log, read the results and the
  report, and download each output file individually. Includes a runs gallery, a predict page, light and
  dark themes, phone layouts and an optional access token for sharing on a network.
- **More models**: 22 for classification and 30 for regression. The extra tier (AdaBoost, bagging, SGD,
  linear SVMs, Gaussian processes, QDA, Theil-Sen, RANSAC, Poisson/Gamma/Tweedie, kernel ridge, PLS) and a
  stacked ensemble run with `--thorough`. An optional PyTorch MLP comes with the `torch` extra.
- **Feature importance** (`plainml importance`): 19 methods (filters, tree/linear importances corrected
  for column width, permutation, drop-column, SHAP, RFE, forward/backward/exhaustive search, Boruta,
  stability selection), a consensus ranking, a score-by-column-count curve and redundancy groups.
  `select` uses the same engine.
- **Drift** (`plainml drift`), plus a warning in `predict` when new rows look different from the training data.
- **Forecasting**: `--group` (one model per series), `--inputs` / `--future` (planned promotions, prices)
  and `--country` (public holidays).
- **Deploying**: `plainml deploy` (Dockerfile + pinned requirements), `serve --api-key`, model cards
  (`model_card.md` in every run), MLflow logging (`train --mlflow`) and export (`export --format mlflow`).
- **Trust**: calibration checks (reliability curve, Brier score, ECE) and `--calibrate`; `--private` runs
  that keep raw data out of reports and saved files; `--time-budget` now times slow models on a sample first.
- **Big data and housekeeping**: `predict --chunk-size`, `runs --prune --keep N --older-than 30d`.
- `plainml train`: detects classification, regression, multi-label and multi-output tasks; profiles the data
  (types, blanks, IDs, leaks, imbalance); compares up to 15 models plus an ensemble with cross-validation
  against a do-nothing baseline; evaluates the winner on a held-out test set; explains it in plain English;
  and saves a run folder with the model, an HTML report and everything needed to reproduce it.
- Data handling: CSV/TSV/Excel/Parquet/JSON/Feather/URLs/databases, numbers stored as text, day-first and
  month-first dates, free text (TF-IDF), with the same preparation applied automatically at prediction time.
- Imbalanced classes: class weights, SMOTE and decision-threshold tuning.
- Commands: `profile`, `clean`, `predict`, `evaluate` (with a drift check), `explain` (importance, effects,
  SHAP, single-row explanations), `report`, `tune` (Optuna or random search), `select`, `cluster`,
  `anomaly`, `forecast`, `serve` (FastAPI), `export` (ONNX), `runs`, `compare`, `models`, `init`.
- Guided mode (`plainml` with no arguments), grouped help with examples, "did you mean" suggestions, friendly
  errors, `--time-budget`, `--quick`, `--config` YAML files, and Ctrl-C to stop early and keep what's trained.
- A Python API (`plainml.train`, `predict`, `explain`, `cluster`, `forecast`, ...).
- Fixes: forecasts continued from newer history keep the right trend; models saved with other library
  versions explain how to load them; constant or all-unique targets get clear errors; year columns are no
  longer mistaken for row counters; mixed time zones, Windows-encoded CSVs and multi-sheet Excel files load;
  ambiguous run names are reported instead of guessed; yes/no targets with blanks stay yes/no;
  `confidence` is the probability of the predicted answer (it was the highest probability, which
  differs when the decision threshold is tuned).
- Tests, CI on Python 3.10–3.14 across Linux/macOS/Windows (including the oldest supported and pre-release
  library versions), ruff, mypy, pre-commit, docs and example datasets.
