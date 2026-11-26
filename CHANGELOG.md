# Changelog

## 1.1.0 (November 25, 2026)

- **A new home page.** A personal project page, built on real results:
  - The hero holds plainml's own **3D model comparison**, live, on a cardinal band; drag to turn it.
    Each model's bar rises in turn, then the winner turns gold (still, with reduced motion).
  - **Before and after:** churn.csv's rows next to what plainml says about them.
  - **The held-out test as a machine:** an animated card sorter locks every fifth row card in a drawer
    that only the winning model opens, once.
  - **The jobs:** pick one to turn its real 3D result.
  - **Where it runs:** in the browser, with pip, or as an API.
  - It ends with a short letter from the author.
- **Uploading moved to `/new/`.** The upload box, the example datasets and your recent results are
  there, and every "New" and "Start over" link goes to it.
- **Scores in plain words.** A run page's headline tiles now read "Score on unseen rows", "Score while
  comparing" and "A simple guess", each with a line saying what the number is and whether higher or
  lower is better, instead of "Test F1", "Cross-validated F1" and "Simple baseline". The 3D chart's
  caption and baseline line use the same words.
- **Clearer upload errors.** An empty file or one that doesn't read as a table now says so in plain words,
  with what to do next, and no longer shows the server's path to the file.
- **Calmer app pages.** The glows behind buttons, badges and selected cards are gone, and the footer
  names the author.
- **Fix:** a long column name no longer makes an app page wider than a phone's screen.

## 1.0.1 (November 22, 2026)

- **Reports match the website.** The HTML reports use the same cardinal and gold, on cream or near-black,
  instead of blue, so the report embedded on a run page no longer clashes with the page around it.
- The landing page no longer has a scrolling banner.

## 1.0.0 (November 21, 2026)

- **A new website.** Rebuilt in Next.js and React with a USC-inspired look: cardinal and gold, light and
  dark, on phones too. The same site runs as `plainml web` and fully in the browser.
  - **A landing page** with a 3D hero (a spreadsheet's rows sorting themselves into groups), a ticker,
    scroll-revealed sections, and the example datasets next to the upload box.
  - **3D results**, which you can turn, zoom and hover: every model's score with the baseline, which
    columns matter, customer groups, unusual rows, forecast lanes and drift by column. Each has a table
    view too.
  - Python loads in the background while you read the page, with its progress in the header.
  - Pages have their own addresses (`/runs/`, `/run/?name=…`, `/predict/`…), so the back button and
    bookmarks work.
- **API:** `GET /api/runs/{name}` includes a `visual` field with the data behind the 3D view. Cluster
  runs save a third projected axis, and anomaly runs save where every flagged row (and a sample of the
  rest) sits.
- **For contributors:** the website lives in `frontend/` and needs Node.js 20.9+ to build; packages
  from PyPI include it already. `plainml web` explains how to build it when it's missing.
- **Fix:** the website's live log no longer fills with terminal color codes when `FORCE_COLOR` is set.
- **Fix:** no more pandas deprecation warnings for `dayofweek` and `dayofyear`.
- The tests work with Starlette 1.10's test client (`httpx2`).
- CI and the release workflow install the built wheel in a fresh environment and check the website
  loads from it, and a test keeps `frontend/package.json` on the same version as the package.

## 0.1.2 (November 20, 2026)

- **Example datasets in one click.** The website's home page offers the six example datasets. Clicking
  one loads it and fills in the task and settings that suit it, so new visitors can try plainml without
  their own data.
  - The datasets now ship inside the package: `plainml.datasets.path("churn")`.
  - A new API route, `POST /api/samples/{name}`, starts from one as if it had been uploaded.
- **Fix:** the website no longer shows a stray "null" on results pages without key findings, such as
  anomaly runs.

## 0.1.1 (September 29, 2026)

- **The website runs in the browser.** `plainml web --export DIR` writes the website as static files
  that run plainml in the visitor's browser with Pyodide: no server, and data never leaves their
  computer.
  - Runs are kept in the browser between visits.
  - Every task works, including LightGBM and XGBoost.
  - `vercel.json` publishes it with the documentation (docs under `/docs`), and it also works as a free
    Hugging Face Static Space.
- **Live at [plainml-tpua.vercel.app](https://plainml-tpua.vercel.app)**, with the documentation at `/docs`. The package's
  home page and documentation links now point there.
- **Fix:** progress bars no longer need a background thread, which browsers don't allow.
- **CI** now tests every Python from 3.10 to 3.14 on Linux, macOS and Windows (15 combinations).

## 0.1.0 (September 28, 2026)

First release of plainml, which grew out of [curdrice](https://github.com/pranay-obla/curdrice-v2).

### Training

- `plainml train` detects classification, regression, multi-label and multi-output tasks.
  - It profiles the data first: types, blanks, IDs, leaks, imbalance.
  - Then it compares models with cross-validation and a held-out test set, against a do-nothing baseline.
  - It explains the winner in plain English, and saves a run folder with the model, an HTML report, a
    model card and a config to reproduce it.
- **Models:** 22 for classification and 30 for regression.
  - Every run tries up to 17: linear models, KNN, SVMs, trees, forests, boosting (including XGBoost,
    LightGBM and CatBoost with the `boost` extra), MLPs, LDA, Bayesian ridge and Huber, plus an ensemble
    of the top three.
  - `--thorough` adds QDA, AdaBoost, bagging, SGD, linear SVMs, Gaussian processes, Theil-Sen, RANSAC,
    Poisson, Gamma and Tweedie regression, kernel ridge, PLS and a stacked ensemble.
  - An optional PyTorch network (`torch` extra) joins them.
  - Models that don't suit the data are skipped with a reason. With `--time-budget`, slow models are timed
    on a small sample first.
- **Rare classes:** class weights, SMOTE, and decision-threshold tuning.
- **Calibration:** a reliability curve, Brier score and ECE in every classification report, and
  `--calibrate` to fix probabilities that are off.
- **Other options:** `--quick`, `--thorough`, `--log-target`, `--config` YAML files (`plainml init`), and
  Ctrl-C to stop early and keep the models trained so far.

### Beyond training

- `plainml forecast`: backtested models against simple baselines, with an 80% range.
  - `--group` makes one forecast per series.
  - `--inputs` and `--future` take planned inputs such as promotions.
  - `--country` adds public holidays.
- `plainml cluster`: K-Means, agglomerative, Gaussian mixtures and HDBSCAN, with each group described.
- `plainml anomaly`: four detectors combined into one score, with a reason for every flagged row.
- `plainml importance`: 19 methods and a consensus ranking.
  - Filters, width-corrected tree and linear importances, permutation, drop-column and SHAP.
  - Searches: RFE, forward, backward, exhaustive, Boruta and stability selection.
  - A score-by-column-count curve and redundant groups. `select` uses the same engine.
- `plainml drift`: which columns have shifted since training, and how much the model relies on them.
  `predict` warns automatically.
- `plainml profile`, `clean`, `tune` (Optuna or random search), `explain` (including SHAP and single rows),
  `evaluate` and `report`.

### Using and sharing models

- **The website (`plainml web`):**
  - upload a file, run any task and watch a live log
  - read the findings and the report, and download each result file individually
  - a runs gallery, a predict page, light and dark themes, phone layouts, and an access token for
    sharing it
- **`predict`:** raw rows in, predictions out. `--chunk-size` handles files too big for memory, and
  `confidence` is the probability of the predicted answer.
- **`serve`:** a FastAPI REST API. `--api-key` requires an `X-API-Key` header.
- **`deploy`:** a Docker build folder with pinned requirements, a non-root user and a health check.
- **`export`:** ONNX (checked against the original model) or an MLflow model. `train --mlflow` logs runs.
- **Model cards:** `model_card.md` in every training and forecasting run.
- **`--private`:** keeps raw data values out of reports, run folders and models.
- **Housekeeping:** `runs --prune`, `compare`, `models`.

### Data handling

- **Sources:** CSV, TSV, Excel, Parquet, JSON, Feather, URLs and databases.
- **Messy values:** numbers stored as text, day-first and month-first dates, mixed time zones, free text
  (TF-IDF).
- **Exports:** Excel's UTF-8 files, older Windows encodings, and multi-sheet workbooks (the largest sheet,
  named in a warning).
- **Prediction time:** saved models apply the same preparation to new rows. Models saved with other library
  versions explain how to load them.
- **Clear errors** for impossible targets (one value everywhere, or a different value in every row) and
  ambiguous run names.

### Ways to use it

- A guided mode (`plainml` with no arguments), grouped help with examples, "did you mean" suggestions and
  friendly errors.
- A Python API: `plainml.train`, `predict`, `explain`, `forecast`, `cluster`, `feature_importance`,
  `check_drift` and more.

### Project

- 162 tests, ruff, mypy and pre-commit.
- CI on Python 3.10–3.14 across Linux, macOS and Windows, with every extra (PyTorch included), the oldest
  supported and pre-release library versions, and a docs build.
- Releases publish to PyPI with Trusted Publishing (`RELEASING.md`).
- A documentation site (MkDocs) for Vercel or GitHub Pages.
- A Hugging Face Space setup for the website (`hosting/`), and six example datasets.
