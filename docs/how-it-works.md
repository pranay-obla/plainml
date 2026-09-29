# How plainml works

This page explains what happens under the hood, so you can trust (and question) the results.

## 1. Reading and profiling the data

plainml reads CSV, TSV, Excel, Parquet, JSON/JSONL and Feather files, `http(s)` URLs, and databases
(`sqlite:///shop.db --table orders`, or any SQLAlchemy URL with `--query`). It also handles common export
quirks:
- semicolon-separated "CSV" files
- Excel's UTF-8 files with a byte-order mark, and older Windows encodings
- dates with mixed time zones
- workbooks with several sheets: the largest is used and named in a warning (pick another with `--sheet`)

Every column is classified as one of:

| Kind | How it's recognised | What happens to it |
|---|---|---|
| number | numeric values, or text such as `$1,200`, `45%`, `(12)` | converted to numbers, blanks filled with the median (plus a "was blank" flag), scaled |
| category | a limited set of repeated values | one-hot encoded; the 24 most common values are kept and the rest grouped as "other"; unseen values at prediction time are handled |
| date | text or date values that parse as dates (day-first and month-first are both detected) | split into year, month, day, weekday, hour and "is weekend" |
| free text | long, mostly unique text | TF-IDF word features (up to 300 per column) |
| ID | a unique value in (almost) every row, or a counter like 1, 2, 3… | left out (use `--keep` to force it in) |
| constant / empty | one value, or nothing at all | left out |

Before training you see warnings such as:

- **Possible leaks:** a column almost perfectly correlated with the target, one that is identical to it,
  or a category that maps one-to-one onto the classes. Leaks make scores look far better than they will be in real use.
- **Imbalanced classes:** handled automatically (see below).
- Heavily blank columns, very many categories, duplicate rows, and datasets too small to trust.

## 2. The task

- Text, true/false, or whole numbers with at most 10 distinct values → **classification**.
- Other numbers → **regression**. Override with `--task`.
- Several targets at once: all yes/no → **multi-label classification**, all numeric → **multi-output regression**.

## 3. Fair comparison

1. **A held-out test set.** 20% of rows (`--test-size`) are set aside first, stratified so each class keeps its
   share. No model sees them while models are being chosen.
2. **Cross-validation.** Each model is trained and scored 5 times (`--cv`) on different slices of the rest.
   The leaderboard shows the average and the spread (±). All preprocessing is fitted inside each fold, so nothing
   leaks from the scoring rows into training.
3. **A baseline.** A do-nothing model (always the most common class, or always the average) is scored the same
   way. plainml warns if nothing clearly beats it, or if ROC-AUC or R² shows the columns barely predict the target.
4. **Ensembles.** The top three models are averaged and scored like any other model. With `--thorough`,
   they're also stacked (a simple model learns how much to trust each one), and the extra tier of models
   runs too: AdaBoost, bagging, Gaussian processes, robust and generalised linear regressions, and a
   PyTorch network when the `torch` extra is installed. Models that can't suit the data are skipped with a
   reason (Poisson and Gamma regression need non-negative or positive targets, for example), and with
   `--time-budget`, slow models are timed on a sample first and skipped if they wouldn't finish in time.
5. **Overfitting check.** Each model's score on its own training rows is compared with its cross-validated score;
   a big gap is flagged.

### Metrics

| Task | Default ranking metric | Also reported |
|---|---|---|
| Classification | F1, averaged over classes (fair to rare classes) | accuracy, balanced accuracy, precision, recall, ROC-AUC, log loss |
| Regression | RMSE (typical error in the target's units) | MAE, R², MAPE (only when the target has no zeros) |
| Multi-label | F1 over all label decisions | macro F1, exact-match accuracy, Hamming loss |
| Multi-output regression | R² | RMSE, MAE |

Change it with `--metric`, e.g. `--metric roc_auc` or `--metric mae`.

### Imbalanced classes

When the rarest class is under about 20% of rows, plainml weights the rare classes more heavily
(`--balance auto`). You can also choose `--balance smote` (synthetic oversampling, needs the `imbalance`
extra) or `--balance none`. For yes/no problems it then tunes the decision threshold, so the model says
"yes" at the probability that maximises your metric rather than at a fixed 50% (`--threshold`).

## 4. The winner

The best model (ties go to the steadier, then the faster one) is:

1. trained on the training rows and scored on the **held-out test rows**; these are the numbers to quote.
2. explained: **permutation importance** (how much the score drops when a column is shuffled) and **effect
   curves** (the average prediction as one column sweeps across its typical range).
3. checked for **calibration**: do rows given "70%" turn out positive about 70% of the time? The report
   shows a reliability curve with the Brier score and the average gap (ECE). `--calibrate` fixes poorly
   calibrated probabilities (Platt scaling on small data, isotonic regression on larger data).
4. **retrained on all rows** (`--no-refit` to skip) and saved with its whole preprocessing pipeline, plus a
   summary of each column's training distribution for drift checks.

## 5. What's saved

```
runs/20260925-125240_churn/
  model.joblib             the model, preprocessing included; accepts raw rows
  report.html              the report (self-contained, works offline)
  leaderboard.csv          every model's cross-validated scores
  evaluation.json          test-set details used by the report
  importance.csv           which columns mattered
  holdout_predictions.csv  test rows with actual and predicted values
  model_card.md            a one-page summary: data, scores, drivers, caveats, how to use it
  run.json                 settings, library versions, data fingerprint, timings
  config.yaml              rerun with: plainml train --config .../config.yaml
```

With `--private`, the report, run folder and model keep only file names, column names and summary
numbers: no example rows, raw values or data paths.

`plainml predict` checks new data against the columns the model expects. Missing columns are treated as
blank (with a warning, or an error with `--strict`), and extra columns are ignored. It also compares the
new rows with the training data and warns when they look clearly different. `--chunk-size` streams files
too big for memory.

## Drift

`plainml drift` compares new data with a model's training data (or with an older file) column by column.
Numbers and dates are compared with the **population stability index** (PSI) over the training data's
deciles; categories by their shares, with brand-new categories called out. A PSI under 0.1 is stable,
0.1–0.25 a moderate shift, and over 0.25 a major one. Columns are listed by how much the model relies on
them, so a shift in an important column stands out.

## Other tasks

- **Clustering** tries K-Means, agglomerative clustering, Gaussian mixtures and HDBSCAN (plus K-Medoids with the
  `cluster` extra) over a range of group counts, picks the clearest separation (silhouette score), renumbers groups
  largest-first, and describes each one by what sets it apart. New rows can be assigned with `plainml predict`.
- **Anomaly detection** combines Isolation Forest, Local Outlier Factor, One-Class SVM and a robust
  distance-from-median into one 0–1 score (the share of training rows a row is more unusual than). With
  `--label`, each detector is scored against your known anomalies and the best is kept. Flagged rows come with the
  values that make them stand out.
- **Forecasting** puts the series on a regular calendar (filling gaps), builds lag, rolling-average and calendar
  features, and **backtests** each model: it forecasts several past windows using only earlier data. Two simple
  rules ("same as last period", "same as last season") are always included. Tree models learn period-to-period
  changes so they can follow trends. The 80% range comes from the backtest errors.
  - `--group store` forecasts each store separately. The model type is chosen on the largest series using an
    error relative to "same as last period", so big stores don't drown out small ones; then each series gets
    its own fitted model.
  - `--inputs promo,price` adds columns known in advance. Rows after the last known target value (in the same
    file, or a `--future` file) hold their planned values; without them the last values are carried forward,
    with a warning.
  - `--country US` adds public holidays (and the days either side) as features.
- **Feature importance** (`plainml importance`) runs up to 19 methods in four families: filters (mutual
  information, F-test, correlation, chi², variance), model-based (random forest, extra trees, gradient
  boosting, L1, linear coefficients), model-agnostic (permutation, drop-column, SHAP) and searches (RFE,
  forward and backward selection, exhaustive search, Boruta, stability selection). Tree importances are
  corrected against shuffled copies of each column, since trees otherwise favour wide columns such as free
  text. The methods' ranks are combined into a consensus, a curve shows the score with the top 1, 2, 3…
  columns, and redundant groups of columns are pointed out.
- **Tuning** searches each model's key settings (Optuna's TPE when installed, random search otherwise) using the
  same folds as the original run, and re-scores the defaults on those folds so the before/after comparison is fair.

## Limitations

- Everything runs in memory on one machine. For very large files, train with `--sample` (e.g.
  `--sample 200000`) and `--quick`, and predict with `--chunk-size`. Files over 200 MB are read with polars
  when it's installed.
- Free text gets simple word features, not language-model embeddings.
- Forecasting uses backtested machine-learning models and simple baselines, not classical statistical
  models such as ARIMA or exponential smoothing.
- The website runs one job at a time, in order, and keeps its job list in memory: restarting it (or
  reloading the page, in the in-browser version) stops a running job. Finished runs are saved as usual.
- The in-browser version uses one CPU core and takes uploads of up to 200 MB. Very large jobs are faster
  with the command line or the server version.
- ONNX export covers scikit-learn models without free-text columns (not XGBoost, LightGBM, CatBoost or ensembles).
