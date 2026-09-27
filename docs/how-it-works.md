# How plainml works

This page explains what happens under the hood, so you can trust (and question) the results.

## 1. Reading and profiling the data

plainml reads CSV, TSV, Excel, Parquet, JSON/JSONL and Feather files, `http(s)` URLs, and databases
(`sqlite:///shop.db --table orders`, or any SQLAlchemy URL with `--query`). Semicolon-separated "CSV"
files are detected automatically.

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
4. **An ensemble.** The top three models are averaged and scored like any other model.
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
3. **retrained on all rows** (`--no-refit` to skip) and saved with its whole preprocessing pipeline.

## 5. What's saved

```
runs/20260925-125240_churn/
  model.joblib             the model, preprocessing included; accepts raw rows
  report.html              the report (self-contained, works offline)
  leaderboard.csv          every model's cross-validated scores
  evaluation.json          test-set details used by the report
  importance.csv           which columns mattered
  holdout_predictions.csv  test rows with actual and predicted values
  run.json                 settings, library versions, data fingerprint, timings
  config.yaml              rerun with: plainml train --config .../config.yaml
```

`plainml predict` checks new data against the columns the model expects. Missing columns are treated as
blank (with a warning, or an error with `--strict`), and extra columns are ignored.

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
- **Tuning** searches each model's key settings (Optuna's TPE when installed, random search otherwise) using the
  same folds as the original run, and re-scores the defaults on those folds so the before/after comparison is fair.

## Limitations

- Everything runs in memory on one machine. For very large files use `--sample` (e.g. `--sample 200000`) and
  `--quick`; files over 200 MB are read with polars when it's installed.
- Forecasting handles one series at a time and doesn't use other columns as inputs yet.
- Free text gets simple word features, not language-model embeddings.
- ONNX export covers scikit-learn models without free-text columns (not XGBoost, LightGBM, CatBoost or ensembles).
