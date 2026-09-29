# Command reference

Generated from `plainml COMMAND --help` (run `python docs/generate_commands.py` to refresh).

## plainml

```text
Usage: plainml [OPTIONS] [COMMAND] [ARGS]...

  Machine learning in plain English: from a spreadsheet to a trained, explained model.

  Quick start:
    plainml train data.csv --target price       train and compare models
    plainml predict latest new.csv -o out.csv   predict with the best model
    plainml                                     guided mode: answers a few questions

  Run 'plainml COMMAND --help' for a command's options and examples.

Options:
  -V, --version  Show the version and exit.
  --debug        Show full tracebacks on errors.
  -q, --quiet    Only print errors.
  --no-color     Plain text output.
  -h, --help     Show this message and exit.

Start here:
  train    Train and compare models; save the best with a report.
  profile  Summarise a dataset and flag problems, without training.
  clean    Fix common data problems and save a cleaned copy.

Use a trained model:
  predict   Predict on new data with a trained model.
  evaluate  Score a trained model on new labelled data (spot drift).
  explain   Explain what a model relies on, in plain English.
  drift     Check whether new data has drifted from the training data.
  report    Rebuild (and open) a run's HTML report.

Improve a model:
  tune        Search hyperparameters to squeeze out more accuracy.
  importance  Rank columns with up to 19 methods; find the ones that matter.
  select      Keep only the columns that matter; save a smaller dataset.

Other kinds of problem:
  cluster   Group similar rows (no target needed).
  anomaly   Find unusual rows (fraud, errors, outliers).
  forecast  Forecast a value over time (sales, demand, traffic...).

Deploy and share:
  web     Open the plainml website: upload data, run anything, download results.
  serve   Serve a model as a REST API (FastAPI).
  deploy  Package a model as a Docker image that serves it.
  export  Export a model to ONNX or MLflow format.

Housekeeping:
  runs     List past runs, or delete old ones.
  compare  Compare runs side by side.
  models   List the models plainml can train.
  init     Create a starter config file.
```

## Start here

### plainml train

```text
Usage: plainml train [OPTIONS] [DATA]

  Train many models, rank them with cross-validation, and save the best one.

  DATA is a CSV/Excel/Parquet/JSON file, a URL, or a database URL.

Options:
  -t, --target TEXT               Column to predict. Repeat or comma-separate for several.
  --task [auto|classification|regression]
                                  Override the auto-detected task.
  -m, --metric TEXT               Metric to rank models by (default: f1 or rmse). E.g. accuracy,
                                  roc_auc, r2, mae.
  --models TEXT                   Only try these models, e.g. rf,xgboost. See: plainml models
  --exclude TEXT                  Skip these models.
  --quick                         Only fast models, no ensemble.
  --thorough                      Also try the extra models (AdaBoost, Gaussian process, PyTorch...)
                                  and a stacked ensemble.
  --cv INTEGER RANGE              Cross-validation folds.  [default: 5]  [2<=x<=20]
  --test-size FLOAT RANGE         Share of rows held out for the final test.  [default: 0.2]
                                  [0.05<=x<=0.5]
  --seed INTEGER                  Random seed for reproducible results.  [default: 42]
  --time-budget DURATION          Stop starting new models after this long, e.g. 90s, 5m, 1h.
  --balance [auto|none|weights|smote]
                                  Handle imbalanced classes.  [default: auto]
  --threshold [auto|on|off]       Tune the yes/no decision threshold.  [default: auto]
  --log-target                    Model log(target): helps with skewed, positive amounts like
                                  prices.
  --calibrate                     Adjust predicted probabilities so '70% sure' really means right
                                  70% of the time.
  --mlflow                        Also log the run to MLflow (the server in MLFLOW_TRACKING_URI, or
                                  a local store).
  --ensemble / --no-ensemble      Also try averaging the top 3 models.  [default: on]
  --refit / --no-refit            Retrain the winner on all rows before saving.  [default: on]
  --save-all                      Also save every model, not just the best.
  --zip                           Also zip the run folder.
  --drop COLUMNS                  Columns to ignore.
  --keep COLUMNS                  Columns to use even if they look like IDs.
  --sample N                      Train on a random sample: a row count, or a fraction like 0.1.
  --n-jobs INTEGER                CPU cores to use (-1 = all).  [default: -1]
  -o, --out DIR                   Where to save runs.  [default: runs]
  --name TEXT                     Name for the run folder.
  --report / --no-report          Write report.html.  [default: on]
  --private                       Keep raw data out of the report, model and run folder (for sharing
                                  results).
  --open                          Open the report when done.
  -c, --config FILE               YAML file of options (see: plainml init).
  --sheet NAME                    Excel sheet to read (default: the first).
  --query SQL                     SQL query to run (with a database URL).
  --table NAME                    Database table to read (with a database URL).
  --engine [auto|pandas|polars]   Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                      Show this message and exit.

  Examples:
    plainml train churn.csv --target churned
    plainml train houses.xlsx -t price --metric mae --quick
    plainml train data.csv -t label --models rf,lightgbm,xgboost --time-budget 10m
    plainml train data.csv -t label --drop customer_id --balance smote
    plainml train reviews.csv -t is_spam,is_urgent        (multi-label)
    plainml train --config plainml.yaml
```

### plainml profile

```text
Usage: plainml profile [OPTIONS] DATA

  Show each column's type, gaps and values, plus warnings (imbalance, leaks, IDs...).

  Examples:
    plainml profile sales.csv
    plainml profile churn.csv --target churned --html profile.html --open

Options:
  -t, --target TEXT              Also check this target column (imbalance, leakage, task).
  --drop COLUMNS                 Columns to ignore.
  --keep COLUMNS                 Columns to use even if they look like IDs.
  --html FILE                    Also write an HTML version.
  --open                         Open the HTML version.
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml clean

```text
Usage: plainml clean [OPTIONS] DATA

  Standardise blanks, fix numbers stored as text, parse dates, remove duplicates and more.

  Examples:
    plainml clean raw.csv -o clean.csv
    plainml clean raw.xlsx --impute --outliers clip --target price

Options:
  -o, --output FILE               Where to save the cleaned data.  [default: DATA_clean.csv]
  -t, --target TEXT               Target column: rows missing it are dropped; it's never imputed or
                                  encoded.
  --duplicates / --keep-duplicates
                                  Remove duplicate rows.  [default: duplicates]
  --impute / --no-impute          Fill blanks (median / most common value).  [default: no-impute]
  --outliers [none|clip|remove]   Handle extreme numeric values (1.5×IQR rule).  [default: none]
  --encode                        One-hot encode text categories (for tools that need numbers).
  --drop-ids / --keep-ids         Remove ID-like and constant columns.  [default: drop-ids]
  --dry-run                       Show what would change without saving.
  --sheet NAME                    Excel sheet to read (default: the first).
  --query SQL                     SQL query to run (with a database URL).
  --table NAME                    Database table to read (with a database URL).
  --engine [auto|pandas|polars]   Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                      Show this message and exit.
```

## Use a trained model

### plainml predict

```text
Usage: plainml predict [OPTIONS] MODEL [DATA]

  Predict with MODEL: a .joblib file, a run folder, part of a run name, or 'latest'.

  Examples:
    plainml predict latest new_customers.csv -o predictions.csv
    plainml predict runs/20260924-101500_churn new.xlsx --proba
    plainml predict latest --horizon 30          (forecast models)
    plainml predict latest huge.csv -o out.csv --chunk-size 200000

Options:
  -o, --output FILE              Save predictions (.csv, .xlsx, .json, .parquet).
  --proba                        Add a probability column per class.
  --strict                       Fail if any input column is missing (instead of treating it as
                                 blank).
  --horizon INTEGER              Forecast models: how many periods ahead.
  --chunk-size INTEGER RANGE     Stream a big CSV/Parquet file this many rows at a time (needs -o).
                                 [x>=1000]
  --show INTEGER                 Rows to print.  [default: 10]
  --runs-dir TEXT                Where runs are saved.  [default: runs]
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml evaluate

```text
Usage: plainml evaluate [OPTIONS] MODEL DATA

  Check how MODEL does on DATA (which must include the target column).

  Compares against the scores measured at training time, so you can tell when the world has changed
  and the model needs retraining.

  Example:
    plainml evaluate latest march_labelled.csv --report march.html

Options:
  --report FILE                  Also write an HTML report.
  --runs-dir TEXT                [default: runs]
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml explain

```text
Usage: plainml explain [OPTIONS] [MODEL] [DATA]

  Which columns matter, which way they push predictions, and why one row got its prediction.

  Uses the run's held-out test rows unless you pass DATA.

  Examples:
    plainml explain
    plainml explain latest new.csv --row 3
    plainml explain runs/20260924-101500_churn --shap

Options:
  --top INTEGER                  How many columns to show.  [default: 15]
  --shap                         Also compute SHAP values (pip install "plainml[explain]").
  --row INTEGER                  Explain the prediction for this row number (0-based) of DATA.
  -o, --output FILE              Save the importance table.
  --runs-dir TEXT                [default: runs]
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml drift

```text
Usage: plainml drift [OPTIONS] REFERENCE DATA

  Compare DATA with a model's training data (REFERENCE = 'latest', a run, a .joblib) or with an
  older data file. No labels needed.

  Examples:
    plainml drift latest this_month.csv
    plainml drift january.csv june.csv --target churned

Options:
  -t, --target TEXT              Target column to leave out (when REFERENCE is a data file).
  --out, --runs-dir TEXT         Where runs are found (for REFERENCE) and this one is saved.
                                 [default: runs]
  --name TEXT                    Name for the run folder.
  --report / --no-report         Write report.html.  [default: report]
  --open                         Open the report when done.
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml report

```text
Usage: plainml report [OPTIONS] [RUN]

  Regenerate report.html for RUN ('latest', a folder, or part of a run name).

Options:
  --open / --no-open  [default: open]
  --runs-dir TEXT     [default: runs]
  -h, --help          Show this message and exit.
```

## Improve a model

### plainml tune

```text
Usage: plainml tune [OPTIONS] [SOURCE]

  Tune the best models from a run (default: latest), or models on a data file.

  Uses Optuna when installed (pip install "plainml[tune]"), otherwise random search.

  Examples:
    plainml tune
    plainml tune latest --trials 100 --timeout 20m
    plainml tune data.csv -t price --models lightgbm,rf

Options:
  -t, --target TEXT   Target column (when SOURCE is a data file).
  --models TEXT       Models to tune (default: the run's best models).
  --top INTEGER       Tune this many of the run's best models.  [default: 3]
  --trials INTEGER    Settings to try per model.  [default: 30]
  --timeout DURATION  Overall time limit, e.g. 10m.
  -m, --metric TEXT   Metric to optimise (default: the run's).
  --cv INTEGER RANGE  Cross-validation folds.  [2<=x<=20]
  --seed INTEGER      Random seed.
  --open              Open the report when done.
  --runs-dir TEXT     [default: runs]
  -h, --help          Show this message and exit.
```

### plainml importance

```text
Usage: plainml importance [OPTIONS] DATA

  Which columns matter for predicting the target, measured many different ways.

  Filters (mutual information, F-test, correlation...), model-based importances (random forest,
  L1...), model-agnostic ones (permutation, drop-column, SHAP) and searches (RFE, forward/backward
  selection, exhaustive search, Boruta, stability selection) are combined into one consensus
  ranking, with a recommended set and a report.

  Examples:
    plainml importance data.csv -t price
    plainml importance data.csv -t churned --methods all --open
    plainml importance data.csv -t churned --methods boruta,exhaustive -o reduced.csv

Options:
  -t, --target TEXT              Column to predict.  [required]
  --methods TEXT                 Methods (comma-separated), or 'all'. Default: mutual_info, f_test,
                                 correlation, random_forest, l1, permutation, rfe, boruta. Also:
                                 chi2, variance, extra_trees, gradient_boosting, linear,
                                 drop_column, shap, forward, backward, exhaustive, stability.
  -k, --keep-top INTEGER         How many columns to recommend (default: those that clearly help).
  -o, --output FILE              Also save DATA with only the recommended columns (+ target).
  --drop COLUMNS                 Columns to ignore.
  --sample N                     Use a random sample: a row count, or a fraction like 0.1.
  --seed INTEGER                 [default: 42]
  --out TEXT                     Where to save the run.  [default: runs]
  --name TEXT                    Name for the run folder.
  --private                      Keep raw values out of the report.
  --open                         Open the report when done.
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml select

```text
Usage: plainml select [OPTIONS] DATA

  Like 'importance', without saving a run: rank the columns and keep the best ones.

  Examples:
    plainml select data.csv -t price
    plainml select data.csv -t churned -k 10 -o reduced.csv

Options:
  -t, --target TEXT              Column to predict.  [required]
  -k, --keep-top INTEGER         How many columns to keep (default: those that clearly help).
  --methods TEXT                 Methods (comma-separated), or 'all'. Default: mutual_info, f_test,
                                 correlation, random_forest, l1, permutation, rfe, boruta. Also:
                                 chi2, variance, extra_trees, gradient_boosting, linear,
                                 drop_column, shap, forward, backward, exhaustive, stability.
  -o, --output FILE              Save DATA with only the selected columns (+ target).
  --drop COLUMNS                 Columns to ignore.
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

## Other kinds of problem

### plainml cluster

```text
Usage: plainml cluster [OPTIONS] DATA

  Find natural groups (segments) in DATA and describe what makes each one different.

  Examples:
    plainml cluster customers.csv
    plainml cluster customers.csv -k 4 --drop customer_id -o segments.csv

Options:
  -k, --clusters TEXT            Number of groups, a range like 2-8, or auto.  [default: auto]
  --algorithms TEXT              Subset of: kmeans, agglomerative, gmm, hdbscan, kmedoids.
  --drop COLUMNS                 Columns to ignore.
  -o, --output FILE              Save DATA with a 'cluster' column.
  --seed INTEGER                 [default: 42]
  --out TEXT                     Where to save the run.  [default: runs]
  --name TEXT                    Name for the run folder.
  --private                      Keep raw values out of the report and run folder.
  --open                         Open the report when done.
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml anomaly

```text
Usage: plainml anomaly [OPTIONS] DATA

  Score every row by how unusual it is and explain what's odd about the top ones.

  Examples:
    plainml anomaly transactions.csv -o scored.csv
    plainml anomaly transactions.csv --label is_fraud --contamination 0.01

Options:
  --contamination TEXT           Expected share of anomalies, e.g. 0.02, or auto.  [default: auto]
  --label TEXT                   Optional column marking known anomalies (1/0), used to score the
                                 detectors.
  --algorithms TEXT              Subset of: iforest, lof, ocsvm, robust.
  --drop COLUMNS                 Columns to ignore.
  -o, --output FILE              Save DATA with anomaly scores.
  --seed INTEGER                 [default: 42]
  --out TEXT                     Where to save the run.  [default: runs]
  --name TEXT                    Name for the run folder.
  --private                      Keep raw values out of the report and run folder.
  --open                         Open the report when done.
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

### plainml forecast

```text
Usage: plainml forecast [OPTIONS] DATA

  Backtest several forecasting models on DATA's history and forecast the future.

  Rows dated after the last known target value are read as plans for --inputs, so a promo calendar
  can sit in the same file as the history.

  Examples:
    plainml forecast sales.csv -t revenue --horizon 30
    plainml forecast visits.csv -t visitors --date day --freq D -o next_month.csv
    plainml forecast stores.csv -t sales --group store --inputs promo --country US
    plainml forecast sales.csv -t sales --inputs promo,price --future plans.csv

Options:
  -t, --target TEXT              Column to forecast.  [required]
  --date TEXT                    Date/time column (auto-detected if left out).
  --horizon INTEGER              How many periods ahead (default: depends on frequency).
  --freq TEXT                    Frequency: D (daily), W, M, H... (auto-detected if left out).
  --agg [sum|mean|last]          How to combine several rows with the same date.  [default: sum]
  --models TEXT                  Subset of: naive, seasonal_naive, ridge, rf, histgb, lightgbm,
                                 xgboost.
  --group TEXT                   Forecast each value of this column separately (store, product,
                                 region...).
  --inputs TEXT                  Columns known in advance that affect the target (promo, price...).
                                 Comma-separated.
  --future FILE                  File with the inputs' planned future values (date, inputs, and
                                 group if used).
  --country TEXT                 Add public holidays for this country code (US, GB, IN, DE...).
  -o, --output FILE              Save the forecast.
  --seed INTEGER                 [default: 42]
  --out TEXT                     Where to save the run.  [default: runs]
  --name TEXT                    Name for the run folder.
  --open                         Open the report when done.
  --sheet NAME                   Excel sheet to read (default: the first).
  --query SQL                    SQL query to run (with a database URL).
  --table NAME                   Database table to read (with a database URL).
  --engine [auto|pandas|polars]  Reader for big CSV/Parquet files.  [default: auto]
  -h, --help                     Show this message and exit.
```

## Deploy and share

### plainml web

```text
Usage: plainml web [OPTIONS]

  Start a local website for everything plainml does.

  Drag in a file, choose a task (predict a column, forecast, find groups or anomalies, rank columns,
  check drift, profile or clean), watch it run, then read the report and download any result file.

  Examples:
    plainml web
    plainml web --port 9000 --runs-dir projects/churn/runs
    plainml web --host 0.0.0.0 --token change-me      (share on your network)

Options:
  --host TEXT                    Use 0.0.0.0 to let other machines connect (add --token).  [default:
                                 127.0.0.1]
  --port INTEGER                 [default: 8765]
  --runs-dir TEXT                Where runs are saved.  [default: runs]
  --token TEXT                   Require this access token to use the site (or set
                                 PLAINML_WEB_TOKEN).
  --max-upload-mb INTEGER RANGE  Largest upload.  [default: 500; x>=1]
  --no-browser                   Don't open a browser tab.
  -h, --help                     Show this message and exit.
```

### plainml serve

```text
Usage: plainml serve [OPTIONS] [MODEL]

  Start a prediction API for MODEL, with interactive docs at /docs.

  Example:
    plainml serve latest --port 8000
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
         -d '{"rows": [{"age": 42, "plan": "pro"}]}'

Options:
  --host TEXT      Use 0.0.0.0 to accept outside connections.  [default: 127.0.0.1]
  --port INTEGER   [default: 8000]
  --api-key TEXT   Require this key in the X-API-Key header (or set PLAINML_API_KEY).
  --runs-dir TEXT  [default: runs]
  -h, --help       Show this message and exit.
```

### plainml deploy

```text
Usage: plainml deploy [OPTIONS] [MODEL]

  Write a folder with a Dockerfile, pinned requirements and MODEL, ready to build.

  Example:
    plainml deploy latest
    docker build -t churn deploy/20260101-120000_churn
    docker run -p 8000:8000 -e PLAINML_API_KEY=secret churn

Options:
  -o, --output DIRECTORY  Folder to write.  [default: deploy/<run>]
  --port INTEGER          [default: 8000]
  --python TEXT           Python version for the image, e.g. 3.12.
  --runs-dir TEXT         [default: runs]
  -h, --help              Show this message and exit.
```

### plainml export

```text
Usage: plainml export [OPTIONS] [MODEL]

  Convert MODEL to ONNX (checked against the original) or save it as an MLflow model.

  Examples:
    plainml export latest -o churn.onnx
    plainml export latest --format mlflow -o churn_mlflow

Options:
  --format [onnx|mlflow]  [default: onnx]
  -o, --output FILE       Output file.  [default: next to the model]
  --runs-dir TEXT         [default: runs]
  -h, --help              Show this message and exit.
```

## Housekeeping

### plainml runs

```text
Usage: plainml runs [OPTIONS]

  List saved runs, newest last.

  Examples:
    plainml runs
    plainml runs --prune --keep 10
    plainml runs --prune --older-than 30d --kind drift

Options:
  --runs-dir TEXT       [default: runs]
  -n, --limit INTEGER   [default: 20]
  --prune               Delete old runs (asks first). Use with --keep / --older-than.
  --keep INTEGER RANGE  With --prune: keep the newest N runs.  [x>=0]
  --older-than AGE      With --prune: only runs older than this, e.g. 30d, 2w.
  --kind TEXT           With --prune: only this kind of run (train, forecast, importance...).
  -y, --yes             Don't ask for confirmation.
  -h, --help            Show this message and exit.
```

### plainml compare

```text
Usage: plainml compare [OPTIONS] [RUN_REFS]...

  Compare RUNs (default: the last five): data, target, best model and scores.

  Examples:
    plainml compare
    plainml compare runs/*churn*

Options:
  --runs-dir TEXT  [default: runs]
  -h, --help       Show this message and exit.
```

### plainml models

```text
Usage: plainml models [OPTIONS]

  Show every model, which tasks it supports, and whether it's installed.

Options:
  --task [classification|regression]
                                  Only models for this task.
  -h, --help                      Show this message and exit.
```

### plainml init

```text
Usage: plainml init [OPTIONS] [PATH]

  Write a commented YAML config you can edit and run with: plainml train --config PATH

Options:
  --data TEXT        Data file to put in the config.
  -t, --target TEXT  Target column to put in the config.
  -h, --help         Show this message and exit.
```
