<h1 align="center">plainml</h1>

<p align="center">
  <b>Machine learning in plain English.</b><br>
  From a spreadsheet to a trained, explained, deployable model in one command.
</p>

<p align="center">
  <a href="https://pypi.org/project/plainml/"><img src="https://img.shields.io/pypi/v/plainml?color=blue&label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/plainml/"><img src="https://img.shields.io/pypi/pyversions/plainml" alt="Python versions"></a>
  <a href="https://github.com/pranay-obla/plainml/actions/workflows/ci.yml"><img src="https://github.com/pranay-obla/plainml/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

<p align="center"><img src="assets/train.svg" alt="plainml train output: data checks, 10 models compared, the best one explained and saved" width="900"></p>

Point plainml at a CSV (or Excel, Parquet, JSON, a URL, or a database) and name the column you want to
predict. It works out whether that's classification or regression, checks the data for problems, trains
and fairly compares up to 15 models (plus an ensemble), explains the winner in plain English, and saves it with an HTML
report, ready to make predictions, serve as an API or export.

## Why plainml

- **No code, no setup.** Blank cells, prices stored as text (`"$1,200"`), dates, free-text notes and ID
  columns are all handled automatically, and every saved model applies exactly the same preparation to new data.
- **Honest scores.** Models are ranked by cross-validation, the winner is checked on rows it never saw,
  and everything is compared against a do-nothing baseline. You get warnings for likely data leaks,
  imbalanced classes, overfitting, and data that simply can't predict the target.
- **Explained.** *"Higher support_calls → 'yes' more likely (26% → 45%)"*, which columns matter, and
  why any single prediction came out the way it did.
- **Beyond one step.** Profile → clean → train → tune → explain → predict → serve → export, plus
  clustering, anomaly detection and time-series forecasting.
- **For every level.** A guided mode that asks questions, a point-and-click web app, one-line commands,
  and a Python API.

## Install

```bash
pip install plainml
```

Python 3.10 or newer. Optional extras add more power:

| Extra | Adds |
|---|---|
| `boost` | XGBoost, LightGBM and CatBoost models |
| `tune` | Smarter hyperparameter search with Optuna |
| `explain` | SHAP explanations |
| `imbalance` | SMOTE oversampling for rare classes |
| `serve` | `plainml serve` (FastAPI) |
| `onnx` | `plainml export` to ONNX |
| `ui` | `plainml ui` web app (Streamlit) |
| `formats` / `fast` | Parquet and databases / faster loading of big files |
| `all` | Everything above |

```bash
pip install "plainml[all]"
```

## 30-second quickstart

The repository ships with example datasets in [`examples/`](examples):

```bash
plainml train examples/churn.csv --target churned
```

That prints the leaderboard and a summary, and saves a run folder such as `runs/20260925-125240_churn/`
containing the model, `report.html` and more. Then predict on any file with the same columns:

```bash
plainml predict latest examples/churn.csv -o predictions.csv
```

Not sure where to start? Run `plainml` on its own for **guided mode**, or `plainml ui` for the web app.

## The report

Every run writes a self-contained `report.html` (works offline, light and dark mode, readable on a phone)
with the model comparison, test-set results (confusion matrix, ROC and precision-recall curves, or
predicted-vs-actual for numbers), what drives the predictions, the data checks, and how to reproduce the run.

<p align="center"><img src="assets/report.png" alt="plainml HTML report: headline scores, plain-English summary, and a model comparison chart" width="800"></p>

## Commands

| | Command | What it does |
|---|---|---|
| **Start here** | `plainml train DATA -t COLUMN` | Train and compare models; save the best with a report |
| | `plainml profile DATA` | Column types, gaps, and warnings, without training |
| | `plainml clean DATA -o clean.csv` | Fix blanks, text numbers, dates, duplicates, capitalisation… |
| **Use a model** | `plainml predict MODEL DATA` | Predictions on new rows (`MODEL` can be `latest`) |
| | `plainml evaluate MODEL DATA` | Score on new labelled data and spot drift |
| | `plainml explain MODEL [DATA] [--row N]` | What the model relies on, in plain English |
| | `plainml report RUN` | Rebuild and open a run's HTML report |
| **Improve** | `plainml tune [RUN]` | Hyperparameter search on the best models |
| | `plainml select DATA -t COLUMN` | Rank columns; find how many you actually need |
| **Other problems** | `plainml cluster DATA` | Find groups of similar rows, and describe them |
| | `plainml anomaly DATA` | Find unusual rows and say why they're unusual |
| | `plainml forecast DATA -t COLUMN` | Forecast a value over time, with a range |
| **Deploy** | `plainml serve MODEL` | A REST API with interactive docs at `/docs` |
| | `plainml export MODEL` | ONNX, for C#, Java, JavaScript, C++… |
| | `plainml ui` | The point-and-click web app |
| **Housekeeping** | `plainml runs` / `compare` / `models` / `init` | List and compare runs, list models, write a config |

Every command has examples in `plainml COMMAND --help`. See the [command reference](docs/commands.md) for all options.

## Examples

```bash
# Regression, ranked by mean absolute error, fast models only
plainml train examples/house_prices.csv -t price --metric mae --quick

# Choose the models, cap the time, ignore a column, use SMOTE for rare classes
plainml train examples/churn.csv -t churned --models rf,lightgbm,xgboost --time-budget 5m --drop region --balance smote

# Several yes/no targets at once (multi-label)
plainml train data.csv -t is_spam,is_urgent

# Squeeze out more accuracy from the last run
plainml tune latest --trials 50

# Why did row 3 get its prediction?
plainml explain latest new_customers.csv --row 3

# Customer segments, suspicious payments, next month's sales
plainml cluster examples/customers.csv --drop member_id
plainml anomaly examples/transactions.csv --label is_fraud
plainml forecast examples/daily_sales.csv -t units_sold --horizon 30

# Put the model behind an API, or export it
plainml serve latest --port 8000
plainml export latest -o churn.onnx
```

A walkthrough of every example dataset is in [examples/README.md](examples/README.md).

## Python API

Everything the command line does is available from Python and notebooks:

```python
import plainml

result = plainml.train("examples/churn.csv", target="churned", quick=True)
print(result.best_model, result.cv_score)
result.leaderboard                      # a DataFrame (renders as a table in Jupyter)

predictions = plainml.predict(result.run_dir, "new_customers.csv", proba=True)
why = plainml.explain(result.run_dir)
print(why.sentences)

groups = plainml.cluster("examples/customers.csv", drop=["member_id"])
forecast = plainml.forecast("examples/daily_sales.csv", "units_sold", horizon=30)
```

Saved models are ordinary scikit-learn objects that accept raw rows (strings, blanks and all):

```python
import joblib, pandas as pd

model = joblib.load("runs/20260925-125240_churn/model.joblib")
model.predict(pd.read_csv("new_customers.csv"))
```

Only load model files you trust: loading a model file can run code.

See the [Python API guide](docs/python-api.md).

## Reproducible runs

Each run saves its settings, library versions and a fingerprint of the data. Rerun it exactly with:

```bash
plainml train --config runs/20260925-125240_churn/config.yaml
```

or start your own config with `plainml init`.

## How it works

1. **Profile.** Each column is classified as number, category, date, free text, ID or constant. IDs and constants
   are set aside, and you get warnings about leaks, imbalance, heavy blanks and tiny datasets.
2. **Split.** 20% of rows are held out as a final test set that no model sees while models are being chosen.
3. **Compare.** Every model runs inside the same pipeline (fill blanks, scale, one-hot encode, TF-IDF for
   text, date parts) and is scored with 5-fold cross-validation, next to a baseline. The top three
   are also averaged into an ensemble.
4. **Check and explain.** The winner is scored on the held-out rows, and permutation importance and effect
   curves show what drives it.
5. **Save.** The winner is retrained on all rows and saved with its preprocessing, report and metadata.

More detail in [docs/how-it-works.md](docs/how-it-works.md).

## Development

```bash
git clone https://github.com/pranay-obla/plainml && cd plainml
pip install -e ".[all,dev]"
pre-commit install
pytest
```

CI runs the tests on Python 3.10 to 3.13 (Linux, macOS, Windows), with every optional extra, against the
oldest supported library versions, and weekly against the newest releases, so a library update can't
silently break plainml.

## Author

[@pranay-obla](https://github.com/pranay-obla)

plainml grew out of [curdrice](https://github.com/pranay-obla/curdrice-v2), a hackathon project by
[@hrishitb](https://www.github.com/Hrishit-B), [@pranayobla](https://www.github.com/pranay-obla),
[@shriharik](https://www.github.com/RiriSensei) and [@ankitthomas](https://www.github.com/AlmondBox-3996).

## Acknowledgements

Built on [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/),
[NumPy](https://numpy.org/), [Rich](https://github.com/Textualize/rich), [Click](https://click.palletsprojects.com/),
[questionary](https://github.com/tmbo/questionary) and [joblib](https://joblib.readthedocs.io/), with optional
[XGBoost](https://xgboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/),
[Optuna](https://optuna.org/), [SHAP](https://shap.readthedocs.io/), [imbalanced-learn](https://imbalanced-learn.org/),
[FastAPI](https://fastapi.tiangolo.com/), [skl2onnx](https://onnx.ai/sklearn-onnx/) and [Streamlit](https://streamlit.io/).

[MIT licensed](LICENSE).
