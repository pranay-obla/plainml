# Python API

Everything is available as `plainml.<function>`. Functions print progress like the CLI; pass
`verbose=False` to keep them quiet. Errors caused by the input (a missing column, an unreadable file) raise
`plainml.PlainMLError`, whose `.message` and `.hint` explain what to do.

## Training

```python
result = plainml.train(
    "examples/churn.csv",  # path, URL, database URL, or a pandas DataFrame
    target="churned",  # or a list for multi-label / multi-output
    metric="roc_auc",  # any CLI option works as a keyword...
    models=["rf", "lightgbm"],
    time_budget="5m",
    drop=["customer_id"],
    verbose=False,
)
```

`train` accepts every `plainml train` option (`task`, `metric`, `models`, `exclude`, `quick`, `cv`,
`test_size`, `seed`, `time_budget`, `balance`, `threshold`, `log_target`, `calibrate`, `ensemble`,
`refit`, `save_all`, `zip`, `drop`, `keep`, `sample`, `n_jobs`, `out_dir`, `name`, `report`, `thorough`,
`private`, `mlflow`), plus
`config="plainml.yaml"` and `progress=callback` (called with a 0–1 fraction and a message).

It returns a `TrainResult`:

| Attribute | |
|---|---|
| `best_model`, `best_key` | the winner's name and registry key |
| `cv_score`, `metric` | its cross-validated score on the ranking metric |
| `holdout_scores` | every metric on the held-out test rows |
| `leaderboard` | DataFrame of all models |
| `importance` | DataFrame of column importances |
| `model` | the fitted model (a scikit-learn estimator) |
| `profile` | data checks (`.issues`, `.columns_frame()`) |
| `run_dir`, `model_path`, `report_path` | where things were saved |
| `predict(data)` | shortcut for `plainml.predict(result.model, data)` |

## Using models

```python
plainml.predict(
    model, data, proba=False, strict=False, output=None, chunk_size=None
)  # -> DataFrame
plainml.check_drift("latest", "this_month.csv")  # -> DriftResult (.verdict, .drifted, .sentences)
plainml.evaluate(model, data, report=None)  # -> dict of scores
plainml.explain(model, data=None, row=None, use_shap=False)  # -> Explanation
model, meta = plainml.load_model("latest")
```

`model` can be a `.joblib` path, a run folder, part of a run name, `"latest"`, or a loaded model.

`Explanation` has `.importance` (DataFrame), `.sentences` (plain English), `.effects`, `.shap` and, with
`row=`, `.row_headline` and `.row`.

## Data tools

```python
plainml.load_data("sales.xlsx", sheet="2026")
plainml.profile("churn.csv", target="churned")  # -> Profile
plainml.clean("raw.csv", output="clean.csv", impute=True, outliers="clip")  # -> DataFrame
plainml.select_features("data.csv", "price", k=10)  # -> ranking DataFrame
ranked = plainml.feature_importance("data.csv", "price", methods=["rf", "rfe", "boruta", "shap"])
(
    ranked.table,
    ranked.selected,
    ranked.curve,
)  # consensus ranking, columns worth keeping, score by column count
```

## Other tasks

```python
plainml.tune("latest", trials=50, timeout="20m")  # -> TrainResult
plainml.cluster("customers.csv", k="2-8")  # -> ClusterResult (.labels, .descriptions)
plainml.detect_anomalies(
    "payments.csv", contamination=0.01
)  # -> AnomalyResult (.scores, .flags, .top)
plainml.forecast("sales.csv", "revenue", horizon=30)  # -> ForecastResult (.forecast, .insights)
plainml.forecast(  # one forecast per store, with planned promotions and public holidays
    "examples/store_sales.csv", "sales", group="store", inputs=["promo"], country="US", horizon=14
)
```

## Runs

```python
plainml.list_runs()  # DataFrame of past runs
run = plainml.load_run("latest")
run.info, run.leaderboard, run.importance, run.load_model()
```

## Serving from your own app

```python
from plainml.serve import create_app  # needs the "serve" extra

app = create_app("runs/20260925-125240_churn", api_key="change-me")  # a FastAPI app
```

The website is a FastAPI app too:

```python
from plainml.web.server import create_app  # needs the "web" extra

app = create_app("runs", token="change-me")
```

Or write the in-browser version, static files that need no server:

```python
from plainml.web.static_site import export_static

export_static("site")  # the same as: plainml web --export site
```

## Sharing and deploying

```python
from plainml.card import write_model_card
from plainml.deploy import deploy
from plainml.tracking import export_mlflow, log_run  # needs the "mlflow" extra

write_model_card("runs/20260925-125240_churn")  # model_card.md (written automatically by train)
deploy("latest", output="deploy/churn")  # Dockerfile + pinned requirements + model
log_run("runs/20260925-125240_churn", experiment="churn")
export_mlflow("latest", output="churn_mlflow")
```
