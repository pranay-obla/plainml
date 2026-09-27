"""End-to-end training, prediction, evaluation and explanation (small data, fast models)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import plainml
from plainml.errors import PlainMLError
from plainml.runs import list_runs, load_run, resolve_run

FAST = {"models": ["linear", "tree"], "cv": 3, "verbose": False}


def test_classification_run_folder(iris_csv: Path) -> None:
    result = plainml.train(iris_csv, target="species", **FAST)
    run = result.run_dir
    for name in (
        "model.joblib",
        "report.html",
        "leaderboard.csv",
        "evaluation.json",
        "importance.csv",
        "holdout_predictions.csv",
        "run.json",
        "config.yaml",
    ):
        assert (run / name).is_file(), name
    info = json.loads((run / "run.json").read_text())
    assert info["task"] == "classification" and info["metric"] == "f1"
    assert info["best"]["cv_score"] > 0.85
    assert set(result.leaderboard["key"]) >= {"baseline", "linear", "tree"}
    assert "Predicting species" in (run / "report.html").read_text()
    # nothing is written to the working directory except runs/
    assert sorted(p.name for p in Path.cwd().iterdir()) == ["runs"]


def test_saved_model_accepts_raw_rows(iris_csv: Path) -> None:
    """Saved models carry their preprocessing, so raw rows work."""
    result = plainml.train(iris_csv, target="species", **FAST)
    model = joblib.load(result.model_path)
    raw = pd.read_csv(iris_csv).drop(columns="species")
    assert (model.predict(raw) == pd.read_csv(iris_csv)["species"]).mean() > 0.9
    assert model.plainml_meta_["features"] == list(raw.columns)


def test_regression_with_negative_targets(diabetes_csv: Path) -> None:
    """Targets below -1 must not break metrics (no MSLE)."""
    result = plainml.train(diabetes_csv, target="target", **FAST)
    assert result.task == "regression" and result.metric == "rmse"
    baseline = result.leaderboard.set_index("key").loc["baseline", "rmse"]
    assert result.cv_score < baseline


def test_messy_data(messy_csv: Path) -> None:
    """Text, blanks, text numbers, day-first dates, IDs and free text all in one file."""
    result = plainml.train(
        messy_csv, target="churned", models=["linear", "rf"], cv=3, verbose=False
    )
    schema = result.model.plainml_meta_["schema"]
    assert "customer_id" in schema["dropped"]
    assert (
        "income" in schema["numeric"]
        and "signup" in schema["datetime"]
        and "notes" in schema["text"]
    )
    preds = plainml.predict(result.run_dir, messy_csv)
    assert set(preds["predicted_churned"]) <= {"yes", "no"}


def test_leak_warning(iris_csv: Path, tmp_path: Path) -> None:
    frame = pd.read_csv(iris_csv)
    frame["species_code"] = frame["species"].map(
        {"setosa": "S", "versicolor": "V", "virginica": "G"}
    )
    path = tmp_path / "leaky.csv"
    frame.to_csv(path, index=False)
    result = plainml.train(path, target="species", **FAST)
    assert any(issue.code == "leak" for issue in result.profile.issues)


def test_imbalanced_uses_weights_and_threshold(imbalanced_csv: Path) -> None:
    result = plainml.train(imbalanced_csv, target="fraud", models=["linear"], cv=3, verbose=False)
    info = load_run(result.run_dir).info
    assert info["balance"] == "weights"
    assert info["threshold"] is not None


def test_smote(imbalanced_csv: Path) -> None:
    pytest.importorskip("imblearn")
    result = plainml.train(
        imbalanced_csv, target="fraud", models=["linear"], balance="smote", cv=3, verbose=False
    )
    assert load_run(result.run_dir).info["balance"] == "smote"


def test_multilabel(multilabel_csv: Path) -> None:
    result = plainml.train(
        multilabel_csv,
        target=["is_spam", "is_urgent"],
        models=["rf", "linear"],
        cv=3,
        verbose=False,
    )
    assert result.task == "multilabel"
    preds = plainml.predict(result.run_dir, multilabel_csv)
    assert set(preds["predicted_is_spam"]) <= {"yes", "no"}
    assert set(preds["predicted_is_urgent"]) <= {0, 1}


def test_multi_regression(diabetes_csv: Path) -> None:
    result = plainml.train(
        diabetes_csv, target="target,bmi", models=["ridge", "rf"], cv=3, verbose=False
    )
    assert result.task == "multi_regression"


def test_log_target_and_regression_metric(diabetes_csv: Path, tmp_path: Path) -> None:
    frame = pd.read_csv(diabetes_csv)
    frame["target"] = frame["target"] + 200
    path = tmp_path / "positive.csv"
    frame.to_csv(path, index=False)
    result = plainml.train(
        path, target="target", models=["ridge"], log_target=True, metric="mae", cv=3, verbose=False
    )
    assert result.metric == "mae"
    assert plainml.predict(result.run_dir, path)["predicted_target"].min() > 0


def test_ensemble_and_save_all(iris_csv: Path) -> None:
    result = plainml.train(
        iris_csv,
        target="species",
        models=["linear", "tree", "nb"],
        save_all=True,
        zip=True,
        cv=3,
        verbose=False,
    )
    assert "ensemble" in set(result.leaderboard["key"])
    assert (result.run_dir / "models" / "tree.joblib").is_file()
    assert result.run_dir.with_suffix(".zip").is_file()


def test_time_budget_skips(iris_csv: Path) -> None:
    result = plainml.train(
        iris_csv,
        target="species",
        models=["linear", "tree", "nb", "knn"],
        time_budget=0.001,
        cv=3,
        verbose=False,
    )
    notes = result.leaderboard.set_index("key")["note"]
    assert (
        result.leaderboard["status"].eq("ok").sum() >= 2
    )  # the baseline and at least one real model always run
    assert (notes == "time budget reached").sum() >= 2


def test_friendly_errors(iris_csv: Path) -> None:
    with pytest.raises(PlainMLError, match="Did you mean 'species'"):
        plainml.train(iris_csv, target="specie", verbose=False)
    with pytest.raises(PlainMLError, match="Unknown model"):
        plainml.train(iris_csv, target="species", models=["randomforrest"], verbose=False)
    with pytest.raises(PlainMLError, match="Metric"):
        plainml.train(iris_csv, target="species", metric="rmse", verbose=False)
    with pytest.raises(PlainMLError, match="Unknown training option"):
        plainml.train(iris_csv, target="species", epochs=3, verbose=False)


def test_config_file(iris_csv: Path, tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(f"data: {iris_csv}\ntarget: species\nmodels: [linear]\ncv: 3\n")
    result = plainml.train(config=config, verbose=False)
    assert result.best_key == "linear"
    rerun = (result.run_dir / "config.yaml").read_text()
    assert "target: species" in rerun
    bad = tmp_path / "bad.yaml"
    bad.write_text("targt: species\n")
    with pytest.raises(PlainMLError, match="Did you mean 'target'"):
        plainml.train(config=bad, verbose=False)


# --- using models ---------------------------------------------------------------------------


@pytest.fixture
def iris_run(iris_csv: Path) -> Path:
    return plainml.train(iris_csv, target="species", **FAST).run_dir


def test_predict_variants(iris_run: Path, iris_csv: Path, tmp_path: Path) -> None:
    frame = pd.read_csv(iris_csv)
    out = plainml.predict(
        "latest", frame.drop(columns="species"), proba=True, output=tmp_path / "p.xlsx"
    )
    assert {"predicted_species", "confidence", "probability_setosa"} <= set(out.columns)
    assert (tmp_path / "p.xlsx").is_file()
    partial = frame.drop(columns=["species", "petal width (cm)"])
    assert len(plainml.predict(iris_run, partial)) == len(
        frame
    )  # missing column → blank, with a warning
    with pytest.raises(PlainMLError, match="Missing columns"):
        plainml.predict(iris_run, partial, strict=True)
    with pytest.raises(PlainMLError, match="None of the model's input columns"):
        plainml.predict(iris_run, pd.DataFrame({"x": [1]}))


def test_evaluate(iris_run: Path, iris_csv: Path, tmp_path: Path) -> None:
    result = plainml.evaluate(iris_run, iris_csv, report=tmp_path / "e.html", verbose=False)
    assert result["scores"]["accuracy"] > 0.85
    assert "f1" in result["training_scores"]
    assert (tmp_path / "e.html").is_file()


def test_explain(iris_run: Path, iris_csv: Path) -> None:
    explanation = plainml.explain(iris_run, verbose=False)
    assert explanation.labelled and len(explanation.importance) == 4
    assert any("petal" in s for s in explanation.sentences)
    unlabelled = plainml.explain(
        iris_run, pd.read_csv(iris_csv).drop(columns="species"), row=0, verbose=False
    )
    assert not unlabelled.labelled and unlabelled.row_headline.startswith("Predicted")


def test_shap(iris_run: Path) -> None:
    pytest.importorskip("shap")
    explanation = plainml.explain(iris_run, use_shap=True, verbose=False)
    assert explanation.shap is not None and len(explanation.shap) == 4


def test_runs_listing(iris_run: Path) -> None:
    runs = list_runs()
    assert len(runs) == 1 and runs.iloc[0]["best_model"]
    assert resolve_run("latest") == iris_run
    assert resolve_run(iris_run.name[-4:]) == iris_run
    with pytest.raises(PlainMLError):
        resolve_run("does-not-exist")


def test_report_escapes_user_text(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"<script>alert(1)</script>": rng.normal(size=60), "y": rng.choice(["a</script>", "b"], 60)}
    )
    path = tmp_path / "evil.csv"
    frame.to_csv(path, index=False)
    result = plainml.train(path, target="y", models=["linear"], cv=3, verbose=False)
    html = result.report_path.read_text()
    assert "<script>alert(1)</script>" not in html
    assert html.count("</script>") == 2  # just the page's own data + code blocks
