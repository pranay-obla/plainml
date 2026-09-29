"""Sharing and running models: model cards, calibration, deploy, API keys, big files, MLflow."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import plainml
from plainml.cli import cli
from plainml.evaluation import calibration_summary


@pytest.fixture(scope="module")
def churn_run(tmp_path_factory: pytest.TempPathFactory, messy_csv: Path) -> Path:
    out = tmp_path_factory.mktemp("runs")
    result = plainml.train(
        messy_csv, target="churned", models=["nb"], ensemble=False, cv=3, out_dir=out, verbose=False
    )
    return result.run_dir


def test_calibration_summary() -> None:
    rng = np.random.default_rng(0)
    probability = rng.uniform(size=5000)
    honest = calibration_summary(rng.uniform(size=5000) < probability, probability)
    overconfident = calibration_summary(
        rng.uniform(size=5000) < 0.5, np.where(probability > 0.5, 0.99, 0.01)
    )
    assert honest["ece"] < 0.03 < 0.3 < overconfident["ece"]
    assert sum(p["rows"] for p in honest["points"]) == 5000


def test_calibrate_option(messy_csv: Path) -> None:
    result = plainml.train(
        messy_csv,
        target="churned",
        models=["nb"],
        ensemble=False,
        calibrate=True,
        cv=3,
        verbose=False,
        report=False,
    )
    assert "calibration" in result.evaluation
    assert type(result.model).__name__ in ("CalibratedClassifierCV", "TunedThresholdClassifierCV")


def test_model_card(churn_run: Path) -> None:
    card = (churn_run / "model_card.md").read_text()
    for heading in (
        "## At a glance",
        "## Intended use",
        "## Performance",
        "## Caveats",
        "## How to use it",
    ):
        assert heading in card
    assert "`churned`" in card and "Naive Bayes" in card


def test_deploy_folder(churn_run: Path, tmp_path: Path) -> None:
    from plainml.deploy import deploy

    folder = deploy(churn_run, output=tmp_path / "image", verbose=False)
    requirements = (folder / "requirements.txt").read_text()
    assert "plainml[serve]==" in requirements and "scikit-learn==" in requirements
    assert "lightgbm" not in requirements  # only what this model needs
    dockerfile = (folder / "Dockerfile").read_text()
    assert "USER plainml" in dockerfile and "HEALTHCHECK" in dockerfile
    assert (folder / "model.joblib").is_file() and (folder / "README.md").is_file()
    from plainml.deploy import _local_plainml_source

    if _local_plainml_source() is not None:  # a checkout: the image gets a wheel built from it
        assert list((folder / "wheels").glob("plainml-*.whl"))
    with pytest.raises(plainml.PlainMLError, match="isn't empty"):
        deploy(churn_run, output=folder, verbose=False)


def test_serve_api_key(churn_run: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from plainml.serve import create_app

    client = TestClient(create_app(churn_run / "model.joblib", api_key="s3cret"))
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 401
    row = {"age": 40, "plan": "pro"}
    assert client.post("/predict", json=row, headers={"X-API-Key": "wrong"}).status_code == 401
    answer = client.post("/predict", json=row, headers={"X-API-Key": "s3cret"})
    assert answer.status_code == 200


def test_chunked_prediction(churn_run: Path, messy_csv: Path, tmp_path: Path) -> None:
    whole = plainml.predict(churn_run, messy_csv)
    output = tmp_path / "out.csv"
    preview = plainml.predict(churn_run, messy_csv, output=output, chunk_size=50)
    assert len(preview) == 50 and preview.attrs["rows_written"] == len(whole)
    streamed = pd.read_csv(output)
    assert streamed["predicted_churned"].tolist() == whole["predicted_churned"].tolist()
    with pytest.raises(plainml.PlainMLError, match="output file"):
        plainml.predict(churn_run, messy_csv, chunk_size=50)


def test_runs_prune(messy_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    for _ in range(3):
        plainml.train(
            messy_csv,
            target="churned",
            models=["nb"],
            ensemble=False,
            cv=3,
            out_dir=out,
            verbose=False,
            report=False,
        )
    runner = CliRunner()
    declined = runner.invoke(
        cli, ["runs", "--runs-dir", str(out), "--prune", "--keep", "1"], input="n\n"
    )
    assert declined.exit_code == 0 and len(list(out.iterdir())) == 3
    runner.invoke(cli, ["runs", "--runs-dir", str(out), "--prune", "--keep", "1", "--yes"])
    assert len(list(out.iterdir())) == 1


def test_mlflow_log_and_export(
    churn_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlflow")
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    from plainml.tracking import export_mlflow, log_run

    run_id = log_run(churn_run, experiment="tests", verbose=False)
    logged = mlflow.get_run(run_id)
    assert "test_f1" in logged.data.metrics and logged.data.tags["plainml.target"] == "churned"
    folder = export_mlflow(churn_run, output=tmp_path / "mlflow_model", verbose=False)
    loaded = mlflow.pyfunc.load_model(str(folder))
    assert len(loaded.predict(pd.DataFrame([{"age": 40.0, "plan": "pro"}]))) == 1
    os.environ.pop("MLFLOW_TRACKING_URI", None)
