"""The command line, as a user would run it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from plainml import __version__
from plainml.cli import cli


def run(*args: str, code: int = 0) -> str:
    result = CliRunner().invoke(cli, [str(a) for a in args])
    output = result.output
    assert result.exit_code == code, (
        f"exit {result.exit_code} != {code}\n{output}\n{result.exception!r}"
    )
    assert "Traceback" not in output
    return output


def test_help_and_version() -> None:
    output = run("--help")
    for section in (
        "Start here",
        "Use a trained model",
        "Other kinds of problem",
        "Deploy and share",
    ):
        assert section in output
    assert __version__ in run("--version")
    assert "Examples" in run("train", "--help")


def test_friendly_errors(iris_csv: Path) -> None:
    assert "Did you mean 'train'" in run("trian", code=2)
    assert "plainml train" in run(str(iris_csv), code=2)
    output = run("train", iris_csv, code=1)  # no --target and no terminal to ask in
    assert "Which column" in output and "species" in output
    assert "Did you mean 'species'" in run("train", iris_csv, "-t", "Specis", code=1)
    assert "does not exist" in run("train", "nope.csv", "-t", "x", code=1)
    assert "No runs found" in run("predict", "latest", iris_csv, code=1)


def test_debug_shows_traceback(iris_csv: Path) -> None:
    result = CliRunner().invoke(cli, ["--debug", "train", str(iris_csv), "-t", "nope"])
    assert result.exit_code != 0 and result.exception is not None


def test_train_predict_explain_flow(iris_csv: Path, tmp_path: Path) -> None:
    output = run(
        "train",
        iris_csv,
        "-t",
        "species",
        "--models",
        "linear,tree",
        "--cv",
        "3",
        "--name",
        "flowers",
    )
    assert "Best model" in output and "Leaderboard" in output
    run("predict", "latest", iris_csv, "-o", tmp_path / "p.csv", "--proba")
    predictions = pd.read_csv(tmp_path / "p.csv")
    assert (
        "predicted_species" in predictions.columns and "probability_setosa" in predictions.columns
    )
    assert "In plain English" in run("explain", "flowers")
    assert "This prediction" in run("explain", "latest", iris_csv, "--row", "3")
    assert "Scores on" in run("evaluate", "latest", iris_csv)
    assert "report.html" in run("report", "latest", "--no-open")
    assert "flowers" in run("runs")
    assert "Best model" in run("compare")


def test_quiet(iris_csv: Path) -> None:
    assert (
        run("-q", "train", iris_csv, "-t", "species", "--models", "linear", "--cv", "3").strip()
        == ""
    )


def test_config_flow(iris_csv: Path) -> None:
    assert "Created" in run("init", "c.yaml", "--data", iris_csv, "-t", "species")
    assert "exists" in run("init", "c.yaml", code=1)
    assert "Best model" in run("train", "--config", "c.yaml", "--quick", "--no-report")


def test_data_commands(
    iris_csv: Path, customers_csv: Path, transactions_csv: Path, sales_csv: Path, tmp_path: Path
) -> None:
    assert "species" in run("profile", iris_csv, "-t", "species", "--html", tmp_path / "p.html")
    assert (tmp_path / "p.html").is_file()
    assert "Cleaning" in run("clean", iris_csv, "-o", tmp_path / "c.csv")
    assert "Column ranking" in run("select", iris_csv, "-t", "species", "-k", "2")
    assert "groups found" in run("cluster", customers_csv, "-k", "2-4", "--algorithms", "kmeans")
    assert "Most unusual rows" in run("anomaly", transactions_csv, "--drop", "is_fraud")
    assert "Forecast" in run(
        "forecast", sales_csv, "-t", "revenue", "--horizon", "7", "--models", "naive,ridge"
    )
    regression_models = run("models", "--task", "regression")
    assert "elasticnet" in regression_models and "Naive Bayes" not in regression_models
