"""Drift: detection against a model and between two files, and the warning in predict."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import plainml
from plainml.drift import psi


@pytest.fixture
def shifted(messy_csv: Path, tmp_path: Path) -> Path:
    frame = pd.read_csv(messy_csv)
    frame["age"] = frame["age"] + 30  # everyone is older
    frame.loc[::3, "plan"] = "enterprise"  # a plan the model never saw
    path = tmp_path / "shifted.csv"
    frame.to_csv(path, index=False)
    return path


def test_psi_basics() -> None:
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert psi([0.9, 0.1], [0.1, 0.9]) > 0.25


def test_drift_against_a_model(messy_csv: Path, shifted: Path) -> None:
    run = plainml.train(messy_csv, target="churned", models=["linear"], cv=3, verbose=False)
    same = plainml.check_drift(run.run_dir, messy_csv, verbose=False)
    assert same.verdict == "stable" and not same.drifted
    changed = plainml.check_drift(run.run_dir, shifted, verbose=False)
    assert changed.verdict == "major"
    assert {"age", "plan"} <= set(changed.drifted)
    assert any("enterprise" in s for s in changed.sentences)
    assert (changed.run_dir / "report.html").is_file()


def test_drift_between_files(messy_csv: Path, shifted: Path) -> None:
    result = plainml.check_drift(messy_csv, shifted, target="churned", save=False, verbose=False)
    assert "age" in result.drifted


def test_predict_warns_on_drift(
    messy_csv: Path, shifted: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = plainml.train(messy_csv, target="churned", models=["linear"], cv=3, verbose=False)
    capsys.readouterr()
    plainml.predict(run.run_dir, messy_csv)
    assert "looks different" not in capsys.readouterr().out
    plainml.predict(run.run_dir, shifted)
    assert "looks different" in capsys.readouterr().out.replace("\n", " ")
