"""One test per bug found in the 0.1.0 audit, so none of them can quietly come back."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plainml
from plainml.errors import PlainMLError
from plainml.schema import Kind, infer_column


def test_forecast_continues_trend_from_any_history() -> None:
    """Continuing from recent rows used to restart the trend counter (283 instead of ~1300)."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2023-01-01", periods=600)
    frame = pd.DataFrame(
        {"d": days.astype(str), "v": 100 + 2.0 * np.arange(600) + rng.normal(0, 3, 600)}
    )
    result = plainml.forecast(
        frame, "v", date="d", horizon=7, models=["ridge"], verbose=False, report=False
    )
    recent = plainml.predict(result.run_dir, frame.tail(60), horizon=3)["forecast"]
    assert np.allclose(recent, [1300, 1302, 1304], atol=8)
    early = plainml.predict(result.run_dir, frame.iloc[:300], horizon=3)["forecast"]
    assert np.allclose(early, [700, 702, 704], atol=8)


def test_old_library_versions_get_a_clear_message(
    iris_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = plainml.train(iris_csv, target="species", models=["linear"], cv=3, verbose=False)
    info = json.loads((result.run_dir / "run.json").read_text())
    info["environment"]["packages"]["scikit-learn"] = "0.24.2"
    (result.run_dir / "run.json").write_text(json.dumps(info))

    import plainml.predicting as predicting

    def broken_load(path: Path) -> None:
        raise AttributeError("Can't get attribute '_RemainderColsList'")

    monkeypatch.setattr(predicting.joblib, "load", broken_load)
    with pytest.raises(PlainMLError) as error:
        plainml.load_model(result.run_dir)
    assert "scikit-learn 0.24.2" in error.value.message
    assert '"scikit-learn==0.24.2"' in (error.value.hint or "")


def test_version_mismatch_warning_is_shown(
    iris_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = plainml.train(iris_csv, target="species", models=["linear"], cv=3, verbose=False)
    model = result.model
    model.plainml_meta_["packages"] = {**model.plainml_meta_["packages"], "scikit-learn": "0.24.2"}
    import joblib

    joblib.dump(model, result.model_path)
    plainml.load_model(result.run_dir)
    assert "saved with scikit-learn 0.24.2" in capsys.readouterr().out.replace("\n", " ")


def test_constant_target_is_rejected() -> None:
    frame = pd.DataFrame({"x": np.random.default_rng(0).normal(size=60), "y": [3.5] * 60})
    with pytest.raises(PlainMLError, match="same value"):
        plainml.train(frame, target="y", verbose=False)


def test_unique_value_target_gets_the_right_message() -> None:
    frame = pd.DataFrame({"name": [f"p{i}" for i in range(100)], "age": range(100)})
    with pytest.raises(PlainMLError, match="no repeated classes"):
        plainml.train(frame, target="name", verbose=False)


def test_years_are_not_row_ids() -> None:
    assert infer_column(pd.Series(range(1960, 2021), name="year")).kind == Kind.NUMERIC
    assert infer_column(pd.Series(range(0, 50), name="row")).kind == Kind.ID
    assert infer_column(pd.Series(range(1, 50), name="n")).kind == Kind.ID


def test_mixed_time_zones() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "when": ["2024-01-01T10:00:00+05:30", "2024-01-02T11:00:00Z", "2024-01-03 09:00"] * 30,
            "x": rng.normal(size=90),
            "y": rng.choice(["a", "b"], 90),
        }
    )
    result = plainml.train(frame, target="y", quick=True, verbose=False, report=False)
    assert result.model.plainml_meta_["schema"]["datetime"] == ["when"]


def test_windows_and_bom_encodings(tmp_path: Path) -> None:
    pd.DataFrame({"café": ["crème", "brûlée"], "y": [1, 2]}).to_csv(
        tmp_path / "a.csv", index=False, encoding="cp1252"
    )
    pd.DataFrame({"name": ["a"], "y": [1]}).to_csv(
        tmp_path / "b.csv", index=False, encoding="utf-8-sig"
    )
    assert plainml.load_data(tmp_path / "a.csv")["café"].tolist() == ["crème", "brûlée"]
    assert plainml.load_data(tmp_path / "b.csv").columns.tolist() == ["name", "y"]  # no stray BOM


def test_excel_picks_the_largest_sheet(tmp_path: Path) -> None:
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Notes", index=False)
        pd.DataFrame({"x": range(50), "y": range(50)}).to_excel(
            writer, sheet_name="Data", index=False
        )
    assert plainml.load_data(path).columns.tolist() == ["x", "y"]
    assert plainml.load_data(path, sheet="Notes").columns.tolist() == ["a"]
    with pytest.raises(PlainMLError, match="Did you mean 'Data'"):
        plainml.load_data(path, sheet="Dat")


def test_ambiguous_run_names_are_rejected(iris_csv: Path) -> None:
    first = plainml.train(
        iris_csv, target="species", models=["linear"], cv=3, verbose=False, name="alpha"
    )
    plainml.train(iris_csv, target="species", models=["linear"], cv=3, verbose=False, name="alpine")
    assert plainml.load_run("alpha").path == first.run_dir  # an exact name wins
    with pytest.raises(PlainMLError, match="matches 2 runs"):
        plainml.load_run("alp")


def test_true_false_targets_with_blanks_stay_boolean() -> None:
    frame = pd.DataFrame(
        {"x": np.random.default_rng(0).normal(size=80), "flag": [True, False, None, True] * 20}
    )
    result = plainml.train(frame, target="flag", quick=True, verbose=False, report=False)
    assert result.model.classes_.tolist() == [False, True]
