"""clean, select, tune, cluster, anomaly, forecast, serve and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plainml
from plainml.errors import PlainMLError


def test_clean(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            " id ": range(1, 41),
            "gender": ["Male", "male ", "FEMALE", "Female", "N/A"] * 8,
            "price": ["$1,200", "$950", "?", "$3,400"] * 10,
            "when": ["01/02/2024", "13/02/2024", "25/12/2024", "05/05/2024"] * 10,
            "empty": [None] * 40,
            "y": ["a", "b", None, "a"] * 10,
        }
    )
    raw = pd.concat([raw, raw.iloc[:3]])
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    out = plainml.clean(path, output=tmp_path / "clean.csv", target="y", impute=True, verbose=False)
    assert "id" not in out.columns and "empty" not in out.columns
    assert out["y"].notna().all()
    assert pd.api.types.is_numeric_dtype(out["price"]) and out["price"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(out["when"])
    assert set(out["gender"].dropna().str.lower()) == {"male", "female"}
    assert out["gender"].nunique() == 2  # capitalisation variants unified
    assert (tmp_path / "clean.csv").is_file()
    encoded = plainml.clean(path, encode=True, outliers="clip", verbose=False)
    assert any(c.startswith("gender_") for c in encoded.columns)


def test_select_features(diabetes_csv: Path, tmp_path: Path) -> None:
    table = plainml.select_features(
        diabetes_csv,
        "target",
        methods=["model", "anova", "mutual_info"],
        output=tmp_path / "r.csv",
        verbose=False,
    )
    assert set(table.columns) >= {"column", "consensus_rank", "selected", "random_forest_rank"}
    selected = table.loc[table["selected"], "column"].tolist()
    assert "bmi" in selected or "s5" in selected
    reduced = pd.read_csv(tmp_path / "r.csv")
    assert list(reduced.columns) == [*selected, "target"]
    with pytest.raises(PlainMLError, match="Unknown importance method"):
        plainml.select_features(diabetes_csv, "target", methods=["magic"], verbose=False)


@pytest.mark.parametrize("use_optuna", [True, False])
def test_tune(iris_csv: Path, use_optuna: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if use_optuna:
        pytest.importorskip("optuna")
    else:
        import plainml.tuning as tuning

        original = tuning.is_installed
        monkeypatch.setattr(
            tuning, "is_installed", lambda name: False if name == "optuna" else original(name)
        )
    first = plainml.train(iris_csv, target="species", models=["tree", "knn"], cv=3, verbose=False)
    tuned = plainml.tune(first.run_dir, trials=4, verbose=False)
    info = plainml.load_run(tuned.run_dir).info
    assert info["kind"] == "tune"
    assert {m["key"] for m in info["tuning"]["models"]} == {"tree", "knn"}
    assert info["tuning"]["method"].startswith("Optuna" if use_optuna else "random")
    assert "Tuning" in tuned.report_path.read_text()


def test_tune_data_file(diabetes_csv: Path) -> None:
    result = plainml.tune(diabetes_csv, target="target", models=["tree"], trials=3, verbose=False)
    assert result.best_key in ("tree", "tree@default")


def test_cluster(customers_csv: Path, tmp_path: Path) -> None:
    result = plainml.cluster(customers_csv, output=tmp_path / "seg.csv", verbose=False)
    assert result.k == 3
    assert sorted(result.labels.value_counts().tolist(), reverse=True) == [120, 90, 60]
    assert any("plan = basic" in s for s in result.descriptions)
    assigned = plainml.predict(result.run_dir, customers_csv)
    assert (assigned["cluster"].to_numpy() == result.labels.to_numpy()).mean() > 0.95
    assert "customer_id" not in result.model.plainml_meta_["features"]


def test_cluster_fixed_k_and_algorithms(customers_csv: Path) -> None:
    result = plainml.cluster(customers_csv, k="4", algorithms=["kmeans", "hdbscan"], verbose=False)
    assert set(result.leaderboard["algorithm"]) <= {"kmeans", "hdbscan"}
    with pytest.raises(PlainMLError, match="Unknown algorithm"):
        plainml.cluster(customers_csv, algorithms=["dbscan2"], verbose=False)


def test_anomaly_with_labels(transactions_csv: Path) -> None:
    result = plainml.detect_anomalies(transactions_csv, label="is_fraud", verbose=False)
    board = result.leaderboard
    assert board.iloc[0]["average_precision"] > 0.8
    assert result.flags.sum() == 16
    scored = plainml.predict(result.run_dir, transactions_csv)
    assert {"anomaly_score", "is_anomaly"} <= set(scored.columns)
    assert scored["is_anomaly"].sum() >= 10


def test_anomaly_unlabelled(transactions_csv: Path) -> None:
    result = plainml.detect_anomalies(
        transactions_csv, contamination="2%", drop=["is_fraud"], verbose=False
    )
    assert result.flags.mean() == pytest.approx(0.02, abs=0.01)
    assert result.top["why"].str.contains("amount").any()
    with pytest.raises(PlainMLError):
        plainml.detect_anomalies(transactions_csv, contamination="0.9", verbose=False)


def test_forecast(sales_csv: Path, tmp_path: Path) -> None:
    result = plainml.forecast(
        sales_csv,
        "revenue",
        horizon=14,
        models=["naive", "seasonal_naive", "ridge"],
        output=tmp_path / "f.csv",
        verbose=False,
    )
    frame = result.forecast
    assert (
        len(frame) == 14
        and (frame["lower_80"] <= frame["forecast"]).all()
        and (frame["forecast"] <= frame["upper_80"]).all()
    )
    assert frame["date"].iloc[0] == pd.Timestamp("2025-07-01")  # day-first dates parsed correctly
    assert any("Saturday is highest" in s for s in result.insights)
    assert result.leaderboard.iloc[0]["key"] != "naive"
    again = plainml.predict(result.run_dir, horizon=7)
    assert len(again) == 7
    newer = pd.read_csv(sales_csv).iloc[:-30]
    continued = plainml.predict(result.run_dir, newer, horizon=5)
    assert continued["date"].iloc[0] < frame["date"].iloc[0]


def test_forecast_errors(sales_csv: Path, tmp_path: Path) -> None:
    with pytest.raises(PlainMLError, match="Not enough history"):
        plainml.forecast(sales_csv, "revenue", horizon=400, verbose=False)
    no_dates = tmp_path / "nodates.csv"
    pd.DataFrame({"x": range(30), "y": range(30)}).to_csv(no_dates, index=False)
    with pytest.raises(PlainMLError, match="date column"):
        plainml.forecast(no_dates, "y", verbose=False)


def test_serve(iris_csv: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from plainml.serve import create_app

    result = plainml.train(iris_csv, target="species", models=["linear"], cv=3, verbose=False)
    client = TestClient(create_app(result.run_dir))
    about = client.get("/").json()
    assert about["task"] == "classification" and about["example_request"]
    row = about["example_request"]["rows"][0]
    answer = client.post("/predict", json={"rows": [row]})
    assert answer.status_code == 200 and "predicted_species" in answer.json()["predictions"][0]
    assert client.post("/predict", json=row).status_code == 200
    assert client.post("/predict", json="nonsense").status_code == 422
    assert client.post("/predict", json={"rows": [{"nope": 1}]}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}


def test_export_onnx(tmp_path: Path) -> None:
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "size": rng.normal(size=200),
            "city": rng.choice(["a b", "c"], 200),
            "y": rng.choice(["yes", "no"], 200),
        }
    )
    frame.loc[::7, "size"] = np.nan
    frame.loc[::11, "city"] = None
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    result = plainml.train(path, target="y", models=["rf"], ensemble=False, cv=3, verbose=False)
    from plainml.export import export_onnx

    out = export_onnx(result.run_dir, verbose=False)
    import json

    sidecar = json.loads(out.with_name(out.name + ".json").read_text())
    assert sidecar["verification"]["agreement"] == 1.0
    assert {i["column"] for i in sidecar["inputs"]} == {"size", "city"}


def test_export_rejects_unsupported(iris_csv: Path) -> None:
    pytest.importorskip("skl2onnx")
    pytest.importorskip("lightgbm")
    from plainml.export import export_onnx

    result = plainml.train(
        iris_csv, target="species", models=["lightgbm"], ensemble=False, cv=3, verbose=False
    )
    with pytest.raises(PlainMLError, match="can't be exported"):
        export_onnx(result.run_dir, verbose=False)
