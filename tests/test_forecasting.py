"""Forecasting beyond one plain series: groups, inputs known in advance, and holidays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plainml
from plainml.errors import PlainMLError
from plainml.forecasting import GroupForecaster, build_series


def _stores(days: int = 200, with_plans: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    frames = []
    for store, scale in [("big", 300.0), ("mid", 80.0), ("small", 20.0)]:
        promo = (np.arange(days) % 5 == 0).astype(int)
        weekly = np.array([0.9, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2])[dates.dayofweek]
        sales = scale * weekly * (1 + 0.8 * promo) * rng.normal(1, 0.03, days)
        frames.append(pd.DataFrame({"day": dates, "store": store, "promo": promo, "sales": sales}))
        if with_plans:
            future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7, freq="D")
            plans = np.array([1, 0, 0, 0, 0, 0, 0])
            frames.append(
                pd.DataFrame({"day": future, "store": store, "promo": plans, "sales": np.nan})
            )
    return pd.concat(frames, ignore_index=True)


def test_grouped_forecast_has_a_row_per_group_and_date(tmp_path: Path) -> None:
    result = plainml.forecast(
        _stores(), "sales", group="store", horizon=7, models=["ridge"], verbose=False
    )
    frame = result.forecast
    assert list(frame.columns[:2]) == ["store", "date"]
    assert frame.groupby("store").size().to_dict() == {"big": 7, "mid": 7, "small": 7}
    means = frame.groupby("store")["forecast"].mean()
    assert means["big"] > means["mid"] > means["small"]  # each series keeps its own level
    assert isinstance(result.model, GroupForecaster)
    assert (result.run_dir / "report.html").is_file()
    again = plainml.predict(result.run_dir, horizon=3)
    assert len(again) == 9


def test_future_inputs_from_later_rows_are_used() -> None:
    frame = _stores(with_plans=True).query("store == 'big'").drop(columns="store")
    series, _, _, history, planned = build_series(frame, "sales", "day", "D", "sum", ["promo"])
    assert series.index[-1] == pd.Timestamp("2024-07-18")  # plan rows are not history
    assert planned is not None and len(planned) == 7 and history is not None
    result = plainml.forecast(
        frame, "sales", inputs=["promo"], horizon=7, models=["ridge"], verbose=False, report=False
    )
    forecast = result.forecast.set_index("date")["forecast"]
    first, rest = forecast.iloc[0], forecast.iloc[1:].mean()
    assert first > rest * 1.3  # the planned promo day is forecast well above the others
    assert any("promo is on" in s for s in result.insights)


def test_future_file_and_missing_plans(tmp_path: Path) -> None:
    frame = _stores().query("store == 'big'").drop(columns="store")
    plans = pd.DataFrame(
        {"day": pd.date_range("2024-07-19", periods=7, freq="D"), "promo": [0, 0, 1, 0, 0, 0, 0]}
    )
    plans_path = tmp_path / "plans.csv"
    plans.to_csv(plans_path, index=False)
    options = {
        "inputs": ["promo"],
        "horizon": 7,
        "models": ["ridge"],
        "verbose": False,
        "report": False,
    }
    planned = plainml.forecast(frame, "sales", future=plans_path, **options).forecast["forecast"]
    assert planned.idxmax() == 2
    held = plainml.forecast(frame, "sales", **options)
    assert held.model.assumed_inputs_  # no plans: the last value is carried forward


def test_holidays_become_features() -> None:
    pytest.importorskip("holidays")
    from plainml.forecasting import _calendar

    dates = pd.DatetimeIndex(["2024-12-24", "2024-12-25", "2024-12-26", "2024-12-27"])
    features = _calendar(dates, "D", "US")
    holiday, before, after = features[:, -3], features[:, -2], features[:, -1]
    assert holiday.tolist() == [0, 1, 0, 0]
    assert before.tolist() == [1, 0, 0, 0] and after.tolist() == [0, 0, 1, 0]
    with pytest.raises(PlainMLError, match="No holiday calendar"):
        plainml.forecast(_stores(), "sales", group="store", country="XX", verbose=False)


def test_grouped_prediction_continues_from_new_history() -> None:
    frame = _stores(days=160)
    result = plainml.forecast(
        frame, "sales", group="store", horizon=5, models=["ridge"], verbose=False, report=False
    )
    newer = _stores(days=200)
    continued = plainml.predict(result.run_dir, newer.query("store != 'small'"), horizon=5)
    assert set(continued["store"]) == {"big", "mid"}
    assert continued["date"].min() == pd.Timestamp("2024-07-19")


def test_saved_model_remembers_planned_inputs() -> None:
    frame = _stores(with_plans=True)
    result = plainml.forecast(
        frame,
        "sales",
        group="store",
        inputs=["promo"],
        horizon=7,
        models=["ridge"],
        verbose=False,
        report=False,
    )
    later = plainml.predict(result.run_dir, horizon=7)
    pd.testing.assert_frame_equal(
        later.reset_index(drop=True), result.forecast.reset_index(drop=True)
    )
