"""Guided mode, driven by scripted answers instead of a keyboard."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from plainml import wizard


class _Prompt:
    def __init__(self, answer: Any):
        self.answer = answer

    def ask(self) -> Any:
        return self.answer


@pytest.fixture()
def answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Queue answers; each questionary prompt takes the next one."""
    queue: list[Any] = []

    def prompt(*_: Any, **__: Any) -> _Prompt:
        assert queue, "the wizard asked more questions than the test expected"
        return _Prompt(queue.pop(0))

    for name in ("select", "text", "checkbox", "confirm", "path"):
        monkeypatch.setattr(wizard.questionary, name, prompt)
    yield queue
    assert not queue, f"unused answers: {queue}"


def test_wizard_trains(
    answers: list[Any], iris_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # data file, target, task confirmation, speed, columns to drop, open the report?
    answers += ["__other__", str(iris_csv), "species", "classification", "quick", [], False]
    wizard.run_wizard(start="train")
    assert any((tmp_path / "runs").iterdir())


def test_wizard_forecasts_groups(
    answers: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    frame = pd.concat(
        pd.DataFrame(
            {
                "day": days,
                "shop": shop,
                "promo": rng.integers(0, 2, 120),
                "sales": rng.normal(level, 5, 120),
            }
        )
        for shop, level in (("a", 100), ("b", 40))
    )
    frame.to_csv(tmp_path / "shops.csv", index=False)
    # action, data file, target, date, group, inputs, horizon
    answers += ["forecast", "shops.csv", "sales", "day", "shop", ["promo"], "7"]
    wizard.run_wizard()
    run = next((tmp_path / "runs").iterdir())
    forecast = pd.read_csv(run / "forecast.csv")
    assert set(forecast["shop"]) == {"a", "b"} and len(forecast) == 14
