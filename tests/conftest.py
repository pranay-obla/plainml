"""Shared fixtures: small, fast datasets written to a temporary folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plainml.console import set_quiet


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs in its own folder (so 'runs/' never leaks) with quiet output."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "120")
    set_quiet(False)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


def _save(df: pd.DataFrame, folder: Path, name: str) -> Path:
    path = folder / name
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def iris_csv(data_dir: Path) -> Path:
    from sklearn.datasets import load_iris

    frame = load_iris(as_frame=True).frame
    frame["species"] = frame.pop("target").map({0: "setosa", 1: "versicolor", 2: "virginica"})
    return _save(frame, data_dir, "iris.csv")


@pytest.fixture(scope="session")
def diabetes_csv(data_dir: Path) -> Path:
    from sklearn.datasets import load_diabetes

    frame = load_diabetes(as_frame=True).frame
    frame["target"] = frame["target"] - 150  # negative targets rule out log-based metrics like MSLE
    return _save(frame, data_dir, "diabetes.csv")


@pytest.fixture(scope="session")
def messy_csv(data_dir: Path) -> Path:
    """Strings, blanks, dates, an ID, free text and a column that leaks the target."""
    rng = np.random.default_rng(0)
    n = 240
    age = rng.integers(18, 70, n).astype(float)
    income = rng.normal(60_000, 15_000, n)
    plan = rng.choice(["basic", "pro", "team"], n)
    logit = (age - 40) / 10 + (plan == "team") * 1.5 - (income - 60_000) / 20_000
    churned = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1000 + n),
            "age": age,
            "income": [f"${v:,.0f}" for v in income],
            "plan": plan,
            "signup": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%d/%m/%Y"),
            "notes": [
                " ".join(
                    rng.choice(
                        [
                            "called",
                            "support",
                            "billing",
                            "happy",
                            "service",
                            "asked",
                            "team",
                            "pricing",
                            "slow",
                            "great",
                            "refund",
                            "upgrade",
                        ],
                        6,
                    )
                )
                for _ in range(n)
            ],
            "churned": churned,
        }
    )
    frame.loc[::9, "age"] = np.nan
    frame.loc[::13, "plan"] = None
    return _save(frame, data_dir, "messy.csv")


@pytest.fixture(scope="session")
def imbalanced_csv(data_dir: Path) -> Path:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=4, weights=[0.9], random_state=1
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    frame["fraud"] = np.where(y == 1, "fraud", "ok")
    return _save(frame, data_dir, "imbalanced.csv")


@pytest.fixture(scope="session")
def multilabel_csv(data_dir: Path) -> Path:
    rng = np.random.default_rng(2)
    n = 300
    frame = pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.choice(["a", "b"], n)}
    )
    frame["is_spam"] = np.where(frame["x1"] + rng.normal(0, 0.3, n) > 0, "yes", "no")
    frame["is_urgent"] = (frame["x2"] > 0.5).astype(int)
    return _save(frame, data_dir, "multilabel.csv")


@pytest.fixture(scope="session")
def sales_csv(data_dir: Path) -> Path:
    rng = np.random.default_rng(11)
    days = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    weekly = np.array([0.9, 0.95, 1.0, 1.0, 1.1, 1.35, 1.25])[days.dayofweek]
    sales = (200 + 0.1 * np.arange(len(days))) * weekly * rng.normal(1, 0.05, len(days))
    frame = pd.DataFrame({"day": days.strftime("%d/%m/%Y"), "revenue": sales.round(2)})
    return _save(frame.drop(index=[40, 41, 200]), data_dir, "sales.csv")


@pytest.fixture(scope="session")
def customers_csv(data_dir: Path) -> Path:
    rng = np.random.default_rng(3)

    def segment(
        n: int, age: float, income: float, visits: float, plan_p: list[float]
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": rng.normal(age, 3, n).round(),
                "income": rng.normal(income, 5000, n).round(-2),
                "visits": rng.poisson(visits, n),
                "plan": rng.choice(["basic", "pro", "team"], n, p=plan_p),
            }
        )

    frame = pd.concat(
        [
            segment(120, 24, 32_000, 22, [0.8, 0.15, 0.05]),
            segment(90, 45, 95_000, 6, [0.1, 0.6, 0.3]),
            segment(60, 38, 60_000, 40, [0.05, 0.15, 0.8]),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=1)
    frame.insert(0, "customer_id", range(len(frame)))
    return _save(frame, data_dir, "customers.csv")


@pytest.fixture(scope="session")
def transactions_csv(data_dir: Path) -> Path:
    rng = np.random.default_rng(7)
    n = 800
    frame = pd.DataFrame(
        {
            "amount": rng.lognormal(3.5, 0.4, n).round(2),
            "hour": rng.integers(7, 22, n),
            "country": rng.choice(["US", "CA", "UK"], n, p=[0.7, 0.2, 0.1]),
        }
    )
    fraud = rng.choice(n, 16, replace=False)
    frame.loc[fraud, "amount"] = rng.uniform(3000, 9000, 16).round(2)
    frame["is_fraud"] = 0
    frame.loc[fraud, "is_fraud"] = 1
    return _save(frame, data_dir, "transactions.csv")
