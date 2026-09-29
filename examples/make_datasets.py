"""Regenerate the example datasets in this folder.

They're synthetic (no real people), realistic in shape, and deliberately a bit messy:
blank cells, prices stored as text like "$79.00", day-first dates, an ID column,
free-text notes, and imbalanced classes. Run:  python examples/make_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(2026)


def churn(n: int = 1500) -> pd.DataFrame:
    """Will a subscriber cancel? (classification, imbalanced, messy)"""
    tenure = rng.integers(1, 72, n)
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15])
    monthly = np.select([plan == "basic", plan == "standard"], [29, 59], 99) + rng.normal(0, 6, n)
    support_calls = rng.poisson(1.2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    contract = rng.choice(["monthly", "annual"], n, p=[0.65, 0.35])
    logit = (
        -1.6
        - 0.045 * tenure
        + 0.55 * support_calls
        + 0.9 * (contract == "monthly")
        + 0.012 * (monthly - 50)
        + rng.normal(0, 0.6, n)
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    words = [
        "slow",
        "helpful",
        "billing",
        "price",
        "great",
        "support",
        "app",
        "crash",
        "love",
        "cancel",
        "fast",
        "confusing",
    ]
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "signup_date": (
                pd.Timestamp("2026-06-30") - pd.to_timedelta(tenure * 30, unit="D")
            ).strftime("%d/%m/%Y"),
            "plan": plan,
            "contract": contract,
            "monthly_charges": [f"${v:,.2f}" for v in monthly],
            "tenure_months": tenure,
            "support_calls": support_calls,
            "region": region,
            "last_feedback": [" ".join(rng.choice(words, rng.integers(3, 8))) for _ in range(n)],
            "churned": np.where(churned, "yes", "no"),
        }
    )
    frame.loc[rng.choice(n, 60, replace=False), "region"] = None
    frame.loc[rng.choice(n, 40, replace=False), "tenure_months"] = np.nan
    return frame


def house_prices(n: int = 1200) -> pd.DataFrame:
    """What will a house sell for? (regression)"""
    neighborhood = rng.choice(
        ["riverside", "old town", "hillcrest", "industrial", "lakeview"],
        n,
        p=[0.25, 0.2, 0.2, 0.15, 0.2],
    )
    premium = {
        "riverside": 1.1,
        "old town": 1.25,
        "hillcrest": 1.35,
        "industrial": 0.75,
        "lakeview": 1.5,
    }
    sqft = rng.normal(1700, 550, n).clip(500, 5000).round(-1)
    bedrooms = np.clip((sqft / 600 + rng.normal(0, 0.7, n)).round(), 1, 7)
    bathrooms = np.clip((bedrooms * 0.6 + rng.normal(0, 0.5, n)).round() / 1, 1, 5)
    year = rng.integers(1920, 2025, n)
    condition = rng.choice(["poor", "fair", "good", "excellent"], n, p=[0.08, 0.27, 0.45, 0.2])
    condition_factor = (
        pd.Series(condition)
        .map({"poor": 0.8, "fair": 0.92, "good": 1.0, "excellent": 1.12})
        .to_numpy()
    )
    garage = rng.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])
    price = (
        180 * sqft * pd.Series(neighborhood).map(premium).to_numpy() * condition_factor
        + 12_000 * garage
        + 400 * (year - 1920)
    ) * rng.lognormal(0, 0.08, n)
    frame = pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms,
            "year_built": year,
            "neighborhood": neighborhood,
            "condition": condition,
            "garage_spaces": garage,
            "price": price.round(-2),
        }
    )
    frame.loc[rng.choice(n, 50, replace=False), "year_built"] = np.nan
    return frame


def daily_sales() -> pd.DataFrame:
    """How much will we sell each day? (forecasting: trend + weekly + yearly pattern)"""
    days = pd.date_range("2024-01-01", "2026-06-30", freq="D")
    t = np.arange(len(days))
    weekly = np.array([0.85, 0.9, 0.95, 1.0, 1.15, 1.4, 1.2])[days.dayofweek]
    yearly = 1 + 0.18 * np.sin(2 * np.pi * (days.dayofyear - 100) / 365.25)
    sales = (1200 + 0.9 * t) * weekly * yearly * rng.normal(1, 0.07, len(days))
    return pd.DataFrame(
        {"date": days.strftime("%Y-%m-%d"), "store": "downtown", "units_sold": sales.round()}
    )


def customers(n: int = 600) -> pd.DataFrame:
    """Who are our customers? (clustering: three planted segments)"""

    def segment(
        size: int, age: float, spend: float, visits: float, channel_p: list[float]
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": rng.normal(age, 5, size).round(),
                "annual_spend": rng.normal(spend, spend * 0.1, size).round(2),
                "visits_per_month": rng.poisson(visits, size),
                "preferred_channel": rng.choice(["app", "web", "store"], size, p=channel_p),
            }
        )

    frame = pd.concat(
        [
            segment(int(n * 0.45), 26, 900, 12, [0.7, 0.25, 0.05]),
            segment(int(n * 0.35), 48, 3200, 3, [0.1, 0.3, 0.6]),
            segment(n - int(n * 0.45) - int(n * 0.35), 38, 1900, 32, [0.2, 0.7, 0.1]),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=1)
    frame.insert(0, "member_id", range(5000, 5000 + len(frame)))
    return frame.reset_index(drop=True)


def transactions(n: int = 3000) -> pd.DataFrame:
    """Which card payments look suspicious? (anomaly detection, with a label to check against)"""
    frame = pd.DataFrame(
        {
            "amount": rng.lognormal(3.6, 0.7, n).round(2),
            "hour": rng.choice(
                np.arange(24),
                n,
                p=np.r_[np.full(7, 0.01), np.full(15, 0.058), np.full(2, 0.0165)]
                / np.r_[np.full(7, 0.01), np.full(15, 0.058), np.full(2, 0.0165)].sum(),
            ),
            "merchant_category": rng.choice(
                ["grocery", "fuel", "restaurant", "online", "travel"],
                n,
                p=[0.35, 0.15, 0.2, 0.25, 0.05],
            ),
            "country": rng.choice(["home", "abroad"], n, p=[0.96, 0.04]),
            "items": rng.poisson(2, n) + 1,
        }
    )
    fraud = rng.choice(n, 30, replace=False)
    frame.loc[fraud, "amount"] = rng.uniform(1500, 6000, 30).round(2)
    frame.loc[fraud[:18], "hour"] = rng.integers(1, 5, 18)
    frame.loc[fraud[:12], "country"] = "abroad"
    frame.loc[fraud[:10], "merchant_category"] = "online"
    frame["is_fraud"] = 0
    frame.loc[fraud, "is_fraud"] = 1
    return frame


def store_sales() -> pd.DataFrame:
    """Daily sales for three stores, with promotions planned two weeks ahead.

    Forecasting many series with inputs: each store has its own level and growth, promotions
    lift sales by about a third, and a few public holidays halve them. The last 14 rows of each
    store have a planned promo but no sales yet: that's what gets forecast.
    """
    days = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    holidays = pd.to_datetime(
        ["2025-01-01", "2025-07-04", "2025-11-27", "2025-12-25", "2026-01-01"]
    )
    weekly = np.array([0.9, 0.95, 1.0, 1.0, 1.1, 1.35, 1.2])[days.dayofweek]
    holiday = np.where(days.isin(holidays), 0.5, 1.0)
    future = pd.date_range(days[-1] + pd.Timedelta(days=1), periods=14, freq="D")
    frames = []
    for store, level, growth in (("north", 520, 0.25), ("south", 310, 0.10), ("east", 140, 0.40)):
        promo = (rng.random(len(days)) < 0.12).astype(int)
        trend = 1 + growth * np.arange(len(days)) / len(days)
        noise = rng.normal(1, 0.06, len(days))
        sales = level * trend * weekly * holiday * (1 + 0.35 * promo) * noise
        frames.append(
            pd.DataFrame(
                {
                    "date": days.strftime("%Y-%m-%d"),
                    "store": store,
                    "promo": promo,
                    "sales": sales.round(),
                }
            )
        )
        planned = (future.dayofweek == 4).astype(int)  # promotions planned for the next two Fridays
        frames.append(
            pd.DataFrame(
                {
                    "date": future.strftime("%Y-%m-%d"),
                    "store": store,
                    "promo": planned,
                    "sales": np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    for name, frame in {
        "churn.csv": churn(),
        "house_prices.csv": house_prices(),
        "daily_sales.csv": daily_sales(),
        "customers.csv": customers(),
        "transactions.csv": transactions(),
        "store_sales.csv": store_sales(),  # last, so the files above stay identical
    }.items():
        frame.to_csv(HERE / name, index=False)
        print(f"wrote {name}: {len(frame):,} rows × {frame.shape[1]} columns")
