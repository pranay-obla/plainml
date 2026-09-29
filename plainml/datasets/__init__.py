"""Small example datasets that ship with plainml, to try it without your own data.

They're the same synthetic files as the repository's ``examples/`` folder (no real people),
deliberately a bit messy, like real exports. The website offers each one in a click, with the
task and settings it suits already chosen.

    from plainml.datasets import SAMPLES, path
    path("churn")          # -> Path to churn.csv inside the installed package
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plainml.errors import PlainMLError

HERE = Path(__file__).parent


@dataclass(frozen=True)
class Sample:
    key: str
    file: str
    title: str
    question: str  # what it answers, in a few words
    about: str  # what's in it, and what to look for
    rows: int
    task: str  # the web app's task that suits it
    options: dict[str, Any] = field(default_factory=dict)  # that task's settings

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "file": self.file,
            "title": self.title,
            "question": self.question,
            "about": self.about,
            "rows": self.rows,
            "task": self.task,
            "options": dict(self.options),
        }


SAMPLES = {
    sample.key: sample
    for sample in (
        Sample(
            "churn",
            "churn.csv",
            "Customer churn",
            "Which subscribers will cancel?",
            "Made-up subscribers, 18% of whom cancel. Look for how support calls and monthly "
            "contracts raise the risk.",
            1500,
            "train",
            {"target": "churned", "speed": "quick"},
        ),
        Sample(
            "house_prices",
            "house_prices.csv",
            "House prices",
            "What will a house sell for?",
            "Made-up house sales. Size, neighborhood and condition drive the price.",
            1200,
            "train",
            {"target": "price", "speed": "quick"},
        ),
        Sample(
            "store_sales",
            "store_sales.csv",
            "Store sales",
            "What will each store sell in the next two weeks?",
            "Three made-up stores. Promotions lift sales by about a third, and the last two "
            "weeks hold planned promotions but no sales yet.",
            1680,
            "forecast",
            {
                "target": "sales",
                "group": "store",
                "inputs": ["promo"],
                "country": "US",
                "horizon": 14,
            },
        ),
        Sample(
            "daily_sales",
            "daily_sales.csv",
            "Daily sales",
            "How many units will sell next month?",
            "Two and a half years of one shop's daily sales, with a trend, busy weekends and a "
            "yearly cycle.",
            912,
            "forecast",
            {"target": "units_sold", "horizon": 30},
        ),
        Sample(
            "customers",
            "customers.csv",
            "Customer segments",
            "What kinds of customers are there?",
            "Made-up shoppers with three hidden segments: young app users, big-spending store "
            "shoppers and frequent web visitors.",
            600,
            "cluster",
        ),
        Sample(
            "transactions",
            "transactions.csv",
            "Card payments",
            "Which payments look suspicious?",
            "Made-up card payments, 30 of them fraud. The is_fraud column marks those, so you "
            "can see how many the detectors catch.",
            3000,
            "anomaly",
            {"label": "is_fraud"},
        ),
    )
}


def get(key: str) -> Sample:
    """The example dataset called ``key``."""
    if key not in SAMPLES:
        raise PlainMLError(
            f"No example dataset called '{key}'.", hint="Choose one of: " + ", ".join(SAMPLES)
        )
    return SAMPLES[key]


def path(key: str) -> Path:
    """Where the example dataset called ``key`` is, inside the installed package."""
    return HERE / get(key).file
