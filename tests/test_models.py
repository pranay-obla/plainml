"""Every registered model trains, the tiers and conditions work, and PyTorch behaves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_diabetes, make_classification

import plainml
from plainml.registry import MODELS, REGISTRY, select_models
from plainml.training import (
    Candidate,
    TrainOptions,
    build_estimator,
    cross_validate_candidate,
    prepare,
)


@pytest.fixture(scope="module")
def contexts() -> dict:
    X, y = make_classification(240, 6, n_informative=4, random_state=0)
    classification = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    classification["y"] = np.where(y == 1, "yes", "no")
    regression = load_diabetes(as_frame=True).frame.rename(columns={"target": "y"}).head(240)
    return {
        "classification": prepare(classification, "y", TrainOptions(cv=3, n_jobs=1)),
        "regression": prepare(regression, "y", TrainOptions(cv=3, n_jobs=1)),
    }


@pytest.mark.parametrize("key", [spec.key for spec in MODELS])
def test_every_model_trains(key: str, contexts: dict) -> None:
    spec = REGISTRY[key]
    if not spec.available:
        pytest.skip(f"{spec.requires} not installed")
    for task, ctx in contexts.items():
        if task not in spec.tasks:
            continue
        candidate = Candidate(key, spec.name(task))
        candidate.template = build_estimator(ctx, spec)
        cross_validate_candidate(ctx, candidate)
        score = candidate.scores[ctx.metric.key]
        assert np.isfinite(score), f"{key} gave no {ctx.metric.key} for {task}"


def test_tiers_and_conditions() -> None:
    default = {s.key for s in select_models("regression").specs}
    thorough = {
        s.key for s in select_models("regression", thorough=True, y=np.array([1.0, 2.0])).specs
    }
    assert "bayesian_ridge" in default and "adaboost" not in default
    assert {"adaboost", "poisson", "gamma"} <= thorough
    negative = select_models("regression", thorough=True, y=np.array([-5.0, 2.0]))
    assert "poisson" not in {s.key for s in negative.specs}
    assert "negative" in negative.skipped["poisson"]
    named = select_models("classification", include=["qda"])  # extra models run when named
    assert [s.key for s in named.specs] == ["baseline", "qda"]


def test_thorough_adds_a_stacked_ensemble() -> None:
    X, y = make_classification(200, 5, random_state=1)
    frame = pd.DataFrame(X, columns=list("abcde"))
    frame["y"] = y
    result = plainml.train(
        frame,
        target="y",
        models=["linear", "tree", "nb", "lda"],
        thorough=True,
        cv=3,
        verbose=False,
        report=False,
    )
    assert {"ensemble", "stacking"} <= set(result.leaderboard["key"])


def test_torch_is_a_proper_estimator() -> None:
    pytest.importorskip("torch")
    from plainml.deep import TorchMLPClassifier, TorchMLPRegressor

    params = TorchMLPClassifier().get_params()
    assert {"width", "depth", "learning_rate", "class_weight"} <= set(params)
    tuned = clone(TorchMLPClassifier(width=32, epochs=5))
    assert tuned.width == 32 and tuned.epochs == 5
    X = np.random.default_rng(0).normal(size=(120, 4)).astype("float32")
    y_reg = X[:, 0] * 3 + 1
    regressor = TorchMLPRegressor(epochs=50).fit(X, y_reg)
    assert np.corrcoef(regressor.predict(X), y_reg)[0, 1] > 0.8


def test_torch_after_lightgbm_does_not_hang() -> None:
    """Two OpenMP runtimes in one process used to deadlock PyTorch on macOS."""
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    X, y = make_classification(200, 5, random_state=2)
    frame = pd.DataFrame(X, columns=list("abcde"))
    frame["y"] = y
    result = plainml.train(
        frame,
        target="y",
        models=["lightgbm", "torch"],
        ensemble=False,
        cv=3,
        verbose=False,
        report=False,
    )
    assert set(result.leaderboard.query("status == 'ok'")["key"]) >= {"lightgbm", "torch"}
