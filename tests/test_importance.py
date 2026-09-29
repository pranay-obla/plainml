"""plainml importance: methods, consensus, the noise-column bias fix, and the report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plainml
from plainml.errors import PlainMLError
from plainml.importance import METHODS, resolve_methods


@pytest.fixture(scope="module")
def signal_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two real drivers, a duplicate of one of them, a noise number and a noise text column."""
    rng = np.random.default_rng(0)
    n = 600
    a, b = rng.normal(size=n), rng.normal(size=n)
    words = ["red", "blue", "green", "fast", "slow", "big", "small", "new", "old", "hot"]
    frame = pd.DataFrame(
        {
            "driver_a": a,
            "driver_b": b,
            "copy_of_a": a * 2 + rng.normal(0, 0.01, n),
            "noise": rng.normal(size=n),
            "noise_text": [" ".join(rng.choice(words, 6)) for _ in range(n)],
            "y": np.where(a + b + rng.normal(0, 0.5, n) > 0, "yes", "no"),
        }
    )
    path = tmp_path_factory.mktemp("imp") / "signal.csv"
    frame.to_csv(path, index=False)
    return path


def test_default_methods_find_the_drivers(signal_csv: Path) -> None:
    result = plainml.feature_importance(signal_csv, "y", verbose=False)
    top = result.table["column"].head(3).tolist()
    assert set(top) == {"driver_a", "driver_b", "copy_of_a"}
    assert "noise_text" not in result.selected and "noise" not in result.selected
    decisions = result.details["boruta"]["decisions"]
    assert decisions["driver_a"] == "confirmed" and decisions["noise"] != "confirmed"
    groups = result.details["redundancy"]["groups"]
    assert any({"driver_a", "copy_of_a"} <= set(g) for g in groups)
    for name in ("rankings.csv", "selected_columns.txt", "report.html", "run.json"):
        assert (result.run_dir / name).is_file()


def test_tree_importance_ignores_wide_noise(signal_csv: Path) -> None:
    """A 100-word text column used to top tree-based rankings by sheer width."""
    result = plainml.feature_importance(
        signal_csv, "y", methods=["random_forest", "extra_trees", "rfe"], save=False, verbose=False
    )
    table = result.table.set_index("column")
    for method in ("random_forest", "extra_trees", "rfe"):
        assert table.loc["noise_text", f"{method}_rank"] >= 4, method


def test_every_method_runs(signal_csv: Path) -> None:
    methods = [m.key for m in METHODS if m.key != "shap"]
    result = plainml.feature_importance(signal_csv, "y", methods=methods, save=False, verbose=False)
    assert set(result.methods) == set(methods)
    assert result.details["exhaustive"]["tried"] == 2**5 - 1


def test_regression_and_reduced_output(diabetes_csv: Path, tmp_path: Path) -> None:
    table = plainml.select_features(
        diabetes_csv, "target", k=3, output=tmp_path / "r.csv", verbose=False
    )
    assert table["selected"].sum() == 3
    assert list(pd.read_csv(tmp_path / "r.csv").columns)[-1] == "target"


def test_method_names() -> None:
    assert resolve_methods(["efs", "sfs", "lasso"], "classification") == [
        "exhaustive",
        "forward",
        "l1",
    ]
    assert "chi2" not in resolve_methods(["all"], "regression")
    with pytest.raises(PlainMLError, match="Unknown importance method"):
        resolve_methods(["magic"], "classification")


def test_column_count_ignores_gains_within_the_noise() -> None:
    """A slightly higher score with more columns only counts if it beats the scores' wobble."""
    from plainml.importance import enough_columns
    from plainml.metrics import metrics_for, resolve_metric

    f1 = resolve_metric(
        "f1", "classification", metrics_for("classification", pd.Series(["a", "b"]))
    )
    points = [(1, 0.7259), (2, 0.7111), (3, 0.8647), (5, 0.8748)]
    assert enough_columns(points, f1) == 5  # a fixed margin alone lets noise columns in
    assert enough_columns(points, f1, [0.003, 0.0129, 0.0204, 0.0078]) == 3


def test_methods_do_not_depend_on_global_random_state(signal_csv: Path) -> None:
    """Every method is seeded: results can't change with numpy's global random state."""
    from plainml.importance import _l1, build_workspace
    from plainml.io import load_data
    from plainml.schema import infer_schema

    frame = load_data(signal_csv)
    X, y = frame.drop(columns="y"), frame["y"]
    schema, _ = infer_schema(X)
    ws = build_workspace(X, y, schema, "classification", 42)
    np.random.seed(0)
    first = _l1(ws)[0]
    np.random.seed(1)
    second = _l1(ws)[0]
    pd.testing.assert_series_equal(first, second)
