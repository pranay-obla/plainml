"""Unit tests for loading, column typing, tasks, metrics, preprocessing and helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plainml.console import esc, fmt_num, fmt_value, parse_duration
from plainml.errors import PlainMLError, did_you_mean, find_column
from plainml.io import fingerprint, load_data, save_table
from plainml.metrics import metrics_for, resolve_metric
from plainml.preprocessing import (
    ColumnPrep,
    build_preprocessor,
    category_strings,
    describe_features,
)
from plainml.schema import (
    Kind,
    coerce_numeric,
    guess_date_format,
    infer_column,
    infer_schema,
    parse_dates,
)
from plainml.tasks import (
    CLASSIFICATION,
    MULTI_REGRESSION,
    MULTILABEL,
    REGRESSION,
    detect_task,
    prepare_target,
)

# --- loading ---------------------------------------------------------------------------------


def test_load_formats(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    for name in ("d.csv", "d.tsv", "d.xlsx", "d.json", "d.jsonl"):
        path = save_table(frame, tmp_path / name)
        loaded = load_data(path)
        assert list(loaded.columns) == ["a", "b"], name
        assert len(loaded) == 3, name


def test_load_semicolon_csv(tmp_path: Path) -> None:
    path = tmp_path / "euro.csv"
    path.write_text("a;b\n1,5;x\n2,5;y\n")
    assert list(load_data(path).columns) == ["a", "b"]


def test_load_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = save_table(pd.DataFrame({"a": [1, 2]}), tmp_path / "d.parquet")
    assert load_data(path)["a"].tolist() == [1, 2]


def test_load_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as connection:
        pd.DataFrame({"a": [1, 2, 3]}).to_sql("orders", connection, index=False)
    assert len(load_data(f"sqlite:///{db}", table="orders")) == 3
    assert len(load_data(f"sqlite:///{db}", query="SELECT * FROM orders WHERE a > 1")) == 2
    with pytest.raises(PlainMLError, match="query or a table"):
        load_data(f"sqlite:///{db}")


def test_load_errors_are_friendly(tmp_path: Path) -> None:
    (tmp_path / "iris.csv").write_text("a\n1\n")
    with pytest.raises(PlainMLError) as info:
        load_data(tmp_path / "irs.csv")
    assert "iris.csv" in (info.value.hint or "")
    (tmp_path / "dup.csv").write_text("a,b\n1,2\n")
    with pytest.raises(PlainMLError, match="Don't know how"):
        load_data(tmp_path / "file.xyz") if (tmp_path / "file.xyz").write_text("x") else None
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(PlainMLError):
        load_data(tmp_path / "empty.csv")


def test_sample_and_fingerprint() -> None:
    frame = pd.DataFrame({"a": range(100)})
    assert len(load_data(frame, sample=10)) == 10
    assert len(load_data(frame, sample=0.25)) == 25
    assert fingerprint(frame) == fingerprint(frame.copy())
    assert fingerprint(frame) != fingerprint(frame.assign(a=frame["a"] + 1))


# --- column types ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        ([1.5, 2.5, 3.5, None] * 10, Kind.NUMERIC),
        (["$1,200", "$950", "$3,400", "$12"] * 10, Kind.NUMERIC),
        (["45%", "50%", "12%", None] * 10, Kind.NUMERIC),
        (["00501", "02134", "90210", "10001"] * 10, Kind.CATEGORICAL),  # zip codes keep their zeros
        (["red", "green", "blue"] * 10, Kind.CATEGORICAL),
        (["2024-01-05", "2024-02-11", "2024-03-09"] * 10, Kind.DATETIME),
        (["13/01/2024", "25/12/2024", "01/02/2024"] * 10, Kind.DATETIME),
        ([f"the customer wrote a long message number {i}" for i in range(40)], Kind.TEXT),
        ([f"user{i}@example.com" for i in range(60)], Kind.ID),
        (["x"] * 30, Kind.CONSTANT),
        ([None] * 30, Kind.EMPTY),
    ],
)
def test_infer_column(values: list, kind: str) -> None:
    assert infer_column(pd.Series(values, name="col")).kind == kind


def test_id_detection() -> None:
    assert infer_column(pd.Series(range(1, 101), name="customer_id")).kind == Kind.ID
    assert infer_column(pd.Series(range(100), name="Unnamed: 0")).kind == Kind.ID
    # a unique number that isn't a counter (prices) stays numeric
    prices = pd.Series(np.random.default_rng(0).permutation(np.arange(1000, 1100)), name="price")
    assert infer_column(prices).kind == Kind.NUMERIC


def test_infer_schema_drop_and_keep() -> None:
    frame = pd.DataFrame({"id": range(50), "a": np.arange(50) % 7, "c": ["k"] * 50})
    schema, _ = infer_schema(frame, keep=["id"], drop=["a"])
    assert "id" in schema.features and "a" not in schema.features
    assert schema.dropped["a"] == "dropped by you" and "c" in schema.dropped


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["01/01/2024", "13/01/2024", "25/12/2024"], ["2024-01-01", "2024-01-13", "2024-12-25"]),
        (["01/02/2024", "12/31/2024"], ["2024-01-02", "2024-12-31"]),
        (["2024-01-05 10:00", "2024-02-06 11:30"], ["2024-01-05", "2024-02-06"]),
        (["Jan 5, 2024", "Feb 13, 2024"], ["2024-01-05", "2024-02-13"]),
    ],
)
def test_date_formats(values: list[str], expected: list[str]) -> None:
    series = pd.Series(values)
    parsed = parse_dates(series, guess_date_format(series))
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == expected


def test_coerce_numeric() -> None:
    assert coerce_numeric(pd.Series(["$1,200", "(50)", "7%", "n/a"])).tolist()[:3] == [
        1200.0,
        -50.0,
        7.0,
    ]


# --- tasks and metrics ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "task"),
    [
        (["yes", "no"] * 10, CLASSIFICATION),
        ([True, False] * 10, CLASSIFICATION),
        ([0, 1, 2] * 10, CLASSIFICATION),
        (list(np.linspace(0, 100, 30)), REGRESSION),
        (list(range(50)), REGRESSION),
    ],
)
def test_detect_task(values: list, task: str) -> None:
    assert detect_task(pd.Series(values, name="y"))[0] == task


def test_detect_task_rejects_dates() -> None:
    with pytest.raises(PlainMLError, match="forecast"):
        detect_task(pd.Series(pd.date_range("2024-01-01", periods=5), name="d"))


def test_prepare_target_cleans_rows() -> None:
    frame = pd.DataFrame({"x": range(9), "y": ["a", "a", "a", "b", "b", "b", "c", None, "a"]})
    kept, y, info = prepare_target(frame, ["y"])
    assert info.task == CLASSIFICATION and info.dropped_rows == 1
    assert "c" not in set(y)  # a class with one example can't be split
    assert len(kept) == len(y) == 7


def test_prepare_multi_targets() -> None:
    frame = pd.DataFrame(
        {
            "x": range(10),
            "a": ["yes", "no"] * 5,
            "b": [0, 1] * 5,
            "r1": np.arange(10.0),
            "r2": np.arange(10.0) * 2,
        }
    )
    _, y, info = prepare_target(frame, ["a", "b"])
    assert info.task == MULTILABEL and info.label_maps["a"] == ["no", "yes"]
    assert set(np.unique(y.to_numpy())) == {0, 1}
    _, _, info = prepare_target(frame, ["r1", "r2"])
    assert info.task == MULTI_REGRESSION


def test_resolve_metric() -> None:
    y = pd.Series([0.0, 1.0, 2.0])
    available = metrics_for(REGRESSION, y)
    assert "mape" not in available  # the target has a zero
    assert resolve_metric("R2_score", REGRESSION, available).key == "r2"
    with pytest.raises(PlainMLError, match="rmse"):
        resolve_metric("mse", REGRESSION, available)


def test_string_label_f1_works() -> None:
    """scikit-learn >= 1.8 rejected macro F1 on 'yes'/'no' labels without pos_label=None."""
    from sklearn.linear_model import LogisticRegression

    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.array(["no"] * 20 + ["yes"] * 20)
    model = LogisticRegression().fit(X, y)
    f1 = metrics_for(CLASSIFICATION, n_classes=2)["f1"]
    assert f1.scorer()(model, X, y) > 0.9


# --- preprocessing ----------------------------------------------------------------------------


def test_column_prep_is_forgiving() -> None:
    train = pd.DataFrame(
        {
            "n": [1.0, 2.0, None],
            "c": [1, 2, 1],
            "d": ["2024-01-01", "2024-06-15", "2024-12-31"],
            "t": ["a b c", None, "d e"],
        }
    )
    prep = ColumnPrep(("n",), ("c",), ("d",), ("t",)).fit(train)
    new = pd.DataFrame(
        {"n": ["$5"], "c": [1.0], "extra": ["ignored"]}
    )  # text number, float category, missing columns
    out = prep.transform(new)
    assert out.loc[0, "n"] == 5.0
    assert out.loc[0, "c"] == "1"  # 1, 1.0 and '1' are the same category
    assert np.isnan(out.loc[0, "d__year"]) and out.loc[0, "t"] == ""
    assert list(out.columns) == prep.output_columns


def test_category_strings() -> None:
    assert category_strings(pd.Series([1, 1.0, "1", True, None])).tolist()[:4] == [
        "1",
        "1",
        "1",
        "True",
    ]


def test_describe_features_matches_width() -> None:
    frame = pd.DataFrame(
        {
            "n": [1.0, None, 3.0, 4.0] * 5,
            "c": list("abcd") * 5,
            "d": pd.date_range("2024-01-01", periods=20).astype(str),
        }
    )
    schema, _ = infer_schema(frame)
    pre = build_preprocessor(schema)
    encoded = pre.fit_transform(frame)
    described = describe_features(pre)
    assert len(described) == encoded.shape[1]
    assert {origin for origin, _ in described} == {"n", "c", "d"}


# --- helpers ----------------------------------------------------------------------------------


def test_parse_duration() -> None:
    assert parse_duration("90") == 90
    assert parse_duration("5m") == 300
    assert parse_duration("1h30m") == 5400
    assert parse_duration(None) is None
    with pytest.raises(PlainMLError):
        parse_duration("soon")


def test_formatting() -> None:
    assert fmt_num(0.93121) == "0.9312"
    assert fmt_num(58.6697) == "58.67"
    assert fmt_num(float("nan")) == "—"
    assert fmt_value(44.0) == "44"
    assert esc("width [cm]") == "width \\[cm]"


def test_find_column() -> None:
    assert find_column(" Price ", ["price", "qty"]) == "price"
    with pytest.raises(PlainMLError) as info:
        find_column("pric", ["price", "qty"])
    assert "Did you mean 'price'" in info.value.message
    assert did_you_mean("xyz", ["price"]) == ""
