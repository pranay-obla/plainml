"""Turning raw columns into model-ready numbers, inside a scikit-learn Pipeline.

The whole preprocessing chain is part of every saved model, so a model file accepts raw
rows (strings, blanks, dates and all) and applies exactly the transformations it was
trained with. That's also what prevents test data leaking into training: every step is
fit inside cross-validation, never on the full dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from plainml.schema import Schema, coerce_numeric, guess_date_format, parse_dates

DATE_PARTS = ("year", "month", "day", "dayofweek", "hour", "is_weekend")
MISSING_CATEGORY = "(missing)"


def _category_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def category_strings(series: pd.Series) -> pd.Series:
    """Normalise category values so 1, 1.0 and '1' are the same category."""
    return series.astype(object).map(_category_value).astype(object)


class ColumnPrep(TransformerMixin, BaseEstimator):
    """Bring raw input columns into a consistent shape before encoding.

    - selects the training columns (extra columns are ignored, missing ones become blanks)
    - converts numbers stored as text ("$1,200") to floats
    - normalises categories to strings
    - expands each date column into year / month / day / weekday / hour / weekend
    """

    def __init__(
        self,
        numeric: tuple[str, ...] = (),
        categorical: tuple[str, ...] = (),
        datetime: tuple[str, ...] = (),
        text: tuple[str, ...] = (),
    ):
        self.numeric = numeric
        self.categorical = categorical
        self.datetime = datetime
        self.text = text

    @property
    def input_columns(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.datetime, *self.text]

    @property
    def numeric_out(self) -> list[str]:
        return [*self.numeric, *(f"{c}__{p}" for c in self.datetime for p in DATE_PARTS)]

    @property
    def output_columns(self) -> list[str]:
        return [*self.numeric_out, *self.categorical, *self.text]

    def origin(self, prepared: str) -> str:
        """Map a prepared column (e.g. 'signup__month') back to its raw column."""
        for column in self.datetime:
            if prepared.startswith(f"{column}__") and prepared[len(column) + 2 :] in DATE_PARTS:
                return column
        return prepared

    def _frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        array = np.asarray(X, dtype=object)
        if array.ndim == 2 and array.shape[1] == len(self.input_columns):
            return pd.DataFrame(array, columns=self.input_columns)
        raise ValueError(
            "plainml models expect a pandas DataFrame with the original column names "
            f"({', '.join(self.input_columns[:10])}...)."
        )

    def fit(self, X: Any, y: Any = None) -> ColumnPrep:
        frame = self._frame(X)
        self.date_formats_ = {
            c: guess_date_format(frame[c]) if c in frame else None for c in self.datetime
        }
        self.feature_names_in_ = np.asarray(self.input_columns, dtype=object)
        self.n_features_in_ = len(self.input_columns)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = self._frame(X)
        blank = pd.Series(np.nan, index=frame.index, dtype=float)
        out: dict[str, pd.Series] = {}
        for column in self.numeric:
            out[column] = coerce_numeric(frame[column]) if column in frame else blank
        for column in self.datetime:
            if column in frame:
                parsed = parse_dates(frame[column], getattr(self, "date_formats_", {}).get(column))
            else:
                parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
            dayofweek = parsed.dt.dayofweek.astype(float)
            parts = {
                "year": parsed.dt.year.astype(float),
                "month": parsed.dt.month.astype(float),
                "day": parsed.dt.day.astype(float),
                "dayofweek": dayofweek,
                "hour": parsed.dt.hour.astype(float),
                "is_weekend": (dayofweek >= 5).astype(float).where(dayofweek.notna()),
            }
            for part in DATE_PARTS:
                out[f"{column}__{part}"] = parts[part]
        for column in self.categorical:
            out[column] = (
                category_strings(frame[column]) if column in frame else blank.astype(object)
            )
        for column in self.text:
            values = frame.get(column, blank)
            out[column] = values.astype(object).where(values.notna(), "").astype(str)
        return pd.DataFrame(out, index=frame.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.output_columns, dtype=object)


def build_preprocessor(
    schema: Schema, *, text_features: int = 300, max_categories: int = 25
) -> Pipeline:
    """Prep + encode: impute and scale numbers, one-hot categories, TF-IDF free text."""
    prep = ColumnPrep(
        tuple(schema.numeric), tuple(schema.categorical), tuple(schema.datetime), tuple(schema.text)
    )
    transformers: list[tuple[str, Any, Any]] = []
    if prep.numeric_out:
        numeric = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric, prep.numeric_out))
    if schema.categorical:
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant", fill_value=MISSING_CATEGORY, keep_empty_features=True
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        max_categories=max_categories,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical, list(schema.categorical)))
    for i, column in enumerate(schema.text):
        vectorizer = TfidfVectorizer(
            max_features=text_features, sublinear_tf=True, stop_words="english", dtype=np.float32
        )
        transformers.append((f"text{i}", vectorizer, column))
    encode = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline([("prep", prep), ("encode", encode)])


def make_model_pipeline(schema: Schema, estimator: Any, sampler: Any = None) -> Pipeline:
    """Full pipeline: preprocessing → (optional resampler such as SMOTE) → estimator."""
    preprocess = build_preprocessor(schema)
    if sampler is not None:
        # imbalanced-learn refuses nested pipelines, so the preprocessing steps are inlined
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline([*preprocess.steps, ("balance", sampler), ("model", estimator)])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def preprocess_step(pipeline: Pipeline) -> Pipeline:
    """The fitted prep→encode part of a model pipeline, whether nested or inlined (SMOTE)."""
    steps = pipeline.named_steps
    if "preprocess" in steps:
        return steps["preprocess"]
    return Pipeline([("prep", steps["prep"]), ("encode", steps["encode"])])


def inner_pipeline(model: Any) -> Pipeline | None:
    """Find the preprocess→model pipeline inside wrappers (threshold tuning, log target)."""
    seen = 0
    while model is not None and seen < 5:
        if isinstance(model, Pipeline) and (
            "preprocess" in model.named_steps or "prep" in model.named_steps
        ):
            return model
        model = getattr(model, "estimator_", None) or getattr(model, "regressor_", None)
        seen += 1
    return None


def describe_features(preprocessor: Pipeline) -> list[tuple[str, str]]:
    """For each encoded feature: (raw column it came from, human-readable label)."""
    prep: ColumnPrep = preprocessor.named_steps["prep"]
    encode: ColumnTransformer = preprocessor.named_steps["encode"]
    described: list[tuple[str, str]] = []
    for name, transformer, columns in encode.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if name == "num":
            imputer = transformer.named_steps["impute"]
            for column in columns:
                origin = prep.origin(column)
                label = column if origin == column else f"{origin} ({column.split('__')[-1]})"
                described.append((origin, label))
            if imputer.indicator_ is not None:
                for index in imputer.indicator_.features_:
                    origin = prep.origin(columns[index])
                    described.append((origin, f"{columns[index]} is missing"))
        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            tokens = [f"\x00{i}\x00" for i in range(len(columns))]
            for encoded in onehot.get_feature_names_out(tokens):
                _, index, value = encoded.split("\x00", 2)
                column = columns[int(index)]
                value = value.lstrip("_")
                shown = "other (rare values)" if value == "infrequent_sklearn" else value
                described.append((column, f"{column} = {shown}"))
        else:
            for token in transformer.get_feature_names_out():
                described.append((columns, f"{columns} contains '{token}'"))
    return described
