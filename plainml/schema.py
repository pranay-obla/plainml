"""Working out what kind of data each column holds.

Every column is classified as one of the ``Kind`` values below. The result (a ``Schema``)
drives preprocessing and is saved with each model, so new data gets exactly the same
treatment at prediction time.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

try:  # pandas >= 2.2
    from pandas.tseries.api import guess_datetime_format
except ImportError:  # pragma: no cover - pandas 2.1
    from pandas._libs.tslibs.parsing import guess_datetime_format


class Kind:
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    CONSTANT = "constant"
    EMPTY = "empty"


USABLE_KINDS = (Kind.NUMERIC, Kind.CATEGORICAL, Kind.DATETIME, Kind.TEXT)

HIGH_CARDINALITY = 50
_NUMERIC_JUNK = re.compile(r"[,\s$€£¥₹%]")
_SAMPLE = 2000


def is_text_like(series: pd.Series) -> bool:
    """True for object, pandas string and categorical columns (pandas 2 and 3)."""
    return (
        ptypes.is_object_dtype(series)
        or ptypes.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def looks_like_id_name(name: str) -> bool:
    lower = name.strip().lower()
    if lower in {"id", "uuid", "guid", "index", "key", "rowid", "row_id", "row id"}:
        return True
    if lower.startswith("unnamed:"):
        return True
    if re.search(r"[_\s\-.](id|uuid|guid|key|number|no)$", lower):
        return True
    if re.search(r"^(id|uuid|guid)[_\s\-.]", lower):
        return True
    return bool(re.search(r"[a-z](ID|Id)$", name.strip()))


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to float, understanding '1,234', '$5', '45%' and similar."""
    if ptypes.is_bool_dtype(series):
        return series.astype(float)
    if ptypes.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.replace(_NUMERIC_JUNK, "", regex=True)
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # accounting negatives: (12)
    return pd.to_numeric(text, errors="coerce").astype(float)


def guess_date_format(series: pd.Series) -> str | None:
    """Return the strftime format that parses the most values, or None ('mixed').

    Candidates come from several values read both month-first and day-first, then each
    is tested on a spread-out sample. Looking at one value isn't enough: '01/02/2024' is
    ambiguous, and guessing month-first would silently drop every day after the 12th.
    """
    non_null = series.dropna()
    if non_null.empty or ptypes.is_datetime64_any_dtype(series):
        return None
    values = non_null.astype(str)
    step = max(1, len(values) // 200)
    sample = values.iloc[::step].iloc[:200]
    candidates: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for value in sample.iloc[:: max(1, len(sample) // 20)]:
            for dayfirst in (False, True):
                try:
                    fmt = guess_datetime_format(value, dayfirst=dayfirst)
                except (TypeError, ValueError):
                    fmt = None
                if fmt and fmt not in candidates:
                    candidates.append(fmt)
        best, best_share = None, 0.0
        for fmt in candidates:  # month-first candidates come first, so they win ties
            share = float(to_datetime(sample, fmt).notna().mean())
            if share > best_share + 1e-9:
                best, best_share = fmt, share
    return best if best_share >= 0.9 else None


def to_datetime(values: Any, fmt: str | None = None) -> Any:
    """pd.to_datetime that never raises: unparseable values become NaT.

    Values with different time zones (e.g. '+05:30' next to 'Z') can't share a column,
    so in that case everything is converted to UTC.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs: dict[str, Any] = {"errors": "coerce", "format": fmt or "mixed"}
        try:
            parsed = pd.to_datetime(values, **kwargs)
        except (ValueError, TypeError):
            parsed = pd.to_datetime(values, utc=True, **kwargs)
        if isinstance(parsed, pd.Series) and not ptypes.is_datetime64_any_dtype(parsed):
            parsed = pd.to_datetime(values, utc=True, **kwargs)  # older pandas returns objects
    return parsed


def parse_dates(series: pd.Series, fmt: str | None = None) -> pd.Series:
    """Parse a column to datetimes (time-zone aware values become UTC); failures become NaT."""
    parsed = series if ptypes.is_datetime64_any_dtype(series) else to_datetime(series, fmt)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed


def _datetime_share(sample: pd.Series) -> float:
    as_text = sample.astype(str)
    if not as_text.str.contains(r"\d").mean() > 0.9:
        return 0.0
    if as_text.str.fullmatch(r"[+-]?\d+(\.\d+)?").mean() > 0.5:
        return 0.0  # plain numbers, not dates
    if as_text.str.len().median() < 6:
        return 0.0  # "1a", "Q1" and similar codes
    fmt = guess_date_format(sample)
    parsed = parse_dates(sample, fmt)
    share = parsed.notna().mean()
    if share < 0.9 and fmt:
        share = parse_dates(sample, None).notna().mean()
    return float(share)


@dataclass
class ColumnInfo:
    name: str
    kind: str
    dtype: str
    n_missing: int
    missing_pct: float
    n_unique: int
    examples: list[str]
    note: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    stored_as_text: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _examples(series: pd.Series, k: int = 3) -> list[str]:
    values = series.dropna().unique()[:k]
    return [str(v)[:40] for v in values]


def infer_column(series: pd.Series) -> ColumnInfo:
    """Classify one column."""
    name = str(series.name)
    n = len(series)
    n_missing = int(series.isna().sum())
    non_null = series.dropna()
    n_non_null = len(non_null)
    try:
        n_unique = int(non_null.nunique())
    except TypeError:  # unhashable values such as lists from JSON
        non_null = non_null.astype(str)
        n_unique = int(non_null.nunique())
    info = ColumnInfo(
        name=name,
        kind=Kind.CATEGORICAL,
        dtype=str(series.dtype),
        n_missing=n_missing,
        missing_pct=(n_missing / n) if n else 0.0,
        n_unique=n_unique,
        examples=_examples(non_null),
    )
    if n_non_null == 0:
        info.kind, info.note = Kind.EMPTY, "every value is missing"
        return info
    if n_unique <= 1:
        info.kind = Kind.CONSTANT
        info.note = "only one value" + (" (plus blanks)" if n_missing else "")
        return info

    unique_ratio = n_unique / n_non_null

    if ptypes.is_datetime64_any_dtype(series):
        info.kind = Kind.DATETIME
        info.stats = {"min": str(non_null.min()), "max": str(non_null.max())}
        return info

    if ptypes.is_bool_dtype(series):
        info.kind = Kind.CATEGORICAL
        info.stats = _top_values(non_null)
        return info

    numeric = None
    if ptypes.is_numeric_dtype(series):
        numeric = non_null.astype(float)
    elif is_text_like(series):
        sample = non_null.sample(min(_SAMPLE, n_non_null), random_state=0)
        converted = coerce_numeric(sample)
        if converted.notna().mean() >= 0.95 and not sample.astype(str).str.fullmatch(r"0\d+").any():
            numeric = coerce_numeric(non_null).dropna()
            info.stored_as_text = True
            info.note = "numbers stored as text; converted"

    if numeric is not None:
        integer_valued = bool(np.all(np.mod(numeric, 1) == 0))
        # A row counter (0, 1, 2, … or 1, 2, 3, …) is an ID; 1960, 1961, … (years) is not.
        is_counter = (
            integer_valued
            and unique_ratio == 1.0
            and n_non_null > 20
            and float(numeric.iloc[0]) in (0.0, 1.0)
            and numeric.is_monotonic_increasing
            and bool((numeric.diff().dropna() == 1).all())
        )
        if (looks_like_id_name(name) and integer_valued and unique_ratio >= 0.95) or is_counter:
            info.kind, info.note = Kind.ID, "a row ID: a different value in every row"
            return info
        info.kind = Kind.NUMERIC
        info.stats = {
            "mean": float(numeric.mean()),
            "std": float(numeric.std()) if n_non_null > 1 else 0.0,
            "min": float(numeric.min()),
            "median": float(numeric.median()),
            "max": float(numeric.max()),
        }
        return info

    as_text = non_null.astype(str)
    sample = as_text.sample(min(_SAMPLE, n_non_null), random_state=0)
    if _datetime_share(sample) >= 0.9:
        info.kind = Kind.DATETIME
        parsed = parse_dates(sample, guess_date_format(sample)).dropna()
        info.stats = {"min": str(parsed.min()), "max": str(parsed.max())}
        return info

    avg_words = float(sample.str.split().str.len().mean())
    if avg_words >= 4 and unique_ratio >= 0.5:
        info.kind = Kind.TEXT
        info.note = f"free text (about {avg_words:.0f} words per value)"
        info.stats = {"avg_words": avg_words}
        return info
    if (unique_ratio >= 0.95 and n_non_null >= 30) or (
        looks_like_id_name(name) and unique_ratio >= 0.8
    ):
        info.kind = Kind.ID
        info.note = "an identifier: almost every value is different"
        return info

    info.kind = Kind.CATEGORICAL
    info.stats = _top_values(as_text)
    if n_unique > HIGH_CARDINALITY:
        info.note = f"{n_unique} different values; only the most common are used"
    return info


def _top_values(series: pd.Series, k: int = 5) -> dict[str, Any]:
    counts = series.astype(str).value_counts().head(k)
    return {"top": {str(key): int(value) for key, value in counts.items()}}


@dataclass
class Schema:
    """Which columns feed the model, grouped by how they're preprocessed."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        used = set(self.numeric) | set(self.categorical) | set(self.datetime) | set(self.text)
        return [c for c in self.order if c in used]

    def kind_of(self, column: str) -> str | None:
        for kind in USABLE_KINDS:
            if column in getattr(self, kind):
                return kind
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schema:
        return cls(**{k: data.get(k, v) for k, v in asdict(cls()).items()})


def infer_schema(
    df: pd.DataFrame,
    exclude: list[str] | tuple[str, ...] = (),
    drop: list[str] | tuple[str, ...] = (),
    keep: list[str] | tuple[str, ...] = (),
) -> tuple[Schema, list[ColumnInfo]]:
    """Classify every column of ``df`` (except ``exclude``, e.g. the targets).

    ``drop`` columns are always removed; ``keep`` columns are used even when they look
    like IDs or constants.
    """
    schema = Schema(order=[c for c in df.columns if c not in exclude])
    infos: list[ColumnInfo] = []
    for column in schema.order:
        info = infer_column(df[column])
        infos.append(info)
        if column in drop:
            schema.dropped[column] = "dropped by you"
            continue
        kind = info.kind
        if kind in (Kind.ID, Kind.CONSTANT, Kind.EMPTY):
            if column not in keep or kind == Kind.EMPTY:
                schema.dropped[column] = info.note
                continue
            kind = Kind.NUMERIC if ptypes.is_numeric_dtype(df[column]) else Kind.CATEGORICAL
        getattr(schema, kind).append(column)
    return schema, infos
