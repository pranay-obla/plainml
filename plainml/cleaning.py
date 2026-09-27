"""``plainml clean``: fix the everyday problems in a dataset and save a tidy copy.

Every change is listed, so nothing happens silently. Training doesn't need a cleaned
file (its preprocessing handles blanks, text numbers and dates), but a clean copy is
useful for other tools, for sharing, and for spotting data-entry problems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from rich.table import Table

from plainml.console import console, esc, heading, note, success, warn
from plainml.errors import PlainMLError, find_column
from plainml.io import load_data, save_table
from plainml.schema import (
    Kind,
    coerce_numeric,
    guess_date_format,
    infer_column,
    is_text_like,
    parse_dates,
)

MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "n.a.",
    "nan",
    "null",
    "none",
    "nil",
    "?",
    "-",
    "--",
    "---",
    "missing",
    "#n/a",
    "#na",
    "undefined",
    "unknown",
    "not available",
    ".",
}
OUTLIER_MODES = ("none", "clip", "remove")
MAX_ENCODED_CATEGORIES = 20


def _is_missing_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in MISSING_TOKENS


def clean(
    data: Any,
    *,
    output: str | Path | None = None,
    target: str | None = None,
    drop_duplicates: bool = True,
    impute: bool = False,
    outliers: str = "none",
    encode: bool = False,
    drop_ids: bool = True,
    verbose: bool = True,
    **load_options: Any,
) -> pd.DataFrame:
    """Return a cleaned copy of ``data`` (and save it to ``output`` if given)."""
    from plainml.console import quiet

    if outliers not in OUTLIER_MODES:
        raise PlainMLError(f"outliers must be one of {', '.join(OUTLIER_MODES)}.")
    df = load_data(data, **load_options)
    rows_before, cols_before = df.shape
    actions: list[tuple[str, str]] = []
    target_column = find_column(target, df.columns, "Target column") if target else None

    # 1. column names with stray spaces
    renamed = {c: c.strip() for c in df.columns if c != c.strip() and c.strip() not in df.columns}
    if renamed:
        df = df.rename(columns=renamed)
        if target_column in renamed:
            target_column = renamed[target_column]
        actions.append(("Column names", f"trimmed spaces from {len(renamed)} name(s)"))

    # 2. text cells: trim spaces, turn "N/A", "?", "-" etc. into real blanks
    blanks = 0
    for column in df.columns:
        if is_text_like(df[column]) and not isinstance(df[column].dtype, pd.CategoricalDtype):
            series = df[column].astype(object)
            text_mask = series.map(lambda v: isinstance(v, str))
            stripped = series.where(
                ~text_mask, series[text_mask].str.strip() if text_mask.any() else series
            )
            token_mask = stripped.map(_is_missing_token)
            blanks += int(token_mask.sum())
            df[column] = stripped.where(~token_mask, np.nan)
    if blanks:
        actions.append(
            (
                "Blanks",
                f"turned {blanks:,} placeholder values ('N/A', '?', '-', …) into real blanks",
            )
        )

    # 3. empty rows and columns
    empty_columns = [c for c in df.columns if df[c].isna().all() and c != target_column]
    if empty_columns:
        df = df.drop(columns=empty_columns)
        actions.append(("Empty columns", "removed " + _list(empty_columns)))
    empty_rows = df.isna().all(axis=1)
    if empty_rows.any():
        df = df.loc[~empty_rows]
        actions.append(("Empty rows", f"removed {int(empty_rows.sum()):,}"))

    # 4. types: numbers and dates stored as text
    converted_numbers, converted_dates = [], []
    for column in df.columns:
        if not is_text_like(df[column]):
            continue
        info = infer_column(df[column])
        if info.kind == Kind.NUMERIC and info.stored_as_text:
            df[column] = coerce_numeric(df[column])
            converted_numbers.append(column)
        elif info.kind == Kind.DATETIME:
            df[column] = parse_dates(df[column], guess_date_format(df[column]))
            converted_dates.append(column)
    if converted_numbers:
        actions.append(("Numbers", "converted text to numbers in " + _list(converted_numbers)))
    if converted_dates:
        actions.append(("Dates", "parsed dates in " + _list(converted_dates)))

    # 5. inconsistent spelling of the same category ("Male", "male ", "MALE")
    unified = []
    for column in df.columns:
        if not is_text_like(df[column]):
            continue
        values = df[column].dropna().astype(str)
        if values.empty or values.nunique() > 500:
            continue
        keys = values.str.strip().str.lower()
        groups = values.groupby(keys)
        variants = groups.nunique()
        if (variants > 1).any():
            canonical = groups.agg(lambda s: s.value_counts().index[0])
            df[column] = df[column].map(
                lambda v, c=canonical: c.get(str(v).strip().lower(), v) if isinstance(v, str) else v
            )
            unified.append(f"{column} ({int((variants > 1).sum())} value(s))")
    if unified:
        actions.append(("Spelling", "unified capitalisation variants in " + _list(unified)))

    # 6. rows without a target
    if target_column:
        missing_target = df[target_column].isna()
        if missing_target.any():
            df = df.loc[~missing_target]
            actions.append(
                ("Target", f"removed {int(missing_target.sum()):,} rows with no {target_column}")
            )

    # 7. duplicates
    if drop_duplicates:
        try:
            duplicated = df.duplicated()
        except TypeError:
            duplicated = pd.Series(False, index=df.index)
        if duplicated.any():
            df = df.loc[~duplicated]
            actions.append(("Duplicates", f"removed {int(duplicated.sum()):,} repeated rows"))

    # 8. ID-like and constant columns
    if drop_ids:
        useless = []
        for column in df.columns:
            if column == target_column:
                continue
            kind = infer_column(df[column]).kind
            if kind in (Kind.ID, Kind.CONSTANT):
                useless.append((column, "an ID" if kind == Kind.ID else "a single value"))
        if useless:
            df = df.drop(columns=[column for column, _ in useless])
            actions.append(
                ("Unhelpful columns", "removed " + _list([f"{c} ({why})" for c, why in useless]))
            )

    numeric_columns = [
        c
        for c in df.columns
        if c != target_column and ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])
    ]

    # 9. outliers (1.5 × IQR), skipping yes/no and low-cardinality columns
    if outliers != "none":
        flagged = pd.Series(False, index=df.index)
        clipped = []
        for column in numeric_columns:
            series = df[column]
            if series.nunique() <= 10:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outside = (series < low) | (series > high)
            if not outside.any():
                continue
            if outliers == "clip":
                df[column] = series.clip(low, high)
                clipped.append(f"{column} ({int(outside.sum())})")
            else:
                flagged |= outside
        if outliers == "clip" and clipped:
            actions.append(("Outliers", "clipped extreme values in " + _list(clipped)))
        if outliers == "remove" and flagged.any():
            share = flagged.mean()
            if share > 0.1:
                warn(
                    f"Removing outliers would drop {share:.0%} of rows; skipped. Try --outliers clip instead."
                )
            else:
                df = df.loc[~flagged]
                actions.append(
                    ("Outliers", f"removed {int(flagged.sum()):,} rows with extreme values")
                )

    # 10. fill blanks
    if impute:
        filled = []
        for column in df.columns:
            if column == target_column or not df[column].isna().any():
                continue
            count = int(df[column].isna().sum())
            if column in numeric_columns:
                df[column] = df[column].fillna(df[column].median())
            elif ptypes.is_datetime64_any_dtype(df[column]):
                continue
            else:
                mode = df[column].mode(dropna=True)
                if mode.empty:
                    continue
                df[column] = df[column].fillna(mode.iloc[0])
            filled.append(f"{column} ({count})")
        if filled:
            actions.append(("Blanks filled", "median / most common value in " + _list(filled)))

    # 11. one-hot encoding
    if encode:
        categorical = [
            c
            for c in df.columns
            if c != target_column and (is_text_like(df[c]) or ptypes.is_bool_dtype(df[c]))
        ]
        for column in categorical:
            top = df[column].value_counts().index[:MAX_ENCODED_CATEGORIES]
            if df[column].nunique() > MAX_ENCODED_CATEGORIES:
                df[column] = df[column].where(df[column].isin(top) | df[column].isna(), "other")
        if categorical:
            df = pd.get_dummies(df, columns=categorical, dtype=int)
            actions.append(("Encoding", "one-hot encoded " + _list(categorical)))

    df = df.reset_index(drop=True)
    if verbose:
        with quiet(False):
            _render(actions, (rows_before, cols_before), df.shape)
    if output:
        path = save_table(df, output)
        if verbose:
            success(f"Saved the cleaned data to {esc(path)}")
    return df


def _list(items: list[str], limit: int = 6) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def _render(
    actions: list[tuple[str, str]], before: tuple[int, int], after: tuple[int, int]
) -> None:
    heading("Cleaning")
    if not actions:
        note("Nothing needed fixing.")
    else:
        table = Table(box=None, header_style="muted", pad_edge=False, show_header=False)
        table.add_column("Step", style="bold", no_wrap=True)
        table.add_column("What changed", overflow="fold")
        for step, description in actions:
            table.add_row(step, esc(description))
        console.print(table)
    console.print(
        f"[muted]{before[0]:,} rows × {before[1]} columns → [/][bold]{after[0]:,} rows × {after[1]} columns[/]"
    )
