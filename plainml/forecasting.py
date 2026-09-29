"""``plainml forecast``: predict a value over time (sales, demand, visitors...).

The series is put on a regular calendar, then each model is *backtested*: it forecasts
several past periods using only the data before them, exactly as it would forecast the
future. Two simple baselines ("same as last period", "same as last season") keep the
fancier models honest. The winner forecasts ahead with an 80% range built from its
backtest errors.

Beyond a single series:

- ``group``: one forecast per store/product/region. The model type is chosen by
  backtesting on the largest series (with a scale-free error, so big series don't
  dominate), then fitted to each series separately.
- ``inputs``: other columns known in advance (a promotion flag, a planned price). Their
  future values come from rows after the last known target value (in the same file, or
  a ``future`` file); if none are given, they're held at their last values.
- ``country``: public holidays as features (needs the ``holidays`` package).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plainml import __version__
from plainml.card import CARD_FILE, write_model_card
from plainml.console import (
    console,
    esc,
    fmt_num,
    fmt_pct,
    fmt_value,
    heading,
    info,
    note,
    quiet,
    warn,
)
from plainml.errors import PlainMLError, did_you_mean, find_column, is_installed, require
from plainml.io import describe_source, fingerprint, load_data, save_table, source_stem
from plainml.runs import (
    EVALUATION_FILE,
    LEADERBOARD_FILE,
    MODEL_FILE,
    REPORT_FILE,
    RUN_FILE,
    create_run_dir,
    environment,
    write_json,
)
from plainml.schema import Kind, coerce_numeric, guess_date_format, infer_column, parse_dates
from plainml.tasks import FORECAST

SEASON = {"min": 60, "h": 24, "D": 7, "B": 5, "W": 52, "M": 12, "Q": 4, "Y": 1}
HORIZON = {"min": 60, "h": 48, "D": 30, "B": 20, "W": 12, "M": 12, "Q": 4, "Y": 3}
UNIT = {
    "min": "minute",
    "h": "hour",
    "D": "day",
    "B": "business day",
    "W": "week",
    "M": "month",
    "Q": "quarter",
    "Y": "year",
}
MODELS = {
    "naive": "Same as last period",
    "seasonal_naive": "Same as last season",
    "ridge": "Ridge regression",
    "rf": "Random forest",
    "histgb": "Gradient boosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
BASELINES = ("naive", "seasonal_naive")
MAX_POINTS = 10_000
MAX_GROUPS = 500
SELECTION_GROUPS = 8  # series used to choose the model type in group mode
FORECAST_FILE = "forecast.csv"
Z80 = 1.2816


def base_unit(freq: str) -> str:
    offset = pd.tseries.frequencies.to_offset(freq)
    checks = [
        (pd.offsets.Minute, "min"),
        (pd.offsets.Hour, "h"),
        (pd.offsets.BusinessDay, "B"),
        (pd.offsets.Day, "D"),
        (pd.offsets.Week, "W"),
        ((pd.offsets.MonthEnd, pd.offsets.MonthBegin), "M"),
        ((pd.offsets.QuarterEnd, pd.offsets.QuarterBegin), "Q"),
        ((pd.offsets.YearEnd, pd.offsets.YearBegin), "Y"),
    ]
    for kind, unit in checks:
        if isinstance(offset, kind):
            return unit
    return "D"


def _infer_freq(dates: pd.DatetimeIndex) -> str:
    unique = dates.unique().sort_values()
    if len(unique) >= 3:
        guess = pd.infer_freq(unique)
        if guess:
            return guess
    median = pd.Series(unique).diff().dropna().median()
    if pd.isna(median):
        raise PlainMLError("Need at least 3 distinct dates to work out the time step.")
    options = [
        ("1min", "min"),
        ("1h", "h"),
        ("1D", "D"),
        ("7D", "W"),
        ("30D", "MS"),
        ("91D", "QS"),
        ("365D", "YS"),
    ]
    best = min(options, key=lambda option: abs(np.log(median / pd.Timedelta(option[0]))))
    return best[1]


# --- features ----------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    unit: str
    lags: list[int]
    windows: list[int]
    season: int
    inputs: list[str] = field(default_factory=list)
    country: str | None = None

    @property
    def max_lag(self) -> int:
        return max([*self.lags, *self.windows])


_HOLIDAY_CACHE: dict[tuple[str, int, int], pd.DatetimeIndex] = {}


def holiday_dates(country: str, first_year: int, last_year: int) -> pd.DatetimeIndex:
    key = (country, first_year, last_year)
    if key not in _HOLIDAY_CACHE:
        holidays = require("holidays", "Holiday features")
        try:
            calendar = holidays.country_holidays(country, years=range(first_year, last_year + 1))
        except (KeyError, NotImplementedError) as exc:
            raise PlainMLError(
                f"No holiday calendar for '{country}'.",
                hint="Use a two-letter country code such as US, GB, IN, DE, FR, CA, AU.",
            ) from exc
        _HOLIDAY_CACHE[key] = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in calendar))
    return _HOLIDAY_CACHE[key]


def _holiday_features(dates: pd.DatetimeIndex, unit: str, country: str | None) -> list[np.ndarray]:
    if not country or len(dates) == 0:
        return []
    days = dates.normalize()
    calendar = holiday_dates(country, int(days.min().year) - 1, int(days.max().year) + 1)
    if unit in ("min", "h", "D", "B"):
        one = pd.Timedelta(days=1)
        return [
            np.asarray(days.isin(calendar), dtype=float),
            np.asarray((days + one).isin(calendar), dtype=float),  # the day before a holiday
            np.asarray((days - one).isin(calendar), dtype=float),  # the day after
        ]
    span = {"W": 7, "M": 31, "Q": 92, "Y": 366}.get(unit, 7)
    counts = [((calendar > d - pd.Timedelta(days=span)) & (calendar <= d)).sum() for d in days]
    return [np.asarray(counts, dtype=float)]


def _calendar(dates: pd.DatetimeIndex, unit: str, country: str | None = None) -> np.ndarray:
    columns: list[Any] = []
    if unit in ("min", "h"):
        columns += [dates.hour, dates.dayofweek]
    if unit in ("D", "B"):
        columns += [dates.dayofweek, dates.month, dates.dayofyear]
    if unit == "W":
        columns += [dates.isocalendar().week.to_numpy(dtype=float), dates.month]
    if unit == "M":
        columns += [dates.month]
    if unit == "Q":
        columns += [dates.quarter]
    columns += _holiday_features(dates, unit, country)
    if not columns:
        return np.zeros((len(dates), 0))
    return np.column_stack([np.asarray(c, dtype=float) for c in columns])


def _training_matrix(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: FeatureConfig,
    inputs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = pd.Series(values)
    parts = [series.shift(lag).to_numpy() for lag in cfg.lags]
    parts += [series.shift(1).rolling(w).mean().to_numpy() for w in cfg.windows]
    extra = inputs if inputs is not None and inputs.size else np.zeros((len(values), 0))
    X = np.column_stack(
        [
            *parts,
            _calendar(dates, cfg.unit, cfg.country),
            extra,
            np.arange(len(values), dtype=float),
        ]
    )
    rows = np.arange(cfg.max_lag, len(values))
    return X[rows], values[rows], values[rows - 1]


def _step_features(
    history: np.ndarray,
    date: pd.Timestamp,
    index: int,
    cfg: FeatureConfig,
    inputs: np.ndarray | None = None,
) -> np.ndarray:
    lags = [history[-lag] for lag in cfg.lags]
    windows = [history[-w:].mean() for w in cfg.windows]
    calendar = _calendar(pd.DatetimeIndex([date]), cfg.unit, cfg.country)[0]
    extra = inputs if inputs is not None else np.zeros(0)
    return np.array([*lags, *windows, *calendar, *extra, float(index)])


def _make_regressor(key: str, seed: int) -> Any:
    if key == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if key == "rf":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if key == "histgb":
        return HistGradientBoostingRegressor(random_state=seed)
    if key == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1
        )
    if key == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=seed, verbosity=0
        )
    raise AssertionError(key)


def periods_between(start: pd.Timestamp, end: pd.Timestamp, offset: Any) -> int:
    """Whole periods from start to end on the series' calendar (negative if end is earlier).

    The trend feature is 'periods since training began', so a forecast continued from a
    different stretch of history must start counting at the right place, not at zero.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end >= start:
        return len(pd.date_range(start, end, freq=offset)) - 1
    return -(len(pd.date_range(end, start, freq=offset)) - 1)


# --- the forecaster ------------------------------------------------------------------------


class Forecaster(BaseEstimator):
    """A fitted forecasting model plus the recent history it continues from."""

    def __init__(
        self,
        key: str = "naive",
        estimator: Any = None,
        config: FeatureConfig | None = None,
        freq: str = "D",
        history: pd.Series | None = None,
        step_rmse: list[float] | None = None,
        horizon: int = 30,
        origin: pd.Timestamp | None = None,
        inputs: pd.DataFrame | None = None,
        planned: pd.DataFrame | None = None,
    ):
        self.key = key
        self.estimator = estimator
        self.config = config
        self.freq = freq
        self.history = history
        self.step_rmse = step_rmse
        self.horizon = horizon
        self.origin = origin
        self.inputs = inputs
        self.planned = planned

    @property
    def input_columns(self) -> list[str]:
        return list(self.config.inputs) if self.config else []

    def fit(self, series: pd.Series, inputs: pd.DataFrame | None = None) -> Forecaster:
        self.history = series
        self.inputs = inputs
        self.origin = pd.Timestamp(series.index[0])  # the trend feature counts periods from here
        if self.key not in BASELINES:
            assert self.config is not None
            X, y, previous = _training_matrix(
                series.to_numpy(dtype=float),
                pd.DatetimeIndex(series.index),
                self.config,
                inputs[self.input_columns].to_numpy(dtype=float)
                if inputs is not None and self.input_columns
                else None,
            )
            target = (
                y if self.key == "ridge" else y - previous
            )  # trees model the change, so trends carry on
            self.estimator_ = clone(self.estimator).fit(X, target)
        return self

    def _future_inputs(
        self, future: pd.DatetimeIndex, planned: pd.DataFrame | None, last: pd.Series | None
    ) -> tuple[np.ndarray | None, bool]:
        """Input values for the future dates: planned ones where given, else the last known."""
        if not self.input_columns:
            return None, False
        frame = pd.DataFrame(index=future, columns=self.input_columns, dtype=float)
        if planned is not None and not planned.empty:
            known = planned.reindex(future)[self.input_columns]
            frame.update(known)
        assumed = bool(frame.isna().any().any())
        if last is not None:
            frame = frame.fillna(last[self.input_columns])
        return frame.fillna(0.0).to_numpy(dtype=float), assumed

    def _path(
        self, values: np.ndarray, future: pd.DatetimeIndex, offset: int, inputs: np.ndarray | None
    ) -> np.ndarray:
        cfg = self.config
        assert cfg is not None
        history = values.astype(float).copy()
        out = []
        for step, date in enumerate(future):
            if self.key == "naive":
                value = history[len(values) - 1]
            elif self.key == "seasonal_naive":
                season = max(1, cfg.season)
                value = (
                    values[len(values) - season + (step % season)]
                    if len(values) >= season
                    else values[-1]
                )
            else:
                row = inputs[step] if inputs is not None else None
                features = _step_features(history, date, offset + len(history), cfg, row)
                predicted = float(self.estimator_.predict(features.reshape(1, -1))[0])
                value = predicted if self.key == "ridge" else history[-1] + predicted
            out.append(value)
            history = np.append(history, value)
        return np.asarray(out)

    def forecast(
        self,
        horizon: int | None = None,
        history: pd.Series | None = None,
        planned: pd.DataFrame | None = None,
        history_inputs: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        horizon = int(horizon or self.horizon)
        series = history if history is not None else self.history
        assert series is not None, "call fit() first"
        offset = pd.tseries.frequencies.to_offset(self.freq)
        future = pd.date_range(series.index[-1] + offset, periods=horizon, freq=offset)
        start = (
            periods_between(self.origin, series.index[0], offset) if self.origin is not None else 0
        )
        known = history_inputs if history_inputs is not None else self.inputs
        last = known.iloc[-1] if known is not None and len(known) else None
        plans = planned if planned is not None else self.planned  # plans given at training time
        inputs, assumed = self._future_inputs(future, plans, last)
        self.assumed_inputs_ = assumed
        values = self._path(series.to_numpy(dtype=float), future, start, inputs)
        rmse = np.asarray(self.step_rmse or [0.0])
        if len(rmse) < horizon:  # beyond the backtested horizon, widen like a random walk
            extra = rmse[-1] * np.sqrt(np.arange(len(rmse) + 1, horizon + 1) / len(rmse))
            rmse = np.concatenate([rmse, extra])
        spread = Z80 * rmse[:horizon]
        return pd.DataFrame(
            {
                "date": future,
                "forecast": values,
                "lower_80": values - spread,
                "upper_80": values + spread,
            }
        )

    def predict(self, X: Any = None) -> np.ndarray:
        return self.forecast()["forecast"].to_numpy()


class GroupForecaster(BaseEstimator):
    """One Forecaster per group (store, product...), used together."""

    def __init__(
        self,
        forecasters: dict[str, Forecaster] | None = None,
        group: str = "group",
        horizon: int = 30,
    ):
        self.forecasters = forecasters
        self.group = group
        self.horizon = horizon

    def forecast(
        self,
        horizon: int | None = None,
        histories: dict[str, pd.Series] | None = None,
        planned: dict[str, pd.DataFrame] | None = None,
        history_inputs: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        frames = []
        for name, model in (self.forecasters or {}).items():
            history = (histories or {}).get(name)
            if histories is not None and history is None:
                continue
            frame = model.forecast(
                horizon or self.horizon,
                history,
                (planned or {}).get(name),
                (history_inputs or {}).get(name),
            )
            frame.insert(0, self.group, name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def predict(self, X: Any = None) -> np.ndarray:
        return self.forecast()["forecast"].to_numpy()


@dataclass
class ForecastResult:
    run_dir: Path
    forecast: pd.DataFrame
    leaderboard: pd.DataFrame
    model: Any
    insights: list[str] = field(default_factory=list)

    @property
    def report_path(self) -> Path:
        return self.run_dir / REPORT_FILE

    def __repr__(self) -> str:
        return f"ForecastResult({len(self.forecast)} forecast rows, run_dir='{self.run_dir}')"


# --- preparing series ------------------------------------------------------------------------


def _pick_date(df: pd.DataFrame, date: str | None, target: str) -> str:
    if date:
        return find_column(date, df.columns, "Date column")
    for column in df.columns:
        if column != target and infer_column(df[column]).kind == Kind.DATETIME:
            return column
    raise PlainMLError("Couldn't find a date column.", hint="Name it with --date COLUMN.")


def _resample(series: pd.Series, freq: str, agg: str) -> pd.Series:
    try:
        resampled = series.resample(freq)
    except ValueError as exc:
        raise PlainMLError(
            f"Unknown frequency '{freq}'.",
            hint="Use D (daily), W, MS (monthly), QS, YS, h (hourly)...",
        ) from exc
    return (
        resampled.sum(min_count=1)
        if agg == "sum"
        else resampled.mean()
        if agg == "mean"
        else resampled.last()
    )


def build_series(
    df: pd.DataFrame,
    target: str,
    date: str,
    freq: str | None,
    agg: str,
    inputs: list[str] | None = None,
) -> tuple[pd.Series, str, list[str], pd.DataFrame | None, pd.DataFrame | None]:
    """Turn raw rows into one value per period, filling gaps.

    Returns (series, freq, notes, input history, planned future inputs). Rows dated after
    the last known target value are read as plans for the inputs, not as history.
    """
    notes = []
    inputs = inputs or []
    dates = parse_dates(df[date], guess_date_format(df[date]))
    frame = pd.DataFrame({"date": dates, "y": coerce_numeric(df[target])})
    for column in inputs:
        frame[column] = coerce_numeric(df[column])
    frame = frame.dropna(subset=["date"])
    if frame["y"].isna().all():
        raise PlainMLError(f"'{target}' has no numeric values to forecast.")
    last_known = frame.loc[frame["y"].notna(), "date"].max()
    future_rows = frame[frame["date"] > last_known]
    frame = frame[frame["date"] <= last_known]
    if frame["date"].duplicated().any():
        notes.append(f"Several rows share a date; combined them with {agg}.")
    grouped = frame.groupby("date")
    series = (
        grouped["y"].sum(min_count=1)
        if agg == "sum"
        else grouped["y"].mean()
        if agg == "mean"
        else grouped["y"].last()
    )
    series = series.sort_index()
    freq = freq or _infer_freq(series.index)
    regular = _resample(series, freq, agg)
    gaps = int(regular.isna().sum())
    if gaps:
        regular = regular.interpolate(limit_direction="both")
        notes.append(f"Filled {gaps} missing period(s) by interpolation.")
    history_inputs = planned = None
    if inputs:
        history_inputs = frame.set_index("date")[inputs].sort_index().resample(freq).mean()
        history_inputs = (
            history_inputs.reindex(regular.index).interpolate(limit_direction="both").fillna(0.0)
        )
        if not future_rows.empty:
            planned = (
                future_rows.set_index("date")[inputs]
                .sort_index()
                .resample(freq)
                .mean()
                .dropna(how="all")
            )
    if len(regular) > MAX_POINTS:
        regular = regular.iloc[-MAX_POINTS:]
        if history_inputs is not None:
            history_inputs = history_inputs.iloc[-MAX_POINTS:]
        notes.append(f"Using the most recent {MAX_POINTS:,} periods.")
    regular.name = target
    return regular, freq, notes, history_inputs, planned


def _config(
    n: int, unit: str, horizon: int, inputs: list[str] | None = None, country: str | None = None
) -> FeatureConfig:
    season = SEASON[unit]
    candidates = {1, 2, 3}
    if season > 1:
        candidates |= {season, 2 * season}
    if unit == "h":
        candidates.add(168)
    usable = max(3, (n - horizon) // 3)
    lags = sorted(lag for lag in candidates if lag <= usable)
    windows = sorted(w for w in {3, season} if 1 < w <= usable)
    return FeatureConfig(
        unit=unit,
        lags=lags or [1],
        windows=windows,
        season=season if season <= usable else 1,
        inputs=list(inputs or []),
        country=country,
    )


def _backtest(
    key: str,
    series: pd.Series,
    cfg: FeatureConfig,
    freq: str,
    horizon: int,
    folds: int,
    seed: int,
    inputs: pd.DataFrame | None = None,
) -> np.ndarray:
    """Forecast `folds` past windows from the data before each; return errors (folds × horizon).

    Inputs for each window are their real recorded values, as they'd be known in advance.
    """
    values = series.to_numpy(dtype=float)
    errors = []
    for fold in range(folds, 0, -1):
        cut = len(values) - fold * horizon
        model = Forecaster(
            key,
            _make_regressor(key, seed) if key not in BASELINES else None,
            cfg,
            freq,
            horizon=horizon,
        )
        model.fit(series.iloc[:cut], inputs.iloc[:cut] if inputs is not None else None)
        planned = inputs.iloc[cut : cut + horizon] if inputs is not None else None
        predicted = model.forecast(horizon, planned=planned)["forecast"].to_numpy()
        errors.append(predicted - values[cut : cut + horizon])
    return np.asarray(errors)


def _folds_for(n: int, horizon: int, unit: str, wanted: int) -> int:
    folds = wanted
    while folds > 1 and n - folds * horizon < max(10, 2 * SEASON[unit] + 3):
        folds -= 1
    return folds


def _enough_history(n: int, horizon: int, folds: int) -> bool:
    return n >= 2 * horizon + 5 and n - folds * horizon >= 6


def _insights(
    series: pd.Series,
    unit: str,
    target: str,
    inputs: pd.DataFrame | None = None,
    country: str | None = None,
) -> list[str]:
    """Plain-English notes on trend, seasonality, inputs and holidays."""
    out = []
    values = series.to_numpy(dtype=float)
    mean = float(np.mean(values)) or 1.0
    per_year = {"min": 525_600, "h": 8760, "D": 365, "B": 260, "W": 52, "M": 12, "Q": 4, "Y": 1}[
        unit
    ]
    if len(values) >= 8:
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        yearly = slope * per_year / abs(mean)
        span = len(values) / per_year
        if abs(yearly) > 0.03 and span >= 0.25:
            out.append(
                f"Trend: {target} is {'rising' if yearly > 0 else 'falling'} by about {fmt_pct(abs(yearly), 0)} of its average per year."
            )
        elif span >= 0.25:
            out.append(f"Trend: {target} is roughly flat over the period.")
    index = pd.DatetimeIndex(series.index)
    position: Any = None
    names: list[str] = []
    if unit in ("D", "B") and len(values) >= 21:
        position, names = (
            index.dayofweek,
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
    elif unit == "h" and len(values) >= 72:
        position, names = index.hour, [f"{h}:00" for h in range(24)]
    elif unit == "M" and len(values) >= 24:
        position, names = (
            index.month - 1,
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
        )
    elif unit == "Q" and len(values) >= 8:
        position, names = index.quarter - 1, ["Q1", "Q2", "Q3", "Q4"]
    if position is not None:
        profile = pd.Series(values).groupby(np.asarray(position)).mean() / abs(mean)
        high, low = int(profile.idxmax()), int(profile.idxmin())
        if profile.max() - profile.min() > 0.1:
            out.append(
                f"Seasonality: {names[high]} is highest ({'+' if profile[high] >= 1 else ''}{fmt_pct(profile[high] - 1, 0)} vs average), "
                f"{names[low]} lowest ({fmt_pct(profile[low] - 1, 0)})."
            )
        else:
            out.append("Seasonality: no strong repeating pattern.")
    for column in list(inputs.columns) if inputs is not None else []:
        assert inputs is not None
        values_in = inputs[column].to_numpy(dtype=float)
        if set(np.unique(values_in)) <= {0.0, 1.0} and 0 < values_in.mean() < 1:
            on, off = values[values_in == 1].mean(), values[values_in == 0].mean()
            if off:
                out.append(
                    f"When {column} is on, {target} averages {fmt_pct(abs(on / off - 1), 0)} {'higher' if on > off else 'lower'}."
                )
        elif np.std(values_in) > 0:
            corr = float(np.corrcoef(values_in, values)[0, 1])
            if abs(corr) >= 0.2:
                out.append(
                    f"Higher {column} goes with {'higher' if corr > 0 else 'lower'} {target} (correlation {corr:.2f})."
                )
    if country and unit in ("D", "B"):
        holidays_mask = index.normalize().isin(
            holiday_dates(country, index.min().year, index.max().year)
        )
        if holidays_mask.any() and (~holidays_mask).any():
            on, off = values[holidays_mask].mean(), values[~holidays_mask].mean()
            if off:
                out.append(
                    f"On public holidays ({country}), {target} averages {fmt_pct(abs(on / off - 1), 0)} {'higher' if on > off else 'lower'}."
                )
    return out


def _chosen_models(models: list[str] | None, cfg: FeatureConfig) -> list[str]:
    chosen = []
    for raw in models or list(MODELS):
        key = raw.strip().lower()
        if key not in MODELS:
            raise PlainMLError(
                f"Unknown forecasting model '{raw}'.{did_you_mean(key, MODELS)}",
                hint="Choose from: " + ", ".join(MODELS),
            )
        if key in ("lightgbm", "xgboost") and not is_installed(key):
            if models:
                note(f"Skipping {key}: not installed.")
            continue
        if key == "seasonal_naive" and cfg.season <= 1:
            continue
        chosen.append(key)
    if not any(k in BASELINES for k in chosen):
        chosen.insert(0, "naive")
    return chosen


def _score_row(key: str, errors: np.ndarray, actual: np.ndarray, seconds: float) -> dict[str, Any]:
    flat = errors.ravel()
    row = {
        "key": key,
        "model": MODELS[key],
        "mae": float(np.mean(np.abs(flat))),
        "rmse": float(np.sqrt(np.mean(flat**2))),
        "bias": float(np.mean(flat)),
        "seconds": round(seconds, 2),
    }
    if np.all(np.abs(actual) > 1e-9):
        row["mape"] = float(np.mean(np.abs(flat / actual)))
    return row


def _actuals(series: pd.Series, horizon: int, folds: int) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return np.concatenate(
        [
            values[len(values) - f * horizon : len(values) - f * horizon + horizon]
            for f in range(folds, 0, -1)
        ]
    )


def _step_rmse(errors: np.ndarray) -> list[float]:
    """Typical error at each step ahead, for the forecast range.

    With only a few backtest windows, one step's error is a handful of numbers, so each
    step pools its neighbours (two either side) to keep the range from pinching or jumping.
    """
    squared = errors**2
    steps = squared.shape[1]
    pooled = [np.sqrt(np.mean(squared[:, max(0, k - 2) : k + 3])) for k in range(steps)]
    return np.maximum.accumulate(pooled).tolist()  # uncertainty shouldn't shrink further out


def _read_future(
    future: Any, date_column: str, inputs: list[str], freq: str, group: str | None
) -> Any:
    if future is None:
        return None
    frame = load_data(future)
    date = find_column(date_column, frame.columns, "Date column (in the future file)")
    columns = [find_column(c, frame.columns, "Input column (in the future file)") for c in inputs]
    frame = frame.assign(
        **{"__date": parse_dates(frame[date], guess_date_format(frame[date]))}
    ).dropna(subset=["__date"])
    for column in columns:
        frame[column] = coerce_numeric(frame[column])

    def one(part: pd.DataFrame) -> pd.DataFrame:
        return (
            part.set_index("__date")[columns].sort_index().resample(freq).mean().dropna(how="all")
        )

    if group:
        key = find_column(group, frame.columns, "Group column (in the future file)")
        return {str(name): one(part) for name, part in frame.groupby(frame[key].astype(str))}
    return one(frame)


# --- the main function ------------------------------------------------------------------------


def forecast(
    data: Any,
    target: str,
    *,
    date: str | None = None,
    horizon: int | None = None,
    freq: str | None = None,
    agg: str = "sum",
    models: list[str] | None = None,
    backtests: int = 3,
    group: str | None = None,
    inputs: list[str] | None = None,
    future: Any = None,
    country: str | None = None,
    output: str | Path | None = None,
    seed: int = 42,
    out_dir: str | Path = "runs",
    name: str | None = None,
    report: bool = True,
    verbose: bool = True,
    **load_options: Any,
) -> ForecastResult:
    """Backtest forecasting models on ``target`` over time and forecast ``horizon`` periods ahead.

    ``group`` forecasts each value of that column separately; ``inputs`` are extra columns
    known in advance (future values from later rows, or a ``future`` file); ``country`` adds
    public-holiday features.
    """
    started = time.time()
    if agg not in ("sum", "mean", "last"):
        raise PlainMLError("--agg must be sum, mean or last.")
    with quiet(not verbose):
        df = load_data(data, **load_options)
        target_column = find_column(target, df.columns, "Target column")
        date_column = _pick_date(df, date, target_column)
        input_columns = [find_column(c, df.columns, "Input column") for c in inputs or []]
        group_column = find_column(group, df.columns, "Group column") if group else None
        if country:
            holiday_dates(country.upper(), 2000, 2000)  # fail early on an unknown country
            country = country.upper()
        if group_column:
            return _forecast_groups(
                df,
                data,
                target_column,
                date_column,
                group_column,
                input_columns,
                freq,
                agg,
                horizon,
                models,
                backtests,
                future,
                country,
                output,
                seed,
                out_dir,
                name,
                report,
                verbose,
                started,
            )
        series, freq, notes, history_inputs, planned = build_series(
            df, target_column, date_column, freq, agg, input_columns
        )
        if future is not None and input_columns:
            planned = _read_future(future, date_column, input_columns, freq, None)
        unit = base_unit(freq)
        horizon = int(horizon or HORIZON[unit])
        if horizon < 1:
            raise PlainMLError("--horizon must be at least 1.")
        n = len(series)
        folds = _folds_for(n, horizon, unit, backtests)
        if not _enough_history(n, horizon, folds):
            raise PlainMLError(
                f"Not enough history: {n} {UNIT[unit]}s for a {horizon}-{UNIT[unit]} forecast.",
                hint=f"Use a --horizon of at most {max(1, (n - 5) // 2)}, or more data (at least twice the horizon).",
            )
        cfg = _config(n - folds * horizon, unit, 0, input_columns, country)
        chosen = _chosen_models(models, cfg)

        heading(f"Forecasting {target_column}: {n:,} {UNIT[unit]}s of history, {horizon} ahead")
        for message in notes:
            note(message)
        if input_columns:
            note(
                f"Using {', '.join(input_columns)} as "
                f"{'inputs' if len(input_columns) > 1 else 'an input'} known in advance."
            )
        if country:
            note(f"Including {country} public holidays.")
        note(
            f"Backtesting on the last {folds} × {horizon} {UNIT[unit]}s (each forecast only sees earlier data)."
        )

        rows, errors_by_model = [], {}
        actual = _actuals(series, horizon, folds)
        with console.status("Backtesting…") as status, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for key in chosen:
                status.update(f"Backtesting {MODELS[key]}")
                t0 = time.perf_counter()
                try:
                    errors = _backtest(key, series, cfg, freq, horizon, folds, seed, history_inputs)
                except Exception as exc:
                    note(f"{MODELS[key]} failed: {esc(str(exc).splitlines()[0][:100])}")
                    continue
                errors_by_model[key] = errors
                rows.append(_score_row(key, errors, actual, time.perf_counter() - t0))
        if not rows:
            raise PlainMLError("Every forecasting model failed.")
        board = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
        naive_mae = board.loc[board["key"].isin(BASELINES), "mae"].min()
        board["vs_baseline"] = 1 - board["mae"] / naive_mae if naive_mae else np.nan
        best_key = str(board.iloc[0]["key"])
        errors = errors_by_model[best_key]

        final_cfg = _config(n, unit, 0, input_columns, country)
        model = Forecaster(
            best_key,
            _make_regressor(best_key, seed) if best_key not in BASELINES else None,
            final_cfg,
            freq,
            step_rmse=_step_rmse(errors),
            horizon=horizon,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(series, history_inputs)
            model.planned = planned
            predicted = model.forecast(horizon)
        if getattr(model, "assumed_inputs_", False):
            notes.append(
                "Some future input values weren't given, so they're held at their last known values."
            )
        insights = _insights(series, unit, target_column, history_inputs, country)

        run_dir = create_run_dir(out_dir, name or f"{source_stem(data)}-forecast")
        created = datetime.now().isoformat(timespec="seconds")
        model.plainml_meta_ = _meta(
            created,
            run_dir,
            target_column,
            date_column,
            freq,
            agg,
            horizon,
            MODELS[best_key],
            input_columns,
            None,
            country,
        )
        joblib.dump(model, run_dir / MODEL_FILE, compress=3)
        save_table(predicted, run_dir / FORECAST_FILE)
        if output:
            save_table(predicted, output)
        board.to_csv(run_dir / LEADERBOARD_FILE, index=False)
        recent = series.iloc[-min(len(series), max(4 * horizon, 60)) :]
        last_fold_start = len(series) - horizon
        write_json(
            run_dir / EVALUATION_FILE,
            {
                "history": {"x": _ms(recent.index), "y": recent.tolist()},
                "forecast": _forecast_json(predicted),
                "backtest": {
                    "x": _ms(series.index[last_fold_start:]),
                    "actual": series.iloc[last_fold_start:].tolist(),
                    "predicted": (
                        errors[-1] + series.to_numpy(dtype=float)[last_fold_start:]
                    ).tolist(),
                },
                "insights": insights,
                "notes": notes,
                "unit": UNIT[unit],
                "folds": folds,
            },
        )
        _write_run(
            run_dir,
            created,
            data,
            df,
            target_column,
            n,
            best_key,
            board,
            date_column,
            freq,
            horizon,
            agg,
            chosen,
            folds,
            seed,
            input_columns,
            None,
            country,
            started,
            report,
        )
    result = ForecastResult(run_dir, predicted, board, model, insights)
    if verbose:
        _render(result, target_column, UNIT[unit], output, notes)
    return result


def _forecast_groups(
    df: pd.DataFrame,
    data: Any,
    target: str,
    date: str,
    group: str,
    input_columns: list[str],
    freq: str | None,
    agg: str,
    horizon: int | None,
    models: list[str] | None,
    backtests: int,
    future: Any,
    country: str | None,
    output: Any,
    seed: int,
    out_dir: Any,
    name: str | None,
    report: bool,
    verbose: bool,
    started: float,
) -> ForecastResult:
    notes: list[str] = []
    labels = df[group].astype(str)
    sizes = coerce_numeric(df[target]).abs().groupby(labels).sum().sort_values(ascending=False)
    names = list(sizes.index)
    if len(names) > MAX_GROUPS:
        notes.append(f"Forecasting the {MAX_GROUPS} largest of {len(names)} groups.")
        names = names[:MAX_GROUPS]
    if freq is None:
        freq = _infer_freq(
            pd.DatetimeIndex(parse_dates(df[date], guess_date_format(df[date])).dropna().unique())
        )
    series_by, inputs_by, planned_by = {}, {}, {}
    for label in names:
        part = df[labels == label]
        try:
            series, _, _, hist_inputs, planned = build_series(
                part, target, date, freq, agg, input_columns
            )
        except PlainMLError:
            continue
        series_by[label], inputs_by[label], planned_by[label] = series, hist_inputs, planned
    if future is not None and input_columns:
        planned_by.update(_read_future(future, date, input_columns, freq, group))
    unit = base_unit(freq)
    horizon = int(horizon or HORIZON[unit])
    usable = {g: s for g, s in series_by.items() if _enough_history(len(s), horizon, 1)}
    too_short = len(series_by) - len(usable)
    if too_short:
        notes.append(
            f"Skipped {too_short} group(s) with too little history for a {horizon}-{UNIT[unit]} forecast."
        )
    if not usable:
        raise PlainMLError(
            f"No group has enough history for a {horizon}-{UNIT[unit]} forecast.",
            hint="Use a shorter --horizon.",
        )
    sample = [g for g in names if g in usable][:SELECTION_GROUPS]
    min_len = min(len(usable[g]) for g in sample)
    folds = _folds_for(min_len, horizon, unit, backtests)
    cfg = _config(min_len - folds * horizon, unit, 0, input_columns, country)
    chosen = _chosen_models(models, cfg)
    heading(
        f"Forecasting {target} for {len(usable)} {group} group(s), {horizon} {UNIT[unit]}s ahead"
    )
    for message in notes:
        note(message)
    note(
        f"Choosing the model by backtesting on the {len(sample)} largest groups ({folds} × {horizon} {UNIT[unit]}s each)."
    )

    rows = []
    with console.status("Backtesting…") as status, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        naive_mae = {}
        for key in chosen:
            status.update(f"Backtesting {MODELS[key]}")
            t0 = time.perf_counter()
            relative, maes = [], []
            try:
                for g in sample:
                    errors = _backtest(
                        key, usable[g], cfg, freq, horizon, folds, seed, inputs_by[g]
                    )
                    mae = float(np.mean(np.abs(errors)))
                    maes.append(mae)
                    if key == "naive":
                        naive_mae[g] = mae
                    relative.append(mae / naive_mae[g] if naive_mae.get(g) else np.nan)
            except Exception as exc:
                note(f"{MODELS[key]} failed: {esc(str(exc).splitlines()[0][:100])}")
                continue
            rows.append(
                {
                    "key": key,
                    "model": MODELS[key],
                    "mae": float(np.mean(maes)),
                    "relative_error": float(np.nanmean(relative)) if relative else np.nan,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
    board = pd.DataFrame(rows)
    if board.empty:
        raise PlainMLError("Every forecasting model failed.")
    board["vs_baseline"] = 1 - board["relative_error"]
    board = board.sort_values(["relative_error", "mae"]).reset_index(drop=True)
    best_key = str(board.iloc[0]["key"])

    forecasters: dict[str, Forecaster] = {}
    frames = []
    assumed = False
    with console.status(f"Fitting {MODELS[best_key]} to each group…"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g, series in usable.items():
            g_folds = _folds_for(len(series), horizon, unit, 2)
            g_cfg = _config(len(series), unit, 0, input_columns, country)
            errors = _backtest(
                best_key,
                series,
                _config(len(series) - g_folds * horizon, unit, 0, input_columns, country),
                freq,
                horizon,
                g_folds,
                seed,
                inputs_by[g],
            )
            model = Forecaster(
                best_key,
                _make_regressor(best_key, seed) if best_key not in BASELINES else None,
                g_cfg,
                freq,
                step_rmse=_step_rmse(errors),
                horizon=horizon,
            )
            model.fit(series, inputs_by[g])
            model.planned = planned_by.get(g)
            frame = model.forecast(horizon)
            assumed = assumed or getattr(model, "assumed_inputs_", False)
            frame.insert(0, group, g)
            frames.append(frame)
            forecasters[g] = model
    if assumed:
        notes.append(
            "Some future input values weren't given, so they're held at their last known values."
        )
    predicted = pd.concat(frames, ignore_index=True)
    grouped_model = GroupForecaster(forecasters, group=group, horizon=horizon)
    total = pd.concat(list(usable.values()), axis=1).sum(axis=1, min_count=1).dropna()
    insights = _insights(total, unit, f"total {target}", None, country)
    largest = [g for g in names if g in usable][:3]
    insights.append(f"Largest {group} groups: {', '.join(largest)}.")

    run_dir = create_run_dir(out_dir, name or f"{source_stem(data)}-forecast")
    created = datetime.now().isoformat(timespec="seconds")
    grouped_model.plainml_meta_ = _meta(
        created,
        run_dir,
        target,
        date,
        freq,
        agg,
        horizon,
        MODELS[best_key],
        input_columns,
        group,
        country,
    )
    grouped_model.plainml_meta_["groups"] = list(forecasters)
    joblib.dump(grouped_model, run_dir / MODEL_FILE, compress=3)
    save_table(predicted, run_dir / FORECAST_FILE)
    if output:
        save_table(predicted, output)
    board.to_csv(run_dir / LEADERBOARD_FILE, index=False)
    total_future = (
        predicted.groupby("date")[["forecast", "lower_80", "upper_80"]].sum().reset_index()
    )
    recent = total.iloc[-min(len(total), max(4 * horizon, 60)) :]
    panels = []
    for g in largest + [g for g in names if g in usable and g not in largest][:1]:
        s = usable[g].iloc[-min(len(usable[g]), max(3 * horizon, 45)) :]
        f = predicted[predicted[group] == g]
        panels.append(
            {
                "name": g,
                "history": {"x": _ms(s.index), "y": s.tolist()},
                "forecast": _forecast_json(f),
            }
        )
    write_json(
        run_dir / EVALUATION_FILE,
        {
            "history": {"x": _ms(recent.index), "y": recent.tolist()},
            "forecast": _forecast_json(total_future),
            "groups": panels,
            "group_column": group,
            "insights": insights,
            "notes": notes,
            "unit": UNIT[unit],
            "folds": folds,
        },
    )
    _write_run(
        run_dir,
        created,
        data,
        df,
        target,
        len(total),
        best_key,
        board,
        date,
        freq,
        horizon,
        agg,
        chosen,
        folds,
        seed,
        input_columns,
        group,
        country,
        started,
        report,
        n_groups=len(forecasters),
    )
    result = ForecastResult(run_dir, predicted, board, grouped_model, insights)
    if verbose:
        _render(result, target, UNIT[unit], output, notes)
    return result


def _meta(
    created: str,
    run_dir: Path,
    target: str,
    date: str,
    freq: str,
    agg: str,
    horizon: int,
    model: str,
    inputs: list[str],
    group: str | None,
    country: str | None,
) -> dict[str, Any]:
    return {
        "plainml_version": __version__,
        "packages": environment()["packages"],
        "created": created,
        "run": run_dir.name,
        "task": FORECAST,
        "targets": [target],
        "features": [date, target, *inputs, *([group] if group else [])],
        "date": date,
        "freq": freq,
        "agg": agg,
        "horizon": horizon,
        "model": model,
        "inputs": inputs,
        "group": group,
        "country": country,
        "schema": {"order": [date, target, *inputs]},
    }


def _write_run(
    run_dir: Path,
    created: str,
    data: Any,
    df: pd.DataFrame,
    target: str,
    periods: int,
    best_key: str,
    board: pd.DataFrame,
    date: str,
    freq: str,
    horizon: int,
    agg: str,
    chosen: list[str],
    folds: int,
    seed: int,
    inputs: list[str],
    group: str | None,
    country: str | None,
    started: float,
    report: bool,
    n_groups: int | None = None,
) -> None:
    score = float(board.iloc[0]["mae"])
    write_json(
        run_dir / RUN_FILE,
        {
            "kind": "forecast",
            "plainml_version": __version__,
            "environment": environment(),
            "created": created,
            "task": FORECAST,
            "task_label": "time-series forecasting",
            "target": target,
            "metric": "mae",
            "metric_label": "MAE",
            "data": {
                "source": describe_source(data),
                "rows": len(df),
                "raw_rows": len(df),
                "periods": periods,
                "columns": df.shape[1],
                "fingerprint": fingerprint(df),
            },
            "best": {"key": best_key, "name": MODELS[best_key], "score": score, "cv_score": score},
            "options": {
                "date": date,
                "freq": freq,
                "horizon": horizon,
                "agg": agg,
                "models": chosen,
                "backtests": folds,
                "seed": seed,
                "inputs": inputs,
                "group": group,
                "country": country,
                "groups": n_groups,
            },
            "timings": {"total_seconds": round(time.time() - started, 2)},
            "files": {
                "model": MODEL_FILE,
                "forecast": FORECAST_FILE,
                "report": REPORT_FILE if report else None,
                "model_card": CARD_FILE,
            },
        },
    )
    try:
        write_model_card(run_dir)
    except Exception as exc:
        warn(f"Couldn't write the model card: {esc(str(exc)[:120])}")
    if report:
        try:
            from plainml.report import write_report

            write_report(run_dir)
        except Exception as exc:
            warn(f"Couldn't write the HTML report: {esc(str(exc)[:120])}")


def _forecast_json(frame: pd.DataFrame) -> dict[str, list[float]]:
    return {
        "x": _ms(frame["date"]),
        "y": frame["forecast"].tolist(),
        "lo": frame["lower_80"].tolist(),
        "hi": frame["upper_80"].tolist(),
    }


def _ms(dates: Any) -> list[float]:
    """Milliseconds since 1970, for the browser. (pandas 3 may store dates in µs, so be explicit.)"""
    return [float(v) for v in pd.DatetimeIndex(dates).as_unit("ms").asi8]


def forecast_with_model(
    model: Any, frame: pd.DataFrame | None = None, horizon: int | None = None
) -> pd.DataFrame:
    """Forecast with a saved model, optionally continuing from newer history in ``frame``.

    Rows in ``frame`` after the last known target value supply planned input values.
    """
    meta = getattr(model, "plainml_meta_", {})
    inputs = list(meta.get("inputs") or [])
    if frame is None:
        return model.forecast(horizon)
    target = find_column(meta["targets"][0], frame.columns, "Target column")
    date = find_column(meta["date"], frame.columns, "Date column")
    input_columns = [find_column(c, frame.columns, "Input column") for c in inputs]
    if isinstance(model, GroupForecaster):
        group = find_column(meta["group"], frame.columns, "Group column")
        histories, planned, history_inputs = {}, {}, {}
        for name, part in frame.groupby(frame[group].astype(str)):
            if name not in (model.forecasters or {}):
                note(f"Skipping {esc(name)}: the model wasn't trained on that {esc(group)}.")
                continue
            series, _, _, hist, plan = build_series(
                part, target, date, meta["freq"], meta.get("agg", "sum"), input_columns
            )
            histories[name], planned[name], history_inputs[name] = series, plan, hist
        return model.forecast(horizon, histories, planned, history_inputs)
    series, _, _, hist, plan = build_series(
        frame, target, date, meta["freq"], meta.get("agg", "sum"), input_columns
    )
    needed = (model.config.max_lag if model.config else 0) + 1
    if len(series) < needed:
        raise PlainMLError(f"Need at least {needed} periods of history to continue from.")
    return model.forecast(horizon, series, plan, hist)


# --- output --------------------------------------------------------------------------------


def _render(result: ForecastResult, target: str, unit: str, output: Any, notes: list[str]) -> None:
    board = result.leaderboard
    grouped = "relative_error" in board
    heading("Backtest (lower is better)")
    table = Table(box=None, header_style="muted", pad_edge=False)
    columns: list[tuple[str, Any]] = [("Model", "left"), ("MAE ↓", "right")]
    if not grouped:
        columns.append(("RMSE ↓", "right"))
    if "mape" in board:
        columns.append(("MAPE ↓", "right"))
    columns += [("vs baseline", "right"), ("Time", "right")]
    for column, justify in columns:
        table.add_column(column, justify=justify)
    for i, row in board.iterrows():
        values = [("★ " if i == 0 else "") + row["model"], fmt_num(row["mae"])]
        if not grouped:
            values.append(fmt_num(row["rmse"]))
        if "mape" in board:
            values.append(fmt_pct(row["mape"], 1) if pd.notna(row.get("mape")) else "—")
        gain = row["vs_baseline"]
        values += [
            ""
            if row["key"] in BASELINES or pd.isna(gain)
            else (
                f"[good]{fmt_pct(gain, 0)} better[/]"
                if gain > 0
                else f"[warn]{fmt_pct(-gain, 0)} worse[/]"
            ),
            f"{row['seconds']:.1f}s",
        ]
        table.add_row(
            *values, style="best" if i == 0 else ("muted" if row["key"] in BASELINES else "")
        )
    console.print(table)
    if grouped:
        note(
            "Ranked by 'vs baseline': the average improvement per group over 'same as last period', "
            "so big and small groups count equally. MAE is averaged over the groups tested."
        )
    else:
        note(
            f"MAE: average forecast error, in units of {esc(target)}. 'vs baseline' compares with the best simple rule."
        )
    if board.iloc[0]["key"] in BASELINES:
        warn("A simple rule beat the models: the series may be too short or too noisy for more.")
    for message in notes:
        if "held at their last known values" in message:
            warn(esc(message))
    if result.insights:
        heading("Patterns in the history")
        for sentence in result.insights:
            console.print(f"• {esc(sentence)}")
    frame = result.forecast
    group = next(
        (c for c in frame.columns if c not in ("date", "forecast", "lower_80", "upper_80")), None
    )
    heading(f"Forecast ({frame['date'].nunique()} {unit}s ahead, with an 80% range)")
    preview = Table(box=None, header_style="muted", pad_edge=False)
    headers = ([group] if group else []) + ["Date", "Forecast", "Low (80%)", "High (80%)"]
    for column in headers:
        preview.add_column(
            esc(column),
            justify="right" if column in ("Forecast", "Low (80%)", "High (80%)") else "left",
        )
    shown = frame if len(frame) <= 8 else pd.concat([frame.head(4), frame.tail(3)])
    for i, row in shown.iterrows():
        if len(frame) > 8 and i == frame.index[-3]:
            preview.add_row(*([""] if group else []), "…", "", "", "")
        cells = [esc(row[group])] if group else []
        cells += [
            str(row["date"])[:16].replace(" 00:00", ""),
            fmt_value(round(row["forecast"], 2)),
            fmt_value(round(row["lower_80"], 2)),
            fmt_value(round(row["upper_80"], 2)),
        ]
        preview.add_row(*cells)
    console.print(preview)
    heading("Saved")
    console.print(f"[bold]{esc(result.run_dir)}[/]")
    console.print(f"  {FORECAST_FILE:<22}[muted]the forecast with its range[/]")
    console.print(f"  {REPORT_FILE:<22}[muted]charts of the history, forecast and backtests[/]")
    console.print(f"  {CARD_FILE:<22}[muted]a one-page summary to share with the model[/]")
    if output:
        console.print(f"  [muted]Also saved to {esc(output)}[/]")
    info(
        "Forecast again later (optionally with newer data): plainml predict latest [NEW_DATA.csv] --horizon N"
    )


def _line_spec(history: dict, future: dict, target: str, height: int = 340) -> dict[str, Any]:
    all_y = history["y"] + future["lo"] + future["hi"]
    finite = [v for v in all_y if v is not None and np.isfinite(v)]
    low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)
    pad = (high - low) * 0.06 or 1.0
    return {
        "type": "line",
        "series": [
            {"x": history["x"], "y": history["y"], "role": "context", "name": "history"},
            {"x": future["x"], "y": future["y"], "role": "accent", "name": "forecast"},
        ],
        "band": {"x": future["x"], "lo": future["lo"], "hi": future["hi"], "name": "80% range"},
        "xDomain": [history["x"][0], future["x"][-1]],
        "yDomain": [low - pad, high + pad],
        "xFmt": "time",
        "xLabel": "date",
        "yLabel": target,
        "height": height,
    }


def render_forecast_report(run: Any) -> str:
    from plainml.report import (
        as_table_view,
        bar_list,
        card,
        chart,
        grid,
        page,
        section,
        sentences_list,
        stat_tiles,
        table,
    )

    info_json, evaluation, board = run.info, run.evaluation, run.leaderboard
    best = info_json["best"]
    target = info_json["target"]
    charts: dict[str, Any] = {}
    history, future = evaluation["history"], evaluation["forecast"]
    grouped = bool(evaluation.get("groups"))
    charts["forecast"] = _line_spec(history, future, f"total {target}" if grouped else target)
    unit = evaluation.get("unit", "period")
    tiles = [
        ("Average error (MAE)", fmt_num(best["score"]), f"in backtests, units of {target}"),
        ("Best model", best["name"], None),
        ("Forecast horizon", f"{len(future['x'])} {unit}s", None),
    ]
    if grouped:
        tiles.append(
            (
                "Groups forecast",
                str(info_json["options"].get("groups")),
                info_json["options"].get("group"),
            )
        )
    baseline_gain = board.loc[board["key"] == best["key"], "vs_baseline"]
    if len(baseline_gain) and pd.notna(baseline_gain.iloc[0]) and best["key"] not in BASELINES:
        tiles.append(
            (
                "vs simple rule",
                f"{fmt_pct(float(baseline_gain.iloc[0]), 0)} better",
                "lower error than the best baseline",
            )
        )
    points = evaluation.get("insights", []) + evaluation.get("notes", [])
    body = stat_tiles(tiles) + (
        f'<div class="summary">{sentences_list(points)}</div>' if points else ""
    )
    body = section("Results", body, anchor="results")
    forecast_rows = [
        [pd.Timestamp(x, unit="ms").strftime("%Y-%m-%d %H:%M").replace(" 00:00", ""), y, lo, hi]
        for x, y, lo, hi in zip(future["x"], future["y"], future["lo"], future["hi"], strict=False)
    ]
    cards = [
        card(
            chart("forecast", 340)
            + as_table_view(
                table(
                    ["Date", "Forecast", "Low (80%)", "High (80%)"],
                    forecast_rows,
                    numeric={1, 2, 3},
                )
            ),
            f"{'Total across groups' if grouped else target}: history and forecast",
            note="The shaded band is where the value should land 8 times out of 10, judged from backtest errors."
            + (
                " For groups, the total's range is the sum of the groups' ranges, so it's on the wide side."
                if grouped
                else ""
            ),
            wide=True,
        )
    ]
    for i, panel in enumerate(evaluation.get("groups", [])[:4]):
        chart_id = f"group{i}"
        charts[chart_id] = _line_spec(panel["history"], panel["forecast"], target, height=240)
        cards.append(
            card(
                chart(chart_id, 240), f"{evaluation.get('group_column', 'group')} = {panel['name']}"
            )
        )
    body += section("Forecast", grid(*cards), anchor="forecast")
    items = [{"label": r["model"], "value": r["mae"]} for _, r in board.iterrows()]
    muted = {i for i, (_, r) in enumerate(board.iterrows()) if r["key"] in BASELINES}
    backtest_cards = [
        card(
            bar_list(items, emphasis={0}, muted=muted),
            "Average error (MAE, lower is better)",
            note="Grey bars are simple rules every model should beat.",
        )
    ]
    backtest = evaluation.get("backtest")
    if backtest:
        bt_values = backtest["actual"] + backtest["predicted"]
        bt_low, bt_high = min(bt_values), max(bt_values)
        bt_pad = (bt_high - bt_low) * 0.08 or 1.0
        charts["backtest"] = {
            "type": "line",
            "series": [
                {
                    "x": backtest["x"],
                    "y": backtest["actual"],
                    "role": "context",
                    "name": "what happened",
                },
                {
                    "x": backtest["x"],
                    "y": backtest["predicted"],
                    "role": "accent",
                    "name": "what the model forecast",
                },
            ],
            "xDomain": [backtest["x"][0], backtest["x"][-1]],
            "yDomain": [bt_low - bt_pad, bt_high + bt_pad],
            "xFmt": "time",
            "xLabel": "date",
            "yLabel": target,
        }
        backtest_cards.append(
            card(
                chart("backtest"),
                "The last backtest window",
                note="The winning model forecasting a period it hadn't seen.",
            )
        )
    headers = [
        "Model",
        "MAE",
        *(["Relative error"] if grouped else ["RMSE", "MAPE", "Bias"]),
        "Seconds",
    ]
    board_rows = []
    for _, r in board.iterrows():
        middle = (
            [r["relative_error"]]
            if grouped
            else [r["rmse"], fmt_pct(r["mape"], 1) if pd.notna(r.get("mape")) else "—", r["bias"]]
        )
        board_rows.append([r["model"], r["mae"], *middle, r["seconds"]])
    backtest_cards.append(
        card(
            table(headers, board_rows, numeric=set(range(1, len(headers))), highlight=0),
            "All scores",
            wide=True,
        )
    )
    body += section(
        "How the models did in backtests",
        grid(*backtest_cards),
        intro=f"Each model forecast the last {evaluation.get('folds', 1)} windows of {len(future['x'])} {unit}s using only earlier data.",
        anchor="backtest",
    )
    title = f"Forecast: {target}"
    subtitle = f"time-series forecasting · {Path(str(info_json['data']['source'])).name} · {info_json['data'].get('periods', '')} {unit}s of history · {info_json['created'][:16].replace('T', ' ')}"
    return page(
        title,
        subtitle,
        body,
        charts,
        [("results", "Results"), ("forecast", "Forecast"), ("backtest", "Backtests")],
    )
