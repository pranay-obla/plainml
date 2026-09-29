"""``plainml importance``: which columns matter, measured many different ways.

Methods come in four families, and they disagree for good reasons, which is why a
consensus of several is more trustworthy than any one:

- **filters** score each column on its own against the target (fast; miss interactions)
- **model-based** read a fitted model's own view (tree splits, linear weights)
- **model-agnostic** measure what happens to a model's score without the column
- **searches** try combinations of columns (slower; the most direct answer)

Everything works on the original columns: a category that one-hot encodes into ten
columns is scored as one column. The result is a consensus ranking, a curve answering
"how many columns do I need?", and warnings about columns that duplicate each other.
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVC

from plainml import __version__
from plainml.console import console, esc, fmt_num, heading, info, note, quiet, success, warn
from plainml.errors import PlainMLError, did_you_mean, find_column, is_installed
from plainml.io import describe_source, fingerprint, load_data, save_table, source_stem
from plainml.metrics import Metric, SafeScorer, metrics_for, resolve_metric
from plainml.preprocessing import build_preprocessor, describe_features
from plainml.profiling import build_profile, redact_columns, shown_source
from plainml.runs import (
    EVALUATION_FILE,
    REPORT_FILE,
    RUN_FILE,
    create_run_dir,
    environment,
    write_json,
)
from plainml.schema import Schema, coerce_numeric, infer_schema
from plainml.tasks import CLASSIFICATION, REGRESSION, prepare_target

C, R = CLASSIFICATION, REGRESSION
MAX_ROWS = 20_000  # rows used by filters and model-based methods
SEARCH_ROWS = 2_000  # rows used by the (much slower) search methods
CURVE_ROWS = 5_000
RANKINGS_FILE = "rankings.csv"
SELECTED_FILE = "selected_columns.txt"


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    family: str
    about: str
    default: bool = False
    tasks: tuple[str, ...] = (C, R)
    max_columns: int | None = None  # above this, runs on the consensus top-N instead

    @property
    def short(self) -> str:
        return SHORT.get(self.key, self.key)


METHODS: list[Method] = [
    Method(
        "mutual_info",
        "Mutual information",
        "filter",
        "How much knowing the column reduces uncertainty about the target; catches curved relationships.",
        True,
    ),
    Method(
        "f_test",
        "F-test (ANOVA)",
        "filter",
        "How strongly the target differs across the column's values, assuming straight-line relationships.",
        True,
    ),
    Method(
        "correlation",
        "Rank correlation",
        "filter",
        "How consistently the target rises or falls as the column rises (Spearman).",
        True,
    ),
    Method(
        "chi2",
        "Chi-squared",
        "filter",
        "How dependent the classes are on the column (classification only).",
        False,
        tasks=(C,),
    ),
    Method(
        "variance",
        "Variance",
        "filter",
        "How much the column varies at all. Ignores the target: a sanity check, not a verdict.",
        False,
    ),
    Method(
        "random_forest",
        "Random forest",
        "model",
        "How much a random forest's splits on the column reduce its errors.",
        True,
    ),
    Method(
        "extra_trees",
        "Extra trees",
        "model",
        "The same idea with more randomised trees; less biased towards columns with many values.",
        False,
    ),
    Method(
        "gradient_boosting",
        "Gradient boosting",
        "model",
        "How much a boosted model's splits on the column reduce its errors.",
        False,
    ),
    Method(
        "l1",
        "L1 (lasso) weights",
        "model",
        "Which columns a sparse linear model keeps; unhelpful ones get a weight of exactly zero.",
        True,
    ),
    Method(
        "linear",
        "Linear weights",
        "model",
        "The size of each column's weight in a linear model on standardised data.",
        False,
    ),
    Method(
        "permutation",
        "Permutation",
        "agnostic",
        "How much a model's score on unseen rows drops when the column is shuffled.",
        True,
    ),
    Method(
        "drop_column",
        "Drop-column",
        "agnostic",
        "How much the score drops when the model is retrained without the column. Slow, but the most direct answer.",
        False,
        max_columns=40,
    ),
    Method(
        "shap",
        "SHAP",
        "agnostic",
        "The column's average contribution to individual predictions (needs the shap package).",
        False,
    ),
    Method(
        "rfe",
        "Recursive elimination (RFE)",
        "search",
        "Repeatedly removes the weakest column; the ones removed last matter most.",
        True,
    ),
    Method(
        "forward",
        "Forward selection",
        "search",
        "Builds a set one column at a time, always adding the one that helps the score most.",
        False,
        max_columns=30,
    ),
    Method(
        "backward",
        "Backward elimination",
        "search",
        "Starts with every column and removes the least useful one at a time.",
        False,
        max_columns=30,
    ),
    Method(
        "exhaustive",
        "Exhaustive search (EFS)",
        "search",
        "Scores every combination of columns and credits columns found in the best ones.",
        False,
        max_columns=10,
    ),
    Method(
        "boruta",
        "Boruta",
        "search",
        "Keeps columns that repeatedly beat shuffled 'shadow' copies of themselves in random forests.",
        True,
    ),
    Method(
        "stability",
        "Stability selection",
        "search",
        "How often a sparse model picks the column across many random half-samples of the rows.",
        False,
    ),
]
BY_KEY = {m.key: m for m in METHODS}
SHORT = {
    "mutual_info": "MI",
    "f_test": "F",
    "correlation": "Corr",
    "chi2": "Chi²",
    "variance": "Var",
    "random_forest": "RF",
    "extra_trees": "ET",
    "gradient_boosting": "GB",
    "l1": "L1",
    "linear": "Lin",
    "permutation": "Perm",
    "drop_column": "Drop",
    "shap": "SHAP",
    "rfe": "RFE",
    "forward": "Fwd",
    "backward": "Bwd",
    "exhaustive": "EFS",
    "boruta": "Boruta",
    "stability": "Stab",
}
_ALIASES = {
    "mi": "mutual_info",
    "mutual_information": "mutual_info",
    "anova": "f_test",
    "f": "f_test",
    "spearman": "correlation",
    "corr": "correlation",
    "rf": "random_forest",
    "model": "random_forest",
    "et": "extra_trees",
    "gbm": "gradient_boosting",
    "lasso": "l1",
    "coef": "linear",
    "perm": "permutation",
    "drop": "drop_column",
    "sfs": "forward",
    "sequential": "forward",
    "sbs": "backward",
    "efs": "exhaustive",
    "all_subsets": "exhaustive",
}


def resolve_methods(methods: list[str] | None, task: str) -> list[str]:
    """Method keys to run: defaults, 'all', or a named list (aliases allowed)."""
    if not methods:
        chosen = [m.key for m in METHODS if m.default]
    elif [m.strip().lower() for m in methods] == ["all"]:
        chosen = [m.key for m in METHODS]
    else:
        chosen = []
        for raw in methods:
            key = raw.strip().lower().replace("-", "_")
            key = key if key in BY_KEY else _ALIASES.get(key, key)
            if key not in BY_KEY:
                raise PlainMLError(
                    f"Unknown importance method '{raw}'.{did_you_mean(key, [*BY_KEY, *_ALIASES])}",
                    hint="Methods: " + ", ".join(BY_KEY) + " (or 'all').",
                )
            if key not in chosen:
                chosen.append(key)
    return [k for k in chosen if task in BY_KEY[k].tasks]


def subset_schema(schema: Schema, columns: list[str]) -> Schema:
    keep = set(columns)
    return Schema(
        numeric=[c for c in schema.numeric if c in keep],
        categorical=[c for c in schema.categorical if c in keep],
        datetime=[c for c in schema.datetime if c in keep],
        text=[c for c in schema.text if c in keep],
        order=[c for c in schema.order if c in keep],
    )


# --- the shared workspace ----------------------------------------------------------------


@dataclass
class Workspace:
    """Everything the methods share: encoded data, where each column's pieces are, a metric."""

    task: str
    columns: list[str]
    matrix: np.ndarray
    y: np.ndarray
    blocks: dict[str, list[int]]
    metric: Metric
    seed: int
    schema: Schema
    raw: pd.DataFrame

    def aggregate(self, values: np.ndarray, how: str = "max") -> pd.Series:
        values = np.nan_to_num(np.abs(np.asarray(values, dtype=float)))
        total = how == "sum"
        return pd.Series(
            {
                c: float(values[idx].sum() if total else values[idx].max()) if idx else 0.0
                for c, idx in self.blocks.items()
            }
        )

    def columns_of(self, names: list[str]) -> np.ndarray:
        return np.concatenate([self.blocks[c] for c in names]) if names else np.array([], dtype=int)

    def small(self) -> Workspace:
        """A smaller sample for the expensive search methods."""
        if len(self.y) <= SEARCH_ROWS:
            return self
        rng = np.random.default_rng(self.seed)
        index = np.sort(rng.choice(len(self.y), SEARCH_ROWS, replace=False))
        return Workspace(
            self.task,
            self.columns,
            self.matrix[index],
            self.y[index],
            self.blocks,
            self.metric,
            self.seed,
            self.schema,
            self.raw.iloc[index],
        )

    def forest(self, trees: int = 200, extra: bool = False) -> Any:
        if self.task == C:
            model = ExtraTreesClassifier if extra else RandomForestClassifier
            return model(
                n_estimators=trees, n_jobs=-1, random_state=self.seed, class_weight="balanced"
            )
        model = ExtraTreesRegressor if extra else RandomForestRegressor
        return model(n_estimators=trees, n_jobs=-1, random_state=self.seed)

    def evaluator(self) -> Any:
        """A quick, decent model used to score subsets of columns."""
        if self.task == C:
            return ExtraTreesClassifier(
                n_estimators=60, min_samples_leaf=2, n_jobs=-1, random_state=self.seed
            )
        return ExtraTreesRegressor(
            n_estimators=60, min_samples_leaf=2, n_jobs=-1, random_state=self.seed
        )

    def cv(self) -> Any:
        if self.task == C:
            smallest = int(np.bincount(self.y).min())
            return StratifiedKFold(
                n_splits=max(2, min(3, smallest)), shuffle=True, random_state=self.seed
            )
        return KFold(n_splits=3, shuffle=True, random_state=self.seed)

    def score(self, names: list[str]) -> float:
        """Cross-validated score using only these columns (higher is always better here)."""
        if not names:
            return self._null_score()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = cross_val_score(
                self.evaluator(),
                self.matrix[:, self.columns_of(names)],
                self.y,
                cv=self.cv(),
                scoring=SafeScorer(self.metric.scorer()),
            )
        return float(np.nanmean(values))

    def _null_score(self) -> float:
        from sklearn.dummy import DummyClassifier, DummyRegressor

        dummy = DummyClassifier(strategy="prior") if self.task == C else DummyRegressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = cross_val_score(
                dummy,
                np.zeros((len(self.y), 1)),
                self.y,
                cv=self.cv(),
                scoring=SafeScorer(self.metric.scorer()),
            )
        return float(np.nanmean(values))


def build_workspace(X: pd.DataFrame, y: Any, schema: Schema, task: str, seed: int) -> Workspace:
    if len(X) > MAX_ROWS:
        index = X.sample(MAX_ROWS, random_state=seed).index
        X, y = X.loc[index], y.loc[index]
    preprocess = build_preprocessor(schema, text_features=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = np.asarray(preprocess.fit_transform(X), dtype=float)
    origins = [origin for origin, _ in describe_features(preprocess)]
    blocks: dict[str, list[int]] = {c: [] for c in schema.features}
    for i, origin in enumerate(origins):
        blocks.setdefault(origin, []).append(i)
    target = (
        LabelEncoder().fit_transform(y.astype(str)) if task == C else np.asarray(y, dtype=float)
    )
    metric = resolve_metric(None, task, metrics_for(task, y))
    return Workspace(task, schema.features, matrix, target, blocks, metric, seed, schema, X)


# --- the methods -------------------------------------------------------------------------
# Each returns a score per column (bigger = more important) and optional details.


def _mutual_info(ws: Workspace) -> tuple[pd.Series, dict]:
    func = mutual_info_classif if ws.task == C else mutual_info_regression
    return ws.aggregate(func(ws.matrix, ws.y, random_state=ws.seed)), {}


def _f_test(ws: Workspace) -> tuple[pd.Series, dict]:
    func = f_classif if ws.task == C else f_regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ws.aggregate(func(ws.matrix, ws.y)[0]), {}


def _correlation(ws: Workspace) -> tuple[pd.Series, dict]:
    ranked = pd.DataFrame(ws.matrix).rank().to_numpy()
    targets = [ws.y] if ws.task == R else [(ws.y == k).astype(float) for k in np.unique(ws.y)]
    best = np.zeros(ws.matrix.shape[1])
    for target in targets:
        t = pd.Series(target).rank().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.array(
                [
                    np.corrcoef(ranked[:, j], t)[0, 1] if ranked[:, j].std() else 0.0
                    for j in range(ranked.shape[1])
                ]
            )
        best = np.maximum(best, np.abs(np.nan_to_num(corr)))
    return ws.aggregate(best), {}


def _chi2(ws: Workspace) -> tuple[pd.Series, dict]:
    return ws.aggregate(chi2(MinMaxScaler().fit_transform(ws.matrix), ws.y)[0]), {}


def _variance(ws: Workspace) -> tuple[pd.Series, dict]:
    scores = {}
    for column in ws.columns:
        values = ws.raw[column]
        if column in ws.schema.numeric:
            numeric = coerce_numeric(values).dropna()
            span = numeric.max() - numeric.min() if len(numeric) else 0
            scores[column] = float(((numeric - numeric.min()) / span).var()) if span else 0.0
        else:  # categories, text, dates: how evenly spread the values are
            shares = values.astype(str).value_counts(normalize=True)
            scores[column] = float(1 - shares.iloc[0]) if len(shares) else 0.0
    return pd.Series(scores), {}


def _shadow_corrected(ws: Workspace, model: Any, columns: list[str] | None = None) -> pd.Series:
    """Tree importance minus that of a shuffled copy of the same column.

    Split-based importance favours columns made of many pieces (a free-text column is
    100 word features, each picking up a little importance by chance). Subtracting a
    shuffled 'shadow' copy of each column removes that bias: pure noise scores about zero.
    """
    columns = columns or ws.columns
    rng = np.random.default_rng(ws.seed)
    real_part = ws.matrix[:, ws.columns_of(columns)]
    shadow = real_part.copy()
    position = 0
    spans = {}
    for column in columns:
        width = len(ws.blocks[column])
        span = list(range(position, position + width))
        shadow[:, span] = shadow[rng.permutation(len(shadow))][:, span]
        spans[column] = span
        position += width
    importances = model.fit(np.hstack([real_part, shadow]), ws.y).feature_importances_
    offset = real_part.shape[1]
    corrected = {
        c: float(importances[span].sum() - importances[[offset + i for i in span]].sum())
        for c, span in spans.items()
    }
    return pd.Series(corrected).clip(lower=0)


def _random_forest(ws: Workspace) -> tuple[pd.Series, dict]:
    return _shadow_corrected(ws, ws.forest()), {}


def _extra_trees(ws: Workspace) -> tuple[pd.Series, dict]:
    return _shadow_corrected(ws, ws.forest(extra=True)), {}


def _gradient_boosting(ws: Workspace) -> tuple[pd.Series, dict]:
    small = ws.small() if len(ws.y) > 5000 else ws
    model = GradientBoostingClassifier if ws.task == C else GradientBoostingRegressor
    return _shadow_corrected(small, model(random_state=ws.seed)), {}


def _l1_weights(ws: Workspace, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ws.task == C:
            model = LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=5000).fit(X, y)
            return np.abs(np.atleast_2d(model.coef_)).sum(axis=0)
        return np.abs(LassoCV(cv=3, random_state=ws.seed, n_jobs=-1).fit(X, y).coef_)


def _l1(ws: Workspace) -> tuple[pd.Series, dict]:
    return ws.aggregate(_l1_weights(ws, ws.matrix, ws.y)), {}


def _linear(ws: Workspace) -> tuple[pd.Series, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ws.task == C:
            weights = np.abs(
                np.atleast_2d(LogisticRegression(max_iter=3000).fit(ws.matrix, ws.y).coef_)
            ).sum(axis=0)
        else:
            weights = np.abs(RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(ws.matrix, ws.y).coef_)
    return ws.aggregate(weights), {}


def _permutation(ws: Workspace) -> tuple[pd.Series, dict]:
    stratify = ws.y if ws.task == C else None
    try:
        X_fit, X_test, y_fit, y_test = train_test_split(
            ws.matrix, ws.y, test_size=0.3, random_state=ws.seed, stratify=stratify
        )
    except ValueError:
        X_fit, X_test, y_fit, y_test = train_test_split(
            ws.matrix, ws.y, test_size=0.3, random_state=ws.seed
        )
    model = (HistGradientBoostingClassifier if ws.task == C else HistGradientBoostingRegressor)(
        random_state=ws.seed
    )
    scorer = SafeScorer(ws.metric.scorer())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_fit, y_fit)
        base = scorer(model, X_test, y_test)
        rng = np.random.default_rng(ws.seed)
        drops = {}
        for column in ws.columns:
            idx = ws.blocks[column]
            losses = []
            for _ in range(5):
                shuffled = X_test.copy()
                order = rng.permutation(len(shuffled))
                shuffled[:, idx] = shuffled[order][:, idx]  # the column's pieces move together
                losses.append(base - scorer(model, shuffled, y_test))
            drops[column] = float(np.nanmean(losses))
    return pd.Series(drops).clip(lower=0), {}


def _drop_column(ws: Workspace) -> tuple[pd.Series, dict]:
    small = ws.small()
    full = small.score(small.columns)
    return pd.Series(
        {
            c: max(0.0, full - small.score([o for o in small.columns if o != c]))
            for c in small.columns
        }
    ), {"full_score": full}


def _shap(ws: Workspace) -> tuple[pd.Series, dict]:
    import shap

    small = ws.small()
    model = small.forest(trees=100).fit(small.matrix, small.y)
    sample = small.matrix[: min(500, len(small.matrix))]
    values = shap.TreeExplainer(model).shap_values(sample, check_additivity=False)
    values = np.abs(
        np.asarray(values if not isinstance(values, list) else np.stack(values, axis=-1))
    )
    if values.ndim == 3:
        values = values.mean(axis=2) if values.shape[1] == sample.shape[1] else values.mean(axis=0)
    return ws.aggregate(values.mean(axis=0), "sum"), {}


def _rfe(ws: Workspace) -> tuple[pd.Series, dict]:
    small = ws.small()
    remaining = list(small.columns)
    removed_order: list[str] = []
    while len(remaining) > 1:
        per_column = _shadow_corrected(small, small.forest(trees=100), remaining)
        step = max(1, len(remaining) // 10)
        weakest = sorted(remaining, key=lambda c: per_column[c])[:step]
        for column in weakest:
            remaining.remove(column)
            removed_order.append(column)
    removed_order.extend(remaining)
    return pd.Series({c: float(i + 1) for i, c in enumerate(removed_order)}), {
        "order": removed_order[::-1]
    }


def _forward(ws: Workspace, columns: list[str]) -> tuple[pd.Series, dict]:
    small = ws.small()
    chosen: list[str] = []
    path = []
    left = list(columns)
    while left:
        scores = {c: small.score([*chosen, c]) for c in left}
        best = max(scores, key=lambda c: scores[c])
        chosen.append(best)
        left.remove(best)
        path.append(
            {"columns": len(chosen), "added": best, "score": ws.metric.display(scores[best])}
        )
    return pd.Series({c: float(len(chosen) - i) for i, c in enumerate(chosen)}), {"path": path}


def _backward(ws: Workspace, columns: list[str]) -> tuple[pd.Series, dict]:
    small = ws.small()
    current = list(columns)
    removed: list[str] = []
    path: list[dict[str, Any]] = [
        {"columns": len(current), "removed": None, "score": ws.metric.display(small.score(current))}
    ]
    while len(current) > 1:
        scores = {c: small.score([o for o in current if o != c]) for c in current}
        least = max(scores, key=lambda c: scores[c])  # removing it hurts least
        current.remove(least)
        removed.append(least)
        path.append(
            {"columns": len(current), "removed": least, "score": ws.metric.display(scores[least])}
        )
    removed.extend(current)
    return pd.Series({c: float(i + 1) for i, c in enumerate(removed)}), {"path": path}


def _exhaustive(ws: Workspace, columns: list[str]) -> tuple[pd.Series, dict]:
    small = ws.small()
    results = []
    for size in range(1, len(columns) + 1):
        for combo in itertools.combinations(columns, size):
            results.append((small.score(list(combo)), combo))
    results.sort(key=lambda r: -r[0])
    top = results[: max(1, len(results) // 20)]  # the best 5% of combinations
    credit = {c: sum(1 for _, combo in top if c in combo) / len(top) for c in columns}
    best_by_size: dict[int, tuple[float, tuple[str, ...]]] = {}
    for score, combo in results:
        best_by_size.setdefault(len(combo), (score, combo))
    return pd.Series(credit), {
        "tried": len(results),
        "best": {"columns": list(results[0][1]), "score": ws.metric.display(results[0][0])},
        "best_by_size": [
            {"columns": size, "subset": list(combo), "score": ws.metric.display(score)}
            for size, (score, combo) in sorted(best_by_size.items())
        ],
    }


def _boruta(ws: Workspace, rounds: int = 20) -> tuple[pd.Series, dict]:
    from scipy.stats import binomtest

    small = ws.small()
    rng = np.random.default_rng(ws.seed)
    hits = dict.fromkeys(small.columns, 0)
    width = {c: len(small.blocks[c]) for c in small.columns}
    for _ in range(rounds):
        shadow = small.matrix.copy()
        for column in small.columns:  # shuffle each column's pieces together
            idx = small.blocks[column]
            shadow[:, idx] = shadow[rng.permutation(len(shadow))][:, idx]
        combined = np.hstack([small.matrix, shadow])
        model = small.forest(trees=100)
        model.set_params(max_depth=7, random_state=int(rng.integers(1_000_000)))
        importances = model.fit(combined, small.y).feature_importances_
        real = small.aggregate(importances[: small.matrix.shape[1]], "sum")
        fake = small.aggregate(importances[small.matrix.shape[1] :], "sum")
        for column in small.columns:
            # Beat the best shadow that's no wider than this column: a 100-word text
            # column's shadow gathers importance by sheer size, and shouldn't set the bar
            # for a single number.
            rivals = [fake[o] for o in small.columns if width[o] <= width[column]]
            hits[column] += int(real[column] > max(rivals))
    decisions = {}
    for column, count in hits.items():
        if binomtest(count, rounds, 0.5, alternative="greater").pvalue < 0.05:
            decisions[column] = "confirmed"
        elif binomtest(count, rounds, 0.5, alternative="less").pvalue < 0.05:
            decisions[column] = "rejected"
        else:
            decisions[column] = "tentative"
    return pd.Series({c: hits[c] / rounds for c in small.columns}), {
        "decisions": decisions,
        "rounds": rounds,
    }


def _stability(ws: Workspace, rounds: int = 50) -> tuple[pd.Series, dict]:
    small = ws.small() if len(ws.y) > 5000 else ws
    rng = np.random.default_rng(ws.seed)
    picked = dict.fromkeys(small.columns, 0)
    alpha = None
    if ws.task == R:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = LassoCV(cv=3, random_state=ws.seed).fit(small.matrix, small.y).alpha_
    for _ in range(rounds):
        index = rng.choice(len(small.y), len(small.y) // 2, replace=False)
        if ws.task == C and len(np.unique(small.y[index])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ws.task == C:
                weights = _l1_weights(small, small.matrix[index], small.y[index])
            else:
                weights = np.abs(
                    Lasso(alpha=alpha, max_iter=5000).fit(small.matrix[index], small.y[index]).coef_
                )
        chosen = small.aggregate(weights)
        for column in small.columns:
            picked[column] += int(chosen[column] > 1e-8)
    return pd.Series({c: picked[c] / rounds for c in small.columns}), {"rounds": rounds}


SIMPLE = {
    "mutual_info": _mutual_info,
    "f_test": _f_test,
    "correlation": _correlation,
    "chi2": _chi2,
    "variance": _variance,
    "random_forest": _random_forest,
    "extra_trees": _extra_trees,
    "gradient_boosting": _gradient_boosting,
    "l1": _l1,
    "linear": _linear,
    "permutation": _permutation,
    "drop_column": _drop_column,
    "shap": _shap,
    "rfe": _rfe,
    "boruta": _boruta,
    "stability": _stability,
}
SUBSET = {"forward": _forward, "backward": _backward, "exhaustive": _exhaustive}


# --- redundancy ------------------------------------------------------------------------------


def redundancy(ws: Workspace, threshold: float = 0.9) -> dict[str, Any]:
    """Groups of numeric columns that move together, and variance inflation factors."""
    from plainml.schema import parse_dates

    numeric = [c for c in ws.columns if c in ws.schema.numeric or c in ws.schema.datetime]
    frame = pd.DataFrame(
        {
            c: coerce_numeric(ws.raw[c])
            if c in ws.schema.numeric
            else parse_dates(ws.raw[c])
            .map(lambda d: d.toordinal() if pd.notna(d) else np.nan)
            .astype(float)
            for c in numeric
        }
    )
    groups: list[list[str]] = []
    pairs = []
    if len(numeric) >= 2:
        corr = frame.corr(method="spearman").abs()
        seen: set[str] = set()
        for column in numeric:
            if column in seen:
                continue
            group = [column] + [
                o
                for o in numeric
                if o != column and o not in seen and corr.loc[column, o] >= threshold
            ]
            if len(group) > 1:
                groups.append(group)
                seen.update(group)
        for a, b in itertools.combinations(numeric, 2):
            if corr.loc[a, b] >= threshold:
                pairs.append({"a": a, "b": b, "correlation": float(corr.loc[a, b])})
    vif = {}
    if 2 <= len(numeric) <= 60:
        filled = frame.fillna(frame.median()).to_numpy()
        filled = (filled - filled.mean(axis=0)) / (filled.std(axis=0) + 1e-12)
        for i, column in enumerate(numeric):
            others = np.delete(filled, i, axis=1)
            r2 = LinearRegression().fit(others, filled[:, i]).score(others, filled[:, i])
            vif[column] = float(min(1 / max(1 - r2, 1e-6), 1e6))
    return {"groups": groups, "pairs": pairs, "vif": vif}


# --- the curve --------------------------------------------------------------------------------


def columns_curve(
    X: pd.DataFrame, y: Any, schema: Schema, ranked: list[str], task: str, seed: int
) -> list[tuple[int, float]]:
    """Cross-validated score using the top-n columns, for a few values of n."""
    metric = resolve_metric(None, task, metrics_for(task, y))
    if len(X) > CURVE_ROWS:
        index = X.sample(CURVE_ROWS, random_state=seed).index
        X, y = X.loc[index], y.loc[index]
    sizes = sorted(
        {n for n in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89) if n < len(ranked)} | {len(ranked)}
    )
    if task == C:
        smallest = int(pd.Series(y).value_counts().min())
        cv: Any = StratifiedKFold(
            n_splits=max(2, min(3, smallest)), shuffle=True, random_state=seed
        )
        estimator: Any = HistGradientBoostingClassifier(random_state=seed, max_iter=150)
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        estimator = HistGradientBoostingRegressor(random_state=seed, max_iter=150)
    points = []
    for size in sizes:
        columns = ranked[:size]
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(subset_schema(schema, columns))),
                ("model", estimator),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                pipeline, X[columns], y, cv=cv, scoring=SafeScorer(metric.scorer())
            )
        points.append((size, metric.display(float(np.nanmean(scores)))))
    return points


def enough_columns(points: list[tuple[int, float]], metric: Metric) -> int:
    values = [p[1] for p in points]
    best = max(values) if metric.greater_is_better else min(values)
    for size, value in points:
        gap = (best - value) if metric.greater_is_better else (value - best)
        tolerance = 0.01 if metric.bounded else 0.02 * max(abs(best), 1e-9)
        if gap <= tolerance:
            return size
    return points[-1][0]


# --- running it -----------------------------------------------------------------------------


@dataclass
class ImportanceResult:
    table: pd.DataFrame
    selected: list[str]
    curve: list[tuple[int, float]]
    details: dict[str, Any]
    run_dir: Path | None = None
    methods: list[str] = field(default_factory=list)

    @property
    def report_path(self) -> Path | None:
        return self.run_dir / REPORT_FILE if self.run_dir else None

    def __repr__(self) -> str:
        return f"ImportanceResult({len(self.table)} columns ranked by {len(self.methods)} methods, {len(self.selected)} selected)"


# Methods that ignore the target describe the data, not usefulness: they don't vote.
NON_VOTING = {"variance"}


def _consensus(scores: dict[str, pd.Series], columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame({"column": columns})
    ranks = pd.DataFrame(
        {
            m: s.reindex(columns).fillna(0).rank(ascending=False, method="min")
            for m, s in scores.items()
        }
    )
    voters = [m for m in ranks.columns if m not in NON_VOTING] or list(ranks.columns)
    table["consensus_rank"] = ranks[voters].mean(axis=1).to_numpy()
    table["agreement"] = (
        1 - ranks[voters].std(axis=1).fillna(0) / max(len(columns) - 1, 1)
    ).to_numpy()
    for method, series in scores.items():
        table[f"{method}_score"] = series.reindex(columns).fillna(0).to_numpy()
        table[f"{method}_rank"] = ranks[method].to_numpy()
    table = table.sort_values(["consensus_rank", "column"]).reset_index(drop=True)
    count = len(table)
    table["consensus_score"] = 1 - (table["consensus_rank"] - 1) / max(count - 1, 1)
    return table


def _sentences(
    table: pd.DataFrame,
    methods: list[str],
    selected: list[str],
    details: dict,
    points: list,
    metric: Metric,
) -> list[str]:
    out = []
    top = table["column"].head(3).tolist()
    out.append(f"Most useful overall: {', '.join(top)}.")
    methods = [m for m in methods if m not in NON_VOTING]
    ranks = table[[f"{m}_rank" for m in methods]]
    if len(methods) >= 3:
        spread = ranks.max(axis=1) - ranks.min(axis=1)
        disputed = table.loc[spread.idxmax()]
        if spread.max() >= max(3, len(table) / 3):
            best_m = min(methods, key=lambda m: disputed[f"{m}_rank"])
            worst_m = max(methods, key=lambda m: disputed[f"{m}_rank"])
            out.append(
                f"The methods disagree most about {disputed['column']}: rank {int(disputed[f'{best_m}_rank'])} by "
                f"{BY_KEY[best_m].label}, {int(disputed[f'{worst_m}_rank'])} by {BY_KEY[worst_m].label}. "
                "That usually means its effect is non-linear or shared with other columns."
            )
    if points:
        best = max(p[1] for p in points) if metric.greater_is_better else min(p[1] for p in points)
        out.append(
            f"The top {len(selected)} of {len(table)} columns reach {metric.label} {fmt_num(dict(points)[len(selected)])} "
            f"(best with any number: {fmt_num(best)})."
        )
    decisions = details.get("boruta", {}).get("decisions")
    if decisions:
        confirmed = [c for c, d in decisions.items() if d == "confirmed"]
        rejected = [c for c, d in decisions.items() if d == "rejected"]
        out.append(
            f"Boruta confirms {len(confirmed)} column(s) as genuinely useful and rejects {len(rejected)}."
        )
    groups = details.get("redundancy", {}).get("groups")
    if groups:
        out.append(
            f"{len(groups)} group(s) of columns carry nearly the same information (e.g. {', '.join(groups[0][:3])}); "
            "importance gets split between them, so each can look weaker than the information it carries."
        )
    return out


def feature_importance(
    data: Any,
    target: str,
    *,
    methods: list[str] | None = None,
    k: int | None = None,
    output: str | Path | None = None,
    drop: list[str] | None = None,
    sample: int | float | None = None,
    seed: int = 42,
    out_dir: str | Path = "runs",
    name: str | None = None,
    report: bool = True,
    save: bool = True,
    private: bool = False,
    verbose: bool = True,
    progress: Any = None,
    **load_options: Any,
) -> ImportanceResult:
    """Rank every column with several methods; recommend a subset; save a report."""
    started = time.time()
    with quiet(not verbose):
        df = load_data(data, sample=sample, seed=seed, **load_options)
        target_column = find_column(target, df.columns, "Target column")
        kept, y, target_info = prepare_target(df, [target_column])
        task = target_info.task
        X = kept.drop(columns=[target_column])
        schema, infos = infer_schema(X, drop=[find_column(c, df.columns) for c in drop or []])
        if len(schema.features) < 2:
            raise PlainMLError("Ranking columns needs at least two usable input columns.")
        chosen = resolve_methods(methods, task)
        if not chosen:
            raise PlainMLError("None of those methods apply to this task.")
        if "shap" in chosen and not is_installed("shap"):
            note('Skipping SHAP: pip install "plainml[explain]"')
            chosen.remove("shap")
        if k is not None and not 1 <= k <= len(schema.features):
            raise PlainMLError(
                f"-k must be between 1 and {len(schema.features)} (the number of usable columns)."
            )

        heading(
            f"Ranking {len(schema.features)} columns for '{target_column}' ({task}) with {len(chosen)} methods"
        )
        ws = build_workspace(X, y, schema, task, seed)
        scores: dict[str, pd.Series] = {}
        details: dict[str, Any] = {}
        timings = {}
        consensus_order = list(schema.features)
        ordered = [m for m in chosen if m not in SUBSET] + [m for m in chosen if m in SUBSET]
        for i, method in enumerate(ordered):
            spec = BY_KEY[method]
            if progress:
                progress(i / (len(ordered) + 1), f"Running {spec.label}")
            columns = list(schema.features)
            if method in SUBSET and scores:
                consensus_order = _consensus(scores, schema.features)["column"].tolist()
            if spec.max_columns and len(columns) > spec.max_columns:
                columns = consensus_order[: spec.max_columns]
                note(
                    f"{spec.label}: using the top {spec.max_columns} columns so far (it's slow with {len(schema.features)})."
                )
            t0 = time.perf_counter()
            try:
                with console.status(f"{spec.label}…"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    series, extra = (
                        SUBSET[method](ws, columns) if method in SUBSET else SIMPLE[method](ws)
                    )
            except Exception as exc:  # one method failing shouldn't stop the others
                note(f"{spec.label} failed: {esc(str(exc).splitlines()[0][:100])}")
                continue
            timings[method] = round(time.perf_counter() - t0, 2)
            scores[method] = series.reindex(schema.features).fillna(0.0)
            if extra:
                details[method] = extra
            console.print(f"[good]✓[/] {spec.label:<32} [muted]{timings[method]:.1f}s[/]")
        if not scores:
            raise PlainMLError("Every importance method failed on this data.")
        used = list(scores)
        table = _consensus(scores, schema.features)
        decisions = details.get("boruta", {}).get("decisions")
        if decisions:
            table["boruta"] = table["column"].map(decisions)
        details["redundancy"] = redundancy(ws)
        ranked = table["column"].tolist()
        if progress:
            progress(len(ordered) / (len(ordered) + 1), "Testing how many columns you need")
        with console.status("Testing how many columns you need…"):
            points = columns_curve(X, y, schema, ranked, task, seed)
        if k is None:
            k = enough_columns(points, ws.metric)
        table["selected"] = table.index < k
        selected = ranked[:k]
        sentences = _sentences(table, used, selected, details, points, ws.metric)

    result = ImportanceResult(
        table=table, selected=selected, curve=points, details=details, methods=used
    )
    if output:
        reduced = kept[[*selected, target_column]]
        save_table(reduced, output)
    if save:
        run_dir = create_run_dir(out_dir, name or f"{source_stem(data)}-importance")
        created = datetime.now().isoformat(timespec="seconds")
        table.to_csv(run_dir / RANKINGS_FILE, index=False)
        (run_dir / SELECTED_FILE).write_text("\n".join(selected) + "\n", encoding="utf-8")
        prof = build_profile(df, schema, infos, describe_source(data), target_info, y, X)
        write_json(
            run_dir / EVALUATION_FILE,
            {
                "curve": points,
                "details": details,
                "sentences": sentences,
                "timings": timings,
                "metric_label": ws.metric.label,
            },
        )
        write_json(
            run_dir / RUN_FILE,
            {
                "kind": "importance",
                "plainml_version": __version__,
                "environment": environment(),
                "created": created,
                "task": task,
                "task_label": f"feature importance ({task})",
                "target": target_column,
                "metric": ws.metric.key,
                "metric_label": ws.metric.label,
                "metric_greater_is_better": ws.metric.greater_is_better,
                "data": {
                    "source": shown_source(describe_source(data), private),
                    "rows": len(kept),
                    "raw_rows": len(df),
                    "columns": df.shape[1],
                    "fingerprint": fingerprint(df),
                },
                "schema": schema.to_dict(),
                "profile": {
                    "issues": [i.__dict__ for i in prof.issues],
                    "columns": redact_columns([c.to_dict() for c in prof.columns])
                    if private
                    else [c.to_dict() for c in prof.columns],
                },
                "methods": used,
                "best": {
                    "key": "consensus",
                    "name": f"top {k} columns",
                    "score": dict(points).get(k),
                    "selected": selected,
                },
                "options": {
                    "methods": used,
                    "k": k,
                    "seed": seed,
                    "sample": sample,
                    "drop": drop or [],
                    "private": private,
                },
                "timings": {"total_seconds": round(time.time() - started, 2)},
                "files": {
                    "rankings": RANKINGS_FILE,
                    "selected": SELECTED_FILE,
                    "report": REPORT_FILE if report else None,
                },
            },
        )
        result.run_dir = run_dir
        if report:
            try:
                from plainml.report import write_report

                write_report(run_dir)
            except Exception as exc:
                warn(f"Couldn't write the HTML report: {esc(str(exc)[:120])}")
    if verbose:
        _render(result, sentences, ws.metric, output)
    return result


def select_features(
    data: Any,
    target: str,
    *,
    k: int | None = None,
    methods: list[str] | None = None,
    output: str | Path | None = None,
    drop: list[str] | None = None,
    seed: int = 42,
    verbose: bool = True,
    **load_options: Any,
) -> pd.DataFrame:
    """Rank columns and (optionally) save a dataset with only the selected ones.

    Returns one row per column with a ``selected`` flag. (``feature_importance`` does the
    same and also saves a run folder with a report.)
    """
    result = feature_importance(
        data,
        target,
        methods=methods,
        k=k,
        output=output,
        drop=drop,
        seed=seed,
        save=False,
        verbose=verbose,
        **load_options,
    )
    return result.table


def _render(result: ImportanceResult, sentences: list[str], metric: Metric, output: Any) -> None:
    table = result.table
    heading("Column ranking (1 = most useful)")
    grid = Table(box=None, header_style="muted", pad_edge=False)
    grid.add_column("#", justify="right", style="muted")
    grid.add_column("Column", max_width=28, overflow="ellipsis", no_wrap=True)
    shown = result.methods[:10]
    for method in shown:
        grid.add_column(BY_KEY[method].short, justify="right")
    if "boruta" in table:
        grid.add_column("Verdict", justify="center")
    grid.add_column("", justify="center")
    for i, row in table.head(30).iterrows():
        cells = [str(i + 1), esc(row["column"]), *[str(int(row[f"{m}_rank"])) for m in shown]]
        if "boruta" in table:
            decision = row["boruta"]
            cells.append({"confirmed": "[good]✓[/]", "rejected": "[muted]✗[/]"}.get(decision, "?"))
        cells.append("[good]◀ keep[/]" if row["selected"] else "")
        grid.add_row(*cells, style="" if row["selected"] else "muted")
    console.print(grid)
    if len(table) > 30:
        note(f"… and {len(table) - 30} more columns (all of them are in rankings.csv)")
    note("Key: " + ", ".join(f"{BY_KEY[m].short} = {BY_KEY[m].label}" for m in shown))
    if "boruta" in table:
        note("Verdict: Boruta's test (✓ confirmed useful, ✗ rejected, ? undecided).")
    if len(result.methods) > len(shown):
        note(
            f"Showing 10 of {len(result.methods)} methods; the report and rankings.csv have them all."
        )
    heading(f"How many columns do you need? ({metric.label}, 3-fold cross-validation)")
    for size, score in result.curve:
        marker = "[good]◀ enough[/]" if size == len(result.selected) else ""
        console.print(f"  top {size:>3} columns  {fmt_num(score):>10}  {marker}")
    heading("In plain English")
    for sentence in sentences:
        console.print(f"• {esc(sentence)}")
    if result.run_dir:
        heading("Saved")
        console.print(f"[bold]{esc(result.run_dir)}[/]")
        console.print(
            f"  {RANKINGS_FILE:<22}[muted]every column's score and rank for each method[/]"
        )
        console.print(f"  {SELECTED_FILE:<22}[muted]the recommended columns[/]")
        console.print(f"  {REPORT_FILE:<22}[muted]charts and explanations[/]")
    if output:
        success(f"Saved the {len(result.selected)} selected columns + the target to {esc(output)}")
    elif result.run_dir:
        info("Save a dataset with only these columns with -o reduced.csv")


def render_importance_report(run: Any) -> str:
    from plainml.report import (
        as_table_view,
        bar_list,
        card,
        chart,
        grid,
        heat_table,
        page,
        profile_section,
        section,
        sentences_list,
        stat_tiles,
        table,
    )

    info_json, evaluation = run.info, run.evaluation
    rankings = pd.read_csv(run.path / RANKINGS_FILE)
    methods = [m for m in info_json.get("methods", []) if f"{m}_rank" in rankings]
    selected = info_json["best"].get("selected", [])
    curve = evaluation.get("curve", [])
    details = evaluation.get("details", {})
    metric_label = evaluation.get("metric_label", info_json.get("metric_label", "score"))
    charts: dict[str, Any] = {}
    all_score = curve[-1][1] if curve else None
    tiles = [
        (
            "Columns worth keeping",
            f"{len(selected)} of {len(rankings)}",
            f"{metric_label} {fmt_num(info_json['best'].get('score'))} vs {fmt_num(all_score)} with all",
        ),
        ("Methods compared", str(len(methods)), None),
        ("Target", str(info_json["target"]), info_json.get("task", "")),
    ]
    body = (
        stat_tiles(tiles, hero_first=True)
        + f'<div class="summary">{sentences_list(evaluation.get("sentences", []))}</div>'
    )
    body = section("Results", body, anchor="results")

    top = rankings.head(25)
    items = [
        {
            "label": r.column,
            "value": float(r.consensus_score),
            "tip": f"average rank {r.consensus_rank:.1f}",
        }
        for r in top.itertuples()
    ]
    emphasis = {i for i, r in enumerate(top.itertuples()) if r.column in selected}
    consensus_card = card(
        bar_list(items, emphasis=emphasis, muted=set(range(len(items))) - emphasis),
        "Consensus ranking",
        note="1.0 = ranked first by every method. Blue columns are the recommended set.",
        wide=True,
    )
    count = len(rankings)
    rows = [str(c) for c in top["column"]]
    cols = [BY_KEY[m].short for m in methods]
    cells = [
        [(str(int(r[f"{m}_rank"])), 1 - (r[f"{m}_rank"] - 1) / max(count - 1, 1)) for m in methods]
        for _, r in top.iterrows()
    ]
    heat_card = card(
        heat_table(rows, cols, cells, corner="column ↓ · method →"),
        "Rank by method",
        note="Darker = ranked higher by that method. Columns where the shading varies a lot across a row are the ones methods disagree about. "
        + "Key: "
        + ", ".join(f"{BY_KEY[m].short} = {BY_KEY[m].label}" for m in methods)
        + ".",
        wide=True,
    )
    cards = [consensus_card]
    if curve:
        charts["curve"] = {
            "type": "line",
            "series": [{"x": [p[0] for p in curve], "y": [p[1] for p in curve]}],
            "xDomain": [curve[0][0], curve[-1][0]],
            "yDomain": _pad([p[1] for p in curve]),
            "xLabel": "number of top columns used",
            "yLabel": metric_label,
            "endLabel": True,
        }
        cards.append(
            card(
                chart("curve")
                + as_table_view(
                    table(["Columns", metric_label], [[p[0], p[1]] for p in curve], numeric={0, 1})
                ),
                "How many columns do you need?",
                note="Score using only the top-n columns. Where the curve flattens, extra columns stop helping.",
            )
        )
    cards.append(heat_card)
    body += section("Rankings", grid(*cards), anchor="rankings")

    method_cards = []
    for method in methods:
        spec = BY_KEY[method]
        ordered = rankings.sort_values(f"{method}_rank").head(12)
        items = [
            {"label": r["column"], "value": float(r[f"{method}_score"])}
            for _, r in ordered.iterrows()
        ]
        method_cards.append(
            card(bar_list(items, emphasis=set(range(len(items)))), spec.label, note=esc(spec.about))
        )
    body += section(
        "Each method's view",
        grid(*method_cards),
        intro="The top columns according to each method, with its own scale.",
        anchor="methods",
    )

    extra_cards = []
    boruta = details.get("boruta", {}).get("decisions")
    if boruta:
        order = {"confirmed": 0, "tentative": 1, "rejected": 2}
        rows_b = sorted(boruta.items(), key=lambda kv: (order[kv[1]], kv[0]))
        extra_cards.append(
            card(
                table(["Column", "Boruta verdict"], [[c, d] for c, d in rows_b]),
                "Boruta: genuinely useful?",
                note="Confirmed columns beat shuffled copies of every column in most of the rounds.",
            )
        )
    exhaustive = details.get("exhaustive")
    if exhaustive:
        best_rows = [
            [e["columns"], ", ".join(e["subset"]), e["score"]] for e in exhaustive["best_by_size"]
        ]
        extra_cards.append(
            card(
                table(["Columns", "Best combination", metric_label], best_rows, numeric={0, 2}),
                f"Exhaustive search ({exhaustive['tried']:,} combinations)",
                wide=True,
            )
        )
    for key, verb in (("forward", "added"), ("backward", "removed")):
        path = details.get(key, {}).get("path")
        if path:
            rows_p = [[p["columns"], p.get(verb) or "—", p["score"]] for p in path]
            extra_cards.append(
                card(
                    table(["Columns", f"Column {verb}", metric_label], rows_p, numeric={0, 2}),
                    BY_KEY[key].label,
                )
            )
    redundancy_info = details.get("redundancy", {})
    if redundancy_info.get("groups"):
        extra_cards.append(
            card(
                sentences_list([", ".join(g) for g in redundancy_info["groups"]]),
                "Columns that duplicate each other",
                note="Rank correlation of 0.9 or more. Keeping one from each group is usually enough.",
            )
        )
    vif = redundancy_info.get("vif") or {}
    if vif:
        vif_rows = sorted(vif.items(), key=lambda kv: -kv[1])[:20]
        extra_cards.append(
            card(
                table(["Column", "VIF"], [[c, v] for c, v in vif_rows], numeric={1}),
                "Multicollinearity (VIF)",
                note="Above 5–10, a column is largely predictable from the others, so linear weights for it are unreliable.",
            )
        )
    if extra_cards:
        body += section("Details", grid(*extra_cards), anchor="details")
    body += section(
        "The data",
        profile_section(info_json.get("profile", {}), info_json.get("schema")),
        anchor="data",
    )
    body += section(
        "All scores",
        card(
            table(
                list(rankings.columns),
                rankings.round(4).values.tolist(),
                numeric=set(range(1, len(rankings.columns))),
            ),
            wide=True,
        ),
        anchor="all",
    )
    title = f"What matters for {info_json['target']}"
    subtitle = f"feature importance · {Path(str(info_json['data']['source'])).name} · {info_json['data']['rows']:,} rows · {info_json['created'][:16].replace('T', ' ')}"
    return page(
        title,
        subtitle,
        body,
        charts,
        [
            ("results", "Results"),
            ("rankings", "Rankings"),
            ("methods", "Methods"),
            ("details", "Details"),
            ("data", "Data"),
        ],
    )


def _pad(values: list[float]) -> list[float]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return [0.0, 1.0]
    low, high = min(finite), max(finite)
    pad = (high - low) * 0.15 or abs(high) * 0.05 or 1.0
    return [low - pad, high + pad]
