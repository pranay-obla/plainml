"""Training: try many models with cross-validation, pick the best, explain it and save it.

The flow for ``plainml train``:

1. load the data once and profile it (types, gaps, leakage suspects, imbalance)
2. hold out a test set (stratified for classification) that no model sees during selection
3. cross-validate every candidate model on the rest, including a do-nothing baseline
4. optionally average the top three into an ensemble
5. evaluate the winner on the held-out test set, measure which columns matter
6. retrain the winner on all the data and save it with a report
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_validate,
    train_test_split,
)

from plainml import __version__
from plainml.card import CARD_FILE, write_model_card
from plainml.console import (
    console,
    esc,
    fmt_duration,
    fmt_num,
    fmt_pct,
    heading,
    info,
    note,
    quiet,
    warn,
)
from plainml.drift import describe_distribution
from plainml.errors import PlainMLError, did_you_mean, find_column, require
from plainml.evaluation import calibration_verdict, evaluate_model
from plainml.explaining import (
    describe_effect,
    feature_effects,
    permutation_table,
    summarize_importance,
    typical_values,
)
from plainml.io import describe_source, fingerprint, load_data, source_stem
from plainml.metrics import Metric, SafeScorer, metrics_for, resolve_metric
from plainml.profiling import (
    Profile,
    build_profile,
    is_imbalanced,
    redact_columns,
    render_profile,
    shown_source,
)
from plainml.registry import (
    ENSEMBLE_KEY,
    REGISTRY,
    STACKING_KEY,
    BuildOptions,
    ModelSpec,
    build_ensemble,
    build_stacking,
    resolve_keys,
    select_models,
)
from plainml.runs import (
    CONFIG_FILE,
    EVALUATION_FILE,
    HOLDOUT_FILE,
    IMPORTANCE_FILE,
    LEADERBOARD_FILE,
    MODEL_FILE,
    REPORT_FILE,
    RUN_FILE,
    create_run_dir,
    environment,
    write_json,
)
from plainml.schema import Schema, infer_schema
from plainml.tasks import (
    CLASSIFICATION,
    MULTI_REGRESSION,
    MULTILABEL,
    REGRESSION,
    TASK_LABELS,
    TargetInfo,
    normalize_task,
    prepare_target,
    to_python,
)

ProgressCallback = Callable[[float, str], None]

BALANCE_MODES = ("auto", "none", "weights", "smote")
THRESHOLD_MODES = ("auto", "on", "off")
DISPLAY_METRICS = {
    CLASSIFICATION: ["f1", "accuracy", "roc_auc"],
    REGRESSION: ["rmse", "r2", "mae"],
    MULTILABEL: ["f1", "accuracy", "hamming"],
    MULTI_REGRESSION: ["r2", "rmse", "mae"],
}
PARALLEL_CV_THRESHOLD = 50_000  # rows × columns above which CV folds run in parallel
ENSEMBLE_SIZE = 3


@dataclass
class TrainOptions:
    task: str | None = None
    metric: str | None = None
    models: list[str] | None = None
    exclude: list[str] | None = None
    quick: bool = False
    thorough: bool = False
    cv: int = 5
    test_size: float = 0.2
    seed: int = 42
    time_budget: float | None = None
    balance: str = "auto"
    threshold: str = "auto"
    log_target: bool = False
    calibrate: bool = False
    ensemble: bool = True
    refit: bool = True
    save_all: bool = False
    zip: bool = False
    drop: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)
    sample: float | None = None
    n_jobs: int = -1
    out_dir: str = "runs"
    name: str | None = None
    report: bool = True
    sheet: str | None = None
    query: str | None = None
    table: str | None = None
    engine: str = "auto"
    private: bool = False
    mlflow: bool = False

    @classmethod
    def build(cls, **values: Any) -> TrainOptions:
        known = {f.name for f in fields(cls)}
        unknown = [k for k in values if k not in known]
        if unknown:
            key = unknown[0]
            raise PlainMLError(f"Unknown training option '{key}'.{did_you_mean(key, known)}")
        options = cls(**{k: v for k, v in values.items() if v is not None})
        for name in ("models", "exclude"):
            value = getattr(options, name)
            if isinstance(value, str):
                setattr(options, name, [value])
        for name in ("drop", "keep"):
            value = getattr(options, name)
            if isinstance(value, str):
                value = [value]
            setattr(
                options,
                name,
                [c.strip() for v in value or [] for c in str(v).split(",") if c.strip()],
            )
        options.validate()
        return options

    def validate(self) -> None:
        if not isinstance(self.cv, int) or self.cv < 2:
            raise PlainMLError("--cv must be a whole number of at least 2.")
        if not 0.05 <= float(self.test_size) <= 0.5:
            raise PlainMLError(
                "--test-size must be between 0.05 and 0.5 (the share of rows held out)."
            )
        if self.balance not in BALANCE_MODES:
            raise PlainMLError(
                f"--balance must be one of {', '.join(BALANCE_MODES)}.{did_you_mean(self.balance, BALANCE_MODES)}"
            )
        if self.threshold not in THRESHOLD_MODES:
            raise PlainMLError(f"--threshold must be one of {', '.join(THRESHOLD_MODES)}.")
        if self.n_jobs == 0:
            raise PlainMLError("--n-jobs can't be 0 (use -1 for all cores).")
        resolve_keys(self.models)
        resolve_keys(self.exclude)


@dataclass
class Candidate:
    key: str
    name: str
    template: Any = None
    scores: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    train_score: float | None = None
    seconds: float = 0.0
    status: str = "ok"  # ok | failed | skipped
    note: str = ""
    baseline: bool = False
    members: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(
        default_factory=dict
    )  # tuned hyperparameters (pipeline param names)

    @property
    def spec_key(self) -> str:
        """Registry key, without suffixes like '@default' used to keep leaderboard keys unique."""
        return self.key.split("@")[0]


@dataclass
class Context:
    options: TrainOptions
    data_source: Any
    source: str
    stem: str
    raw_rows: int
    X: pd.DataFrame
    y: Any
    target: TargetInfo
    schema: Schema
    profile: Profile
    metrics: dict[str, Metric]
    metric: Metric
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Any
    y_test: Any
    cv: Any
    folds: int
    build: BuildOptions
    cv_jobs: int
    balance: str
    smote_k: int
    use_threshold: bool
    data_fingerprint: str
    started: float
    notes: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return self.target.task


@dataclass
class TrainResult:
    """What ``plainml.train`` returns: the saved run plus the fitted model."""

    run_dir: Path
    task: str
    target: str | list[str]
    metric: str
    best_model: str
    best_key: str
    cv_score: float
    holdout_scores: dict[str, float]
    leaderboard: pd.DataFrame
    importance: pd.DataFrame
    model: Any
    profile: Profile
    evaluation: dict[str, Any]

    @property
    def model_path(self) -> Path:
        return self.run_dir / MODEL_FILE

    @property
    def report_path(self) -> Path:
        return self.run_dir / REPORT_FILE

    def predict(self, data: Any, **options: Any) -> pd.DataFrame:
        from plainml.predicting import predict

        return predict(self.model, data, **options)

    def __repr__(self) -> str:
        return (
            f"TrainResult(best={self.best_model!r}, {self.metric}={fmt_num(self.cv_score)}, "
            f"run_dir='{self.run_dir}')"
        )

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        header = (
            f"<p><b>Best model:</b> {self.best_model} &middot; <b>{self.metric}</b> "
            f"{fmt_num(self.cv_score)} &middot; saved to <code>{self.run_dir}</code></p>"
        )
        return header + self.leaderboard.to_html(index=False, float_format=lambda v: fmt_num(v))


@contextmanager
def _quiet_warnings() -> Iterator[None]:
    """Silence library noise (convergence chatter etc.) during model fitting only."""
    with warnings.catch_warnings():
        for category in (
            ConvergenceWarning,
            UndefinedMetricWarning,
            FutureWarning,
            DeprecationWarning,
            UserWarning,
            RuntimeWarning,
        ):
            warnings.simplefilter("ignore", category)
        yield


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return (text[:140] + "…") if len(text) > 140 else text


def parse_targets(target: Any, columns: Any) -> list[str]:
    if target is None or (isinstance(target, str) and not target.strip()):
        shown = ", ".join(list(map(str, columns))[:15])
        raise PlainMLError(
            "Which column should the model predict?", hint=f"Add --target COLUMN. Columns: {shown}"
        )
    raw = [target] if isinstance(target, str) else list(target)
    names = [part.strip() for item in raw for part in str(item).split(",") if part.strip()]
    resolved = []
    for name in names:
        column = find_column(name, columns, "Target column")
        if column not in resolved:
            resolved.append(column)
    return resolved


def prepare(data: Any, target: Any, options: TrainOptions) -> Context:
    """Load, validate and split the data; decide CV, balancing and parallelism."""
    started = time.time()
    df = load_data(
        data,
        sheet=options.sheet,
        query=options.query,
        table=options.table,
        sample=options.sample,
        engine=options.engine,
        seed=options.seed,
    )
    targets = parse_targets(target, df.columns)
    drop = [find_column(c, df.columns) for c in options.drop]
    keep = [find_column(c, df.columns) for c in options.keep]
    drop = [c for c in drop if c not in targets]
    task = normalize_task(options.task)
    kept, y, tinfo = prepare_target(df, targets, task)
    X = kept.drop(columns=targets)
    schema, infos = infer_schema(X, drop=drop, keep=keep)
    source = describe_source(data)
    prof = build_profile(df, schema, infos, source, tinfo, y, X)
    if not schema.features:
        reasons = (
            "; ".join(f"{c}: {why}" for c, why in schema.dropped.items())
            or "the target is the only column"
        )
        raise PlainMLError(
            "No usable input columns are left to learn from.",
            hint=f"Not used: {reasons}. Force one in with --keep.",
        )
    if len(X) < 10:
        raise PlainMLError(
            f"Only {len(X)} usable rows; plainml needs at least 10 to train and test a model."
        )

    metrics = metrics_for(tinfo.task, y, n_classes=len(tinfo.classes) if tinfo.classes else None)
    metric = resolve_metric(options.metric, tinfo.task, metrics)
    notes: list[str] = []

    if options.log_target:
        if tinfo.task != REGRESSION:
            raise PlainMLError("--log-target only applies to regression.")
        if float(np.min(y)) < 0:
            raise PlainMLError("--log-target needs a target with no negative values.")

    balance = "none"
    if tinfo.task == CLASSIFICATION:
        if options.balance == "auto":
            balance = "weights" if is_imbalanced(y) else "none"
        else:
            balance = options.balance
        if balance == "smote":
            require("imblearn", "SMOTE balancing")
    elif options.balance in ("weights", "smote"):
        notes.append("--balance only applies to classification; ignored.")

    stratify = y if tinfo.task == CLASSIFICATION else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=options.test_size, random_state=options.seed, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=options.test_size, random_state=options.seed
        )
        notes.append("The test split couldn't keep class proportions (too few rows per class).")

    folds = options.cv
    smote_k = 5
    if tinfo.task == CLASSIFICATION:
        smallest = int(pd.Series(y_train).value_counts().min())
        if smallest < folds:
            folds = max(2, smallest)
            notes.append(
                f"Using {folds}-fold cross-validation: the rarest class has only {smallest} training rows."
            )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=options.seed)
        per_fold = int(smallest * (folds - 1) / folds)
        smote_k = min(5, per_fold - 1)
        if balance == "smote" and smote_k < 1:
            balance = "weights"
            notes.append("Too few rare-class rows for SMOTE; using class weights instead.")
    else:
        folds = min(folds, len(X_train))
        cv = KFold(n_splits=folds, shuffle=True, random_state=options.seed)

    cpus = os.cpu_count() or 1
    size = len(X_train) * max(1, len(schema.features))
    parallel = options.n_jobs != 1 and cpus > 1 and size >= PARALLEL_CV_THRESHOLD
    cv_jobs = min(folds, options.n_jobs if options.n_jobs > 0 else cpus) if parallel else 1
    pos_weight = None
    if tinfo.task == CLASSIFICATION and tinfo.classes and len(tinfo.classes) == 2:
        counts = pd.Series(y_train).value_counts()
        pos_weight = float(
            counts.get(tinfo.classes[0], 1) / max(counts.get(tinfo.classes[1], 1), 1)
        )
    build = BuildOptions(
        seed=options.seed,
        n_jobs=1 if parallel else options.n_jobs,
        class_weight=balance == "weights",
        pos_weight=pos_weight,
        n_rows=len(X_train),
    )
    use_threshold = (
        tinfo.task == CLASSIFICATION
        and tinfo.classes is not None
        and len(tinfo.classes) == 2
        and (
            options.threshold == "on"
            or (
                options.threshold == "auto"
                and balance != "none"
                and metric.key in ("f1", "balanced_accuracy", "recall", "precision")
            )
        )
    )
    if options.threshold == "on" and not use_threshold:
        notes.append("--threshold on only applies to yes/no (binary) classification; ignored.")

    return Context(
        options=options,
        data_source=data,
        source=source,
        stem=source_stem(data),
        raw_rows=len(df),
        X=X,
        y=y,
        target=tinfo,
        schema=schema,
        profile=prof,
        metrics=metrics,
        metric=metric,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=cv,
        folds=folds,
        build=build,
        cv_jobs=cv_jobs,
        balance=balance,
        smote_k=smote_k,
        use_threshold=use_threshold,
        data_fingerprint=fingerprint(kept),
        started=started,
        notes=notes,
    )


def build_estimator(ctx: Context, spec: ModelSpec, build: BuildOptions | None = None) -> Any:
    """Preprocessing + (SMOTE) + model, wrapped for a log-transformed target if asked."""
    from plainml.preprocessing import make_model_pipeline

    estimator = spec.build(ctx.task, build or ctx.build)
    sampler = None
    if ctx.balance == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(k_neighbors=ctx.smote_k, random_state=ctx.options.seed)
    pipeline = make_model_pipeline(ctx.schema, estimator, sampler)
    if ctx.options.log_target:
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        )
    return pipeline


def cross_validate_candidate(ctx: Context, candidate: Candidate) -> None:
    scoring = {key: SafeScorer(metric.scorer()) for key, metric in ctx.metrics.items()}
    start = time.perf_counter()
    with _quiet_warnings():
        result = cross_validate(
            candidate.template,
            ctx.X_train,
            ctx.y_train,
            cv=ctx.cv,
            scoring=scoring,
            n_jobs=ctx.cv_jobs,
            return_train_score=True,
            error_score="raise",
        )
        candidate.seconds = time.perf_counter() - start
        for key, metric in ctx.metrics.items():
            values = np.asarray(result[f"test_{key}"], dtype=float)
            finite = values[~np.isnan(values)]
            candidate.scores[key] = (
                metric.display(float(finite.mean())) if finite.size else float("nan")
            )
            candidate.stds[key] = float(finite.std()) if finite.size else float("nan")
        train = np.asarray(result[f"train_{ctx.metric.key}"], dtype=float)
        train = train[~np.isnan(train)]
        candidate.train_score = ctx.metric.display(float(train.mean())) if train.size else None


def _overfit_note(candidate: Candidate, metric: Metric) -> str:
    """Flag only big train/validation gaps (tree ensembles always look perfect on training data)."""
    cv_score = candidate.scores.get(metric.key)
    train = candidate.train_score
    if train is None or cv_score is None or np.isnan(cv_score) or candidate.baseline:
        return ""
    if metric.greater_is_better and metric.bounded and train - cv_score > 0.25:
        return f"big train/test gap ({fmt_num(train)} vs {fmt_num(cv_score)})"
    if not metric.greater_is_better and train > 0 and cv_score / train > 4:
        return f"big train/test gap ({fmt_num(train)} vs {fmt_num(cv_score)})"
    return ""


def signal_warning(best: Candidate, baseline: Candidate | None, ctx: Context) -> str | None:
    """Say so when the data barely predicts the target, judged by a metric where 'random' is clear."""
    auc = best.scores.get("roc_auc")
    r2 = best.scores.get("r2")
    if ctx.task == CLASSIFICATION and auc is not None and not np.isnan(auc) and auc < 0.6:
        return (
            f"The best ROC-AUC is {fmt_num(auc)}, close to 0.5 (random guessing): "
            "the columns carry little information about the target."
        )
    if ctx.task == REGRESSION and r2 is not None and not np.isnan(r2) and r2 < 0.05:
        return (
            f"The best R² is {fmt_num(r2)}: the model explains almost none of the variation in "
            f"{ctx.target.display}, so the columns carry little information about it."
        )
    metric = ctx.metric
    if baseline is None or np.isnan(baseline.scores.get(metric.key, float("nan"))):
        return None
    gain = best.scores[metric.key] - baseline.scores[metric.key]
    gain = gain if metric.greater_is_better else -gain
    scale = 1.0 if metric.bounded else max(abs(baseline.scores[metric.key]), 1e-9)
    if gain / scale < 0.02:
        return "No model clearly beats the do-nothing baseline: the columns may not carry information about the target."
    return None


def _score_line(candidate: Candidate, metric: Metric) -> str:
    if candidate.status != "ok":
        style = "error" if candidate.status == "failed" else "muted"
        return f"[{style}]{'✗' if candidate.status == 'failed' else '–'} {candidate.name:<34}[/] [muted]{esc(candidate.note)}[/]"
    score = candidate.scores.get(metric.key, float("nan"))
    spread = candidate.stds.get(metric.key, float("nan"))
    marker = "[muted]·[/]" if candidate.baseline else "[good]✓[/]"
    return (
        f"{marker} {candidate.name:<34} {metric.label} [bold]{fmt_num(score)}[/] "
        f"[muted]± {fmt_spread(spread)}   {fmt_duration(candidate.seconds)}[/]"
    )


def run_candidates(
    ctx: Context,
    specs: list[ModelSpec],
    progress_callback: ProgressCallback | None = None,
) -> list[Candidate]:
    """Cross-validate each model (and then an ensemble of the best)."""
    options = ctx.options
    candidates: list[Candidate] = []
    budget = options.time_budget
    use_ensemble = (
        options.ensemble and not options.quick and ctx.task in (CLASSIFICATION, REGRESSION)
    )
    use_stacking = use_ensemble and options.thorough
    total = len(specs) + (1 if use_ensemble else 0) + (1 if use_stacking else 0)
    heading(
        f"Training {len(specs)} models ({ctx.folds}-fold cross-validation on {len(ctx.X_train):,} rows)"
    )
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("[muted]{task.completed}/{task.total}[/]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    stopped = False
    with progress:
        bar = progress.add_task("Starting", total=total)
        for i, spec in enumerate(specs):
            name = spec.name(ctx.task)
            candidate = Candidate(spec.key, name, baseline=spec.baseline)
            elapsed = time.time() - ctx.started
            trained_one = any(c.status == "ok" and not c.baseline for c in candidates)
            if stopped or (budget and elapsed > budget and trained_one and not spec.baseline):
                candidate.status = "skipped"
                candidate.note = "stopped by you" if stopped else "time budget reached"
                candidates.append(candidate)
                progress.advance(bar)
                continue
            if budget and trained_one and not spec.baseline:
                progress.update(bar, description=f"Sizing up {name}")
                expected = estimate_seconds(ctx, spec)
                if expected is not None and expected > budget - elapsed:
                    candidate.status = "skipped"
                    candidate.note = f"would take ~{fmt_duration(expected)}, over the time budget"
                    progress.console.print(_score_line(candidate, ctx.metric))
                    candidates.append(candidate)
                    progress.advance(bar)
                    continue
            progress.update(bar, description=f"Training {name}")
            if progress_callback:
                progress_callback(i / total, f"Training {name}")
            try:
                candidate.template = build_estimator(ctx, spec)
                cross_validate_candidate(ctx, candidate)
            except KeyboardInterrupt:
                stopped = True
                candidate.status, candidate.note = "skipped", "stopped by you"
                warn(
                    "Stopping early: finishing with the models trained so far (Ctrl-C again to quit)."
                )
            except Exception as exc:  # one model failing shouldn't sink the run
                candidate.status, candidate.note = "failed", _short_error(exc)
            candidate.note = candidate.note or _overfit_note(candidate, ctx.metric)
            progress.console.print(_score_line(candidate, ctx.metric))
            candidates.append(candidate)
            progress.advance(bar)

        if use_ensemble and not stopped:
            ranked = [
                c for c in rank(candidates, ctx.metric) if c.status == "ok" and not c.baseline
            ]
            if len(ranked) >= ENSEMBLE_SIZE and not (budget and time.time() - ctx.started > budget):
                members = ranked[:ENSEMBLE_SIZE]
                label = ", ".join(m.name for m in members)
                ensemble = Candidate(
                    ENSEMBLE_KEY,
                    f"Ensemble (average of top {ENSEMBLE_SIZE})",
                    members=[m.key for m in members],
                )
                progress.update(bar, description="Training ensemble")
                if progress_callback:
                    progress_callback(len(specs) / total, "Training ensemble")
                try:
                    ensemble.template = build_ensemble(
                        ctx.task, [(m.key, clone(m.template)) for m in members]
                    )
                    cross_validate_candidate(ctx, ensemble)
                    ensemble.note = f"{label}"
                except KeyboardInterrupt:
                    ensemble.status, ensemble.note = "skipped", "stopped by you"
                except Exception as exc:
                    ensemble.status, ensemble.note = "failed", _short_error(exc)
                progress.console.print(_score_line(ensemble, ctx.metric))
                candidates.append(ensemble)
                if use_stacking and ensemble.status != "skipped":
                    stack = Candidate(
                        STACKING_KEY,
                        f"Stacked ensemble (top {ENSEMBLE_SIZE} + a combiner)",
                        members=[m.key for m in members],
                    )
                    progress.update(bar, description="Training stacked ensemble")
                    try:
                        stack.template = build_stacking(
                            ctx.task, [(m.key, clone(m.template)) for m in members]
                        )
                        cross_validate_candidate(ctx, stack)
                        stack.note = f"{label}"
                    except KeyboardInterrupt:
                        stack.status, stack.note = "skipped", "stopped by you"
                    except Exception as exc:
                        stack.status, stack.note = "failed", _short_error(exc)
                    progress.console.print(_score_line(stack, ctx.metric))
                    candidates.append(stack)
            progress.advance(bar)
    if progress_callback:
        progress_callback(1.0, "Evaluating the best model")
    return candidates


ESTIMATE_MIN_ROWS = 4000


def estimate_seconds(ctx: Context, spec: ModelSpec) -> float | None:
    """Roughly how long cross-validating ``spec`` will take, from a quick fit on a sample.

    Only used with --time-budget, so a slow model isn't started when it can't finish in time.
    """
    n_rows = len(ctx.X_train)
    if spec.baseline or n_rows < ESTIMATE_MIN_ROWS:
        return None
    sample = max(500, min(2000, n_rows // 8))
    index = ctx.X_train.sample(sample, random_state=ctx.options.seed).index
    X, y = ctx.X_train.loc[index], ctx.y_train.loc[index]
    try:
        estimator = build_estimator(ctx, spec)
        start = time.perf_counter()
        with _quiet_warnings():
            estimator.fit(X, y)
            estimator.predict(X)
        seconds = time.perf_counter() - start
    except Exception:
        return None
    fold_rows = n_rows * (ctx.folds - 1) / ctx.folds
    per_fold = seconds * (fold_rows / sample) ** spec.cost
    # parallel folds share CPU and memory bandwidth, so they speed things up far less than n×
    return per_fold * ctx.folds / max(1, ctx.cv_jobs) ** 0.5 * 1.2


def rank(candidates: list[Candidate], metric: Metric) -> list[Candidate]:
    def usable(c: Candidate) -> bool:
        return c.status == "ok" and not np.isnan(c.scores.get(metric.key, float("nan")))

    sign = -1.0 if metric.greater_is_better else 1.0
    # ties go to the steadier model, then the faster one
    ok = sorted(
        (c for c in candidates if usable(c)),
        key=lambda c: (sign * c.scores[metric.key], c.stds.get(metric.key, 0.0), c.seconds),
    )
    return ok + [c for c in candidates if not usable(c)]


def pick_best(ranked: list[Candidate], metric: Metric) -> Candidate:
    for candidate in ranked:
        if (
            candidate.status == "ok"
            and not candidate.baseline
            and not np.isnan(candidate.scores.get(metric.key, float("nan")))
        ):
            return candidate
    failures = [c for c in ranked if c.status == "failed"]
    detail = f" First error ({failures[0].name}): {failures[0].note}" if failures else ""
    raise PlainMLError(f"No model could be trained.{detail}", hint="Run with --debug for details.")


def leaderboard_frame(ranked: list[Candidate], ctx: Context) -> pd.DataFrame:
    rows = []
    position = 0
    for candidate in ranked:
        if candidate.status == "ok":
            position += 1
        row: dict[str, Any] = {
            "rank": position if candidate.status == "ok" else None,
            "model": candidate.name,
            "key": candidate.key,
        }
        for key in ctx.metrics:
            row[key] = candidate.scores.get(key)
        row[f"{ctx.metric.key}_std"] = candidate.stds.get(ctx.metric.key)
        row[f"train_{ctx.metric.key}"] = candidate.train_score
        row["seconds"] = round(candidate.seconds, 2)
        row["status"] = candidate.status
        row["note"] = candidate.note
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_spread(value: float | None) -> str:
    """A standard deviation to two significant digits: it has no more precision than that."""
    if value is None or not np.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2g}"


def render_leaderboard(frame: pd.DataFrame, ctx: Context, best_key: str) -> None:
    heading("Leaderboard")
    shown = [ctx.metric.key] + [
        k for k in DISPLAY_METRICS[ctx.task] if k != ctx.metric.key and k in ctx.metrics
    ]
    labels = [
        ctx.metrics[k].label + (" ↑" if ctx.metrics[k].greater_is_better else " ↓") for k in shown
    ]
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        is_best = row["key"] == best_key
        cells = [fmt_num(row[key]) for key in shown]
        if row["status"] == "ok":
            cells[0] += f" ±{fmt_spread(row[f'{ctx.metric.key}_std'])}"
        rows.append(
            {
                "rank": "" if pd.isna(row["rank"]) else str(int(row["rank"])),
                "model": f"★ {row['model']}" if is_best else str(row["model"]),
                "cells": cells,
                "time": fmt_duration(row["seconds"]) if row["status"] == "ok" else "",
                "note": str(row["note"] or ""),
                "style": "best"
                if is_best
                else ("muted" if row["key"] == "baseline" or row["status"] != "ok" else ""),
            }
        )
    # Lay the table out by hand: scores and times always get their full width, model
    # names shrink if they must, and notes use whatever is left (or are hidden).
    gap = 2
    score_widths = [
        max(len(label), *(len(r["cells"][i]) for r in rows)) for i, label in enumerate(labels)
    ]
    fixed = 2 + sum(score_widths) + max(4, *(len(r["time"]) for r in rows))
    model_width = max(len("Model"), *(len(r["model"]) for r in rows))
    columns_count = 3 + len(labels)
    available = console.width - fixed - gap * columns_count
    model_width = max(14, min(model_width, 36, available))
    notes_width = available - model_width - gap
    show_notes = notes_width >= 12 and any(r["note"] for r in rows)

    table = Table(
        header_style="muted", box=None, pad_edge=False, show_edge=False, collapse_padding=True
    )
    table.add_column("#", justify="right", style="muted", no_wrap=True, width=2)
    table.add_column("Model", no_wrap=True, overflow="ellipsis", width=model_width)
    for i, (label, width) in enumerate(zip(labels, score_widths, strict=True)):
        table.add_column(
            label, justify="right", style="bold" if i == 0 else "", no_wrap=True, width=width
        )
    table.add_column("Time", justify="right", style="muted", no_wrap=True)
    if show_notes:
        table.add_column(
            "Notes", style="muted", overflow="ellipsis", no_wrap=True, width=min(notes_width, 48)
        )
    for r in rows:
        cells = [r["rank"], esc(r["model"]), *r["cells"], r["time"]]
        if show_notes:
            cells.append(esc(r["note"]))
        table.add_row(*cells, style=r["style"])
    console.print(table)
    direction = "higher" if ctx.metric.greater_is_better else "lower"
    note(
        f"Ranked by {ctx.metric.label} ({direction} is better): {esc(ctx.metric.describe(ctx.target.display))}"
    )
    if not show_notes and any(r["note"] for r in rows):
        note("Widen the terminal to see notes, or open leaderboard.csv / report.html.")


def command_line() -> str:
    program = Path(sys.argv[0]).name if sys.argv else ""
    if program in ("plainml", "plainml.exe", "__main__.py"):
        return "plainml " + shlex.join(sys.argv[1:])
    return "Python API"


def _example_row(X: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    if X.empty:
        return {}
    row = X[features].iloc[0]
    return {
        k: (
            None
            if pd.isna(v)
            else to_python(v)
            if not isinstance(v, pd.Timestamp)
            else v.isoformat()
        )
        for k, v in row.items()
    }


def _final_template(ctx: Context, best: Candidate) -> Any:
    build = replace(ctx.build, n_jobs=ctx.options.n_jobs)
    if best.spec_key in REGISTRY:
        template = build_estimator(ctx, REGISTRY[best.spec_key], build)
        if best.params:
            template.set_params(**best.params)
    else:
        template = clone(best.template)
    if ctx.options.calibrate and ctx.task == CLASSIFICATION:
        # isotonic needs plenty of rows; Platt scaling (sigmoid) is safer on small data
        template = CalibratedClassifierCV(
            template,
            method="isotonic" if len(ctx.X_train) >= 1000 else "sigmoid",
            cv=StratifiedKFold(
                n_splits=min(5, ctx.folds), shuffle=True, random_state=ctx.options.seed
            ),
            ensemble=False,
        )
    if ctx.use_threshold:
        template = TunedThresholdClassifierCV(
            template,
            scoring=ctx.metric.scorer(),
            cv=StratifiedKFold(
                n_splits=min(5, ctx.folds), shuffle=True, random_state=ctx.options.seed
            ),
            random_state=ctx.options.seed,
        )
    return template


def _config_for(ctx: Context) -> dict[str, Any]:
    defaults = asdict(TrainOptions())
    config: dict[str, Any] = {
        "data": shown_source(ctx.source, ctx.options.private)
        if not isinstance(ctx.data_source, pd.DataFrame)
        else None,
        "target": ctx.target.names[0] if len(ctx.target.names) == 1 else ctx.target.names,
    }
    for key, value in asdict(ctx.options).items():
        if value != defaults.get(key) and key not in ("out_dir", "name"):
            config[key] = value
    config["task"] = ctx.task
    return {k: v for k, v in config.items() if v is not None}


def finalize(
    ctx: Context,
    candidates: list[Candidate],
    *,
    kind: str = "train",
    extra: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainResult:
    """Evaluate the winner on the test set, explain it, retrain on everything, save."""
    options = ctx.options
    metric = ctx.metric
    ranked = rank(candidates, metric)
    best = pick_best(ranked, metric)
    leaderboard = leaderboard_frame(ranked, ctx)
    render_leaderboard(leaderboard, ctx, best.key)

    with (
        console.status(f"Testing {best.name} on {len(ctx.X_test):,} held-out rows…"),
        _quiet_warnings(),
    ):
        template = _final_template(ctx, best)
        fitted = clone(template).fit(ctx.X_train, ctx.y_train)
        evaluation, rows = evaluate_model(
            fitted,
            ctx.X_test,
            ctx.y_test,
            ctx.task,
            ctx.metrics,
            label_maps=ctx.target.label_maps,
            seed=options.seed,
        )
    features = ctx.schema.features
    with console.status("Measuring which columns matter…"), _quiet_warnings():
        importance = permutation_table(
            fitted, ctx.X_test[features], ctx.y_test, metric, seed=options.seed
        )
        top = importance["feature"].head(6).tolist()
        effects = feature_effects(
            fitted,
            ctx.X_test[features],
            ctx.schema,
            top,
            ctx.task,
            ctx.target.classes,
            seed=options.seed,
        )
    sentences = summarize_importance(importance) + [
        describe_effect(e, ctx.target.display, ctx.task) for e in effects
    ]
    evaluation["effects"] = effects
    evaluation["sentences"] = sentences

    refit_rows = len(ctx.X)
    if options.refit:
        if progress_callback:
            progress_callback(1.0, f"Retraining {best.name} on all rows")
        with (
            console.status(f"Retraining {best.name} on all {len(ctx.X):,} rows…"),
            _quiet_warnings(),
        ):
            final = clone(template).fit(ctx.X, ctx.y)
    else:
        final, refit_rows = fitted, len(ctx.X_train)

    run_dir = create_run_dir(options.out_dir, options.name or ctx.stem)
    created = datetime.now().isoformat(timespec="seconds")
    meta = {
        "plainml_version": __version__,
        "packages": environment()["packages"],
        "created": created,
        "run": run_dir.name,
        "task": ctx.task,
        "targets": ctx.target.names,
        "features": features,
        "schema": ctx.schema.to_dict(),
        "classes": ctx.target.classes,
        "label_maps": ctx.target.label_maps,
        "metric": metric.key,
        "model": best.name,
        "model_key": best.key,
        "cv_score": best.scores.get(metric.key),
        "holdout_scores": evaluation["scores"],
        "threshold": evaluation.get("threshold"),
        "log_target": options.log_target,
        "typical": typical_values(ctx.X, ctx.schema),
        "drift_profile": describe_distribution(ctx.X, ctx.schema),
        "importance": {str(r.feature): float(r.share) for r in importance.itertuples()},
        "example": {} if options.private else _example_row(ctx.X_test, features),
        "private": options.private,
    }
    final.plainml_meta_ = meta
    joblib.dump(final, run_dir / MODEL_FILE, compress=3)

    if options.save_all:
        models_dir = run_dir / "models"
        models_dir.mkdir()
        with console.status("Saving every model…"), _quiet_warnings():
            for candidate in ranked:
                if candidate.status != "ok" or candidate.baseline or candidate.template is None:
                    continue
                model = clone(candidate.template).fit(ctx.X, ctx.y)
                model.plainml_meta_ = {
                    **meta,
                    "model": candidate.name,
                    "model_key": candidate.key,
                    "cv_score": candidate.scores.get(metric.key),
                }
                joblib.dump(model, models_dir / f"{candidate.key}.joblib", compress=3)

    if options.private:  # keep the scores, not copies of the input rows
        holdout = rows.reset_index(drop=True).rename_axis("row").reset_index()
    else:
        holdout = pd.concat([ctx.X_test, rows], axis=1)
    holdout.to_csv(run_dir / HOLDOUT_FILE, index=False)
    leaderboard.to_csv(run_dir / LEADERBOARD_FILE, index=False)
    importance.to_csv(run_dir / IMPORTANCE_FILE, index=False)
    write_json(run_dir / EVALUATION_FILE, evaluation)

    baseline = next((c for c in ranked if c.baseline and c.status == "ok"), None)
    info_json = {
        "kind": kind,
        "plainml_version": __version__,
        "created": created,
        "command": "(hidden: private run)" if options.private else command_line(),
        "task": ctx.task,
        "task_label": TASK_LABELS[ctx.task],
        "task_reason": ctx.target.reason,
        "target": ctx.target.name,
        "metric": metric.key,
        "metric_label": metric.label,
        "metric_explain": metric.describe(ctx.target.display),
        "metric_greater_is_better": metric.greater_is_better,
        "data": {
            "source": shown_source(ctx.source, options.private),
            "rows": len(ctx.X),
            "raw_rows": ctx.raw_rows,
            "columns": len(ctx.schema.order) + len(ctx.target.names),
            "fingerprint": ctx.data_fingerprint,
        },
        "options": asdict(options),
        "schema": ctx.schema.to_dict(),
        "profile": {
            "issues": [asdict(i) for i in ctx.profile.issues],
            "target": ctx.profile.target,
            "columns": redact_columns([c.to_dict() for c in ctx.profile.columns])
            if options.private
            else [c.to_dict() for c in ctx.profile.columns],
            "n_duplicates": ctx.profile.n_duplicates,
        },
        "classes": ctx.target.classes,
        "label_maps": ctx.target.label_maps,
        "split": {
            "train_rows": len(ctx.X_train),
            "test_rows": len(ctx.X_test),
            "cv_folds": ctx.folds,
            "stratified": ctx.task == CLASSIFICATION,
        },
        "balance": ctx.balance,
        "threshold": evaluation.get("threshold"),
        "best": {
            "key": best.key,
            "name": best.name,
            "cv_score": best.scores.get(metric.key),
            "cv_std": best.stds.get(metric.key),
            "train_score": best.train_score,
            "holdout": evaluation["scores"],
            "refit_rows": refit_rows,
            "members": best.members,
            "params": {k.split("__")[-1]: v for k, v in best.params.items()},
        },
        "baseline": {"cv_score": baseline.scores.get(metric.key)} if baseline else None,
        "skipped": ctx.skipped,
        "notes": ctx.notes,
        "timings": {"total_seconds": round(time.time() - ctx.started, 2)},
        "environment": environment(),
        "files": {
            "model": MODEL_FILE,
            "report": REPORT_FILE if options.report else None,
            "leaderboard": LEADERBOARD_FILE,
            "evaluation": EVALUATION_FILE,
            "importance": IMPORTANCE_FILE,
            "holdout_predictions": HOLDOUT_FILE,
            "config": CONFIG_FILE,
            "model_card": CARD_FILE,
        },
        **(extra or {}),
    }
    write_json(run_dir / RUN_FILE, info_json)
    try:
        write_model_card(run_dir)
    except Exception as exc:
        warn(f"Couldn't write the model card: {esc(_short_error(exc))}")
    (run_dir / CONFIG_FILE).write_text(
        "# Rerun with: plainml train --config "
        + str(run_dir / CONFIG_FILE)
        + "\n"
        + yaml.safe_dump(_config_for(ctx), sort_keys=False),
        encoding="utf-8",
    )
    if options.report:
        try:
            from plainml.report import write_report

            write_report(run_dir)
        except Exception as exc:  # the model is saved; a report bug shouldn't lose it
            warn(f"Couldn't write the HTML report: {esc(_short_error(exc))}")
    if options.mlflow:
        try:
            from plainml.tracking import log_run

            log_run(run_dir)
        except PlainMLError:
            raise
        except Exception as exc:  # the run is saved; a tracking-server hiccup shouldn't lose it
            warn(f"Couldn't log to MLflow: {esc(_short_error(exc))}")
    archive = None
    if options.zip:
        archive = Path(shutil.make_archive(str(run_dir), "zip", root_dir=run_dir))

    result = TrainResult(
        run_dir=run_dir,
        task=ctx.task,
        target=ctx.target.name,
        metric=metric.key,
        best_model=best.name,
        best_key=best.key,
        cv_score=best.scores.get(metric.key, float("nan")),
        holdout_scores=evaluation["scores"],
        leaderboard=leaderboard,
        importance=importance,
        model=final,
        profile=ctx.profile,
        evaluation=evaluation,
    )
    render_summary(result, ctx, best, baseline, archive)
    return result


def render_summary(
    result: TrainResult,
    ctx: Context,
    best: Candidate,
    baseline: Candidate | None,
    archive: Path | None,
) -> None:
    metric = ctx.metric
    heading("Result")
    console.print(f"[best]★ Best model: {best.name}[/]")
    spread = best.stds.get(metric.key)
    line = f"  {metric.label} [bold]{fmt_num(best.scores.get(metric.key))}[/] ± {fmt_spread(spread)} in {ctx.folds}-fold cross-validation"
    if baseline:
        line += f" [muted](baseline {fmt_num(baseline.scores.get(metric.key))})[/]"
    console.print(line)
    shown = [metric.key] + [k for k in DISPLAY_METRICS[ctx.task] if k != metric.key]
    held = " · ".join(
        f"{ctx.metrics[k].label} {fmt_num(result.holdout_scores.get(k))}"
        for k in shown
        if k in result.holdout_scores
    )
    console.print(f"  On {len(ctx.X_test):,} held-out test rows: {held}")
    console.print(
        f"  [muted]{metric.label}: {esc(metric.describe(ctx.target.display))}[/]", soft_wrap=True
    )
    threshold = result.evaluation.get("threshold")
    if threshold is not None:
        console.print(
            f"  [muted]Decision threshold tuned to {threshold:.2f} (instead of 0.5) to maximise {metric.label}.[/]"
        )
    calibration = result.evaluation.get("calibration")
    if calibration:
        verdict = calibration_verdict(calibration["ece"])
        line = f"  [muted]Probabilities are {verdict} (off by {fmt_pct(calibration['ece'], 1)} on average)"
        if calibration["ece"] > 0.08 and not ctx.options.calibrate:
            line += "; retrain with --calibrate if you rely on them"
        console.print(line + ".[/]")

    problem = signal_warning(best, baseline, ctx)
    if problem:
        warn(esc(problem))
    if best.note.startswith("big train/test gap"):
        warn(
            f"{best.name} does much better on its training data than on new data ({best.note[len('big train/test gap ') :]}); it may be memorising rather than learning."
        )
    for message in ctx.notes:
        info(esc(message))
    if result.evaluation.get("sentences"):
        heading("What drives the predictions")
        for sentence in result.evaluation["sentences"][:5]:
            console.print(f"• {esc(sentence)}")

    heading("Saved")
    run_dir = result.run_dir
    console.print(f"[bold]{esc(run_dir)}[/]")
    console.print(
        f"  {MODEL_FILE:<24}[muted]the trained model, retrained on all {len(ctx.X):,} rows[/]"
    )
    if ctx.options.report:
        console.print(f"  {REPORT_FILE:<24}[muted]charts and explanations; open it in a browser[/]")
    console.print(f"  {LEADERBOARD_FILE:<24}[muted]every model's scores[/]")
    console.print(f"  {HOLDOUT_FILE:<24}[muted]test rows with actual vs predicted values[/]")
    if (run_dir / CARD_FILE).is_file():
        console.print(f"  {CARD_FILE:<24}[muted]a one-page summary to share with the model[/]")
    if archive:
        console.print(f"  [muted]Zipped to {esc(archive)}[/]")
    console.print()
    console.print("[muted]Next:[/]")
    console.print("  plainml predict latest NEW_DATA.csv -o predictions.csv")
    console.print("  plainml explain latest")
    if best.spec_key in REGISTRY and REGISTRY[best.spec_key].space(ctx.task) and not best.params:
        console.print("  plainml tune latest")


def train(
    data: Any = None,
    target: Any = None,
    *,
    config: str | Path | None = None,
    verbose: bool = True,
    progress: ProgressCallback | None = None,
    **options: Any,
) -> TrainResult:
    """Train and compare models, then save the best one with a report.

    ``data`` is a file path, URL, SQL URL or DataFrame; ``target`` the column(s) to predict.
    Every CLI option is available as a keyword, e.g. ``quick=True``, ``metric="roc_auc"``,
    ``models=["rf", "xgboost"]``, ``time_budget=300``. ``config`` loads a YAML file of options.
    """
    if config:
        from plainml.config import load_config

        loaded = load_config(config)
        data = data if data is not None else loaded.pop("data", None)
        target = target if target is not None else loaded.pop("target", None)
        loaded.pop("data", None)
        loaded.pop("target", None)
        options = {**loaded, **options}
    if data is None:
        raise PlainMLError("No data given.", hint="Pass a file path, URL or DataFrame.")
    if "time_budget" in options:
        from plainml.console import parse_duration

        options["time_budget"] = parse_duration(options["time_budget"])
    opts = TrainOptions.build(**options)

    with quiet(not verbose):
        ctx = prepare(data, target, opts)
        render_profile(ctx.profile, columns=False)
        if ctx.balance != "none":
            info(
                f"Balancing classes with {'class weights' if ctx.balance == 'weights' else 'SMOTE oversampling'}."
            )
        selection = select_models(
            ctx.task,
            include=opts.models,
            exclude=opts.exclude,
            quick=opts.quick,
            thorough=opts.thorough,
            n_rows=len(ctx.X_train),
            y=ctx.y_train if ctx.task in (REGRESSION, MULTI_REGRESSION) else None,
        )
        ctx.skipped = selection.skipped
        for key, reason in selection.skipped.items():
            note(f"Skipping {key}: {esc(reason)}")
        if not [s for s in selection.specs if not s.baseline]:
            raise PlainMLError(
                "No models left to train.",
                hint="Check --models / --exclude, or run: plainml models",
            )
        candidates = run_candidates(ctx, selection.specs, progress)
        return finalize(ctx, candidates, progress_callback=progress)
