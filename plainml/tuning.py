"""``plainml tune``: search hyperparameters for the most promising models.

Uses Optuna's TPE sampler when it's installed (smarter: learns which settings work) and
falls back to scikit-learn's random search otherwise. Every candidate setting is scored
with the same cross-validation folds as the original training run, and the untuned
defaults are re-scored on those folds too, so "before" and "after" are comparable.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from plainml.console import (
    LIVE_REFRESH,
    console,
    esc,
    fmt_num,
    heading,
    info,
    note,
    parse_duration,
    quiet,
    warn,
)
from plainml.errors import PlainMLError, is_installed
from plainml.metrics import SafeScorer
from plainml.registry import REGISTRY, ModelSpec, Space, estimator_param_prefix, resolve_keys
from plainml.runs import DEFAULT_RUNS_DIR, load_run, resolve_run
from plainml.training import (
    Candidate,
    Context,
    TrainOptions,
    TrainResult,
    _quiet_warnings,
    build_estimator,
    cross_validate_candidate,
    finalize,
    prepare,
)

DEFAULT_TUNABLE = [
    "lightgbm",
    "xgboost",
    "histgb",
    "rf",
    "extratrees",
    "catboost",
    "svm",
    "knn",
    "linear",
]


def param_prefix(template: Any) -> str:
    if isinstance(template, TransformedTargetRegressor):
        return "regressor__" + estimator_param_prefix(template.regressor.named_steps["model"])
    return estimator_param_prefix(template.named_steps["model"])


def _suggest(trial: Any, name: str, spec: tuple[Any, ...]) -> Any:
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2], log=bool(spec[3]))
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2], log=bool(spec[3]))
    return trial.suggest_categorical(name, list(spec[1]))


def _distribution(spec: tuple[Any, ...]) -> Any:
    from scipy import stats

    kind = spec[0]
    if kind == "int":
        low, high, log = spec[1], spec[2], spec[3]
        if log:
            return sorted({int(round(v)) for v in np.geomspace(low, high, 40)})
        return stats.randint(low, high + 1)
    if kind == "float":
        low, high, log = spec[1], spec[2], spec[3]
        return stats.loguniform(low, high) if log else stats.uniform(low, high - low)
    return list(spec[1])


def _search_optuna(
    ctx: Context,
    base: Any,
    space: Space,
    prefix: str,
    trials: int,
    timeout: float | None,
    on_trial: Any,
) -> tuple[dict[str, Any], float, int]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    scorer = SafeScorer(ctx.metric.scorer())
    errors: list[str] = []

    def objective(trial: Any) -> float:
        params = {prefix + name: _suggest(trial, name, spec) for name, spec in space.items()}
        try:
            estimator = clone(base).set_params(**params)
        except Exception as exc:
            errors.append(str(exc).splitlines()[0][:160])
            raise
        with _quiet_warnings():
            scores = cross_val_score(
                estimator, ctx.X_train, ctx.y_train, cv=ctx.cv, scoring=scorer, n_jobs=ctx.cv_jobs
            )
        value = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
        if np.isnan(value):
            raise optuna.TrialPruned()
        return value

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=ctx.options.seed)
    )
    study.optimize(
        objective,
        n_trials=trials,
        timeout=timeout,
        catch=(Exception,),
        callbacks=[lambda study, trial: on_trial(study)],
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        detail = f" First error: {errors[0]}" if errors else ""
        raise PlainMLError(f"Every tuning trial failed for this model.{detail}")
    best = study.best_trial
    return (
        {prefix + k: v for k, v in best.params.items()},
        float(best.value or 0.0),
        len(study.trials),
    )


def _search_random(
    ctx: Context, base: Any, space: Space, prefix: str, trials: int
) -> tuple[dict[str, Any], float, int]:
    search = RandomizedSearchCV(
        base,
        {prefix + name: _distribution(spec) for name, spec in space.items()},
        n_iter=trials,
        scoring=SafeScorer(ctx.metric.scorer()),
        cv=ctx.cv,
        n_jobs=ctx.cv_jobs,
        random_state=ctx.options.seed,
        error_score=np.nan,
        refit=False,
    )
    with _quiet_warnings():
        search.fit(ctx.X_train, ctx.y_train)
    return dict(search.best_params_), float(search.best_score_), trials


def _pick_models(
    ctx: Context, models: list[str] | None, top: int, leaderboard: pd.DataFrame | None
) -> list[ModelSpec]:
    if models:
        keys = [k for k in resolve_keys(models) if k in REGISTRY]
    elif leaderboard is not None and not leaderboard.empty:
        ok = leaderboard[
            (leaderboard["status"] == "ok")
            & (~leaderboard["key"].isin(["baseline", "ensemble", "stacking"]))
        ]
        keys = [str(k).split("@")[0] for k in ok["key"]]
        members = []
        for key in keys:
            if (
                key not in members
                and key in REGISTRY
                and REGISTRY[key].space(ctx.task)
                and REGISTRY[key].available
            ):
                members.append(key)
        keys = members[:top]
    else:
        keys = [
            k for k in DEFAULT_TUNABLE if REGISTRY[k].available and ctx.task in REGISTRY[k].tasks
        ][:top]
    specs = []
    for key in keys:
        spec = REGISTRY[key]
        if ctx.task not in spec.tasks:
            note(f"Skipping {key}: it doesn't support {ctx.task}.")
        elif not spec.available:
            note(f"Skipping {key}: '{spec.requires}' isn't installed.")
        elif not spec.space(ctx.task):
            note(f"Skipping {key}: it has nothing to tune (it tunes itself internally).")
        else:
            specs.append(spec)
    if not specs:
        raise PlainMLError(
            "No tunable models selected.", hint="Try: plainml tune SOURCE --models rf,lightgbm"
        )
    return specs


def tune(
    source: Any = "latest",
    target: Any = None,
    *,
    models: list[str] | None = None,
    top: int = 3,
    trials: int = 30,
    timeout: str | float | None = None,
    metric: str | None = None,
    cv: int | None = None,
    seed: int | None = None,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    verbose: bool = True,
) -> TrainResult:
    """Tune models from a previous run (``source`` = run) or on a data file (with ``target``)."""
    if trials < 1:
        raise PlainMLError("--trials must be at least 1.")
    seconds = parse_duration(timeout)
    run = None
    is_data = isinstance(source, pd.DataFrame) or (target is not None)
    if not is_data:
        try:
            run = load_run(resolve_run(source, out_dir), out_dir)
        except PlainMLError:
            if Path(str(source)).is_file():
                raise PlainMLError("Tuning a data file needs --target COLUMN.") from None
            raise
    if run is not None:
        if run.info.get("kind") not in ("train", "tune"):
            raise PlainMLError(
                f"Run '{run.name}' is a {run.info.get('kind')} run; tune works on training runs."
            )
        data = run.info["data"]["source"]
        if data == "<DataFrame>":
            raise PlainMLError(
                "That run was trained on an in-memory DataFrame.",
                hint="Pass the data: plainml tune DATA --target COLUMN",
            )
        target = run.info["target"]
        stored = {
            k: v
            for k, v in run.info.get("options", {}).items()
            if k in TrainOptions.__dataclass_fields__
        }
        options = {**stored, "out_dir": str(out_dir), "name": f"{Path(str(data)).stem}-tuned"}
    else:
        data = source
        options = {"out_dir": str(out_dir), "name": None}
    for key, value in (("metric", metric), ("cv", cv), ("seed", seed)):
        if value is not None:
            options[key] = value
    options.pop("time_budget", None)
    opts = TrainOptions.build(**options)

    with quiet(not verbose):
        ctx = prepare(data, target, opts)
        if (
            run is not None
            and run.info["data"].get("fingerprint")
            and run.info["data"]["fingerprint"] != ctx.data_fingerprint
        ):
            warn(
                "The data has changed since that run, so 'before' scores are recomputed on the new data."
            )
        specs = _pick_models(ctx, models, top, run.leaderboard if run is not None else None)
        method = "Optuna (TPE)" if is_installed("optuna") else "random search"
        if not is_installed("optuna"):
            note(
                'Optuna isn\'t installed, so using random search. For smarter search: pip install "plainml[tune]"'
            )
        if seconds and method == "random search":
            note("--timeout needs Optuna; random search runs all trials.")
        per_model = seconds / len(specs) if seconds else None
        heading(
            f"Tuning {len(specs)} model(s) with {method}: up to {trials} settings each, {ctx.folds}-fold CV"
        )

        candidates: list[Candidate] = []
        baseline = Candidate("baseline", REGISTRY["baseline"].name(ctx.task), baseline=True)
        baseline.template = build_estimator(ctx, REGISTRY["baseline"])
        cross_validate_candidate(ctx, baseline)
        candidates.append(baseline)
        summary = []
        for spec in specs:
            name = spec.name(ctx.task)
            base = build_estimator(ctx, spec)
            default = Candidate(f"{spec.key}@default", f"{name} (default)", template=base)
            try:
                cross_validate_candidate(ctx, default)
            except Exception as exc:
                default.status, default.note = "failed", str(exc).splitlines()[0][:120]
            candidates.append(default)
            prefix = param_prefix(base)
            space = spec.space(ctx.task)
            started = time.time()
            progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(bar_width=20),
                TextColumn("[muted]{task.completed}/{task.total}[/]"),
                TimeElapsedColumn(),
                console=console,
                auto_refresh=LIVE_REFRESH,
                transient=True,
            )
            try:
                with progress:
                    bar = progress.add_task(f"Tuning {name}", total=trials)

                    def on_trial(
                        study: Any, bar: Any = bar, progress: Any = progress, name: str = name
                    ) -> None:
                        try:
                            shown = fmt_num(ctx.metric.display(study.best_value))
                        except ValueError:
                            shown = "—"
                        progress.update(
                            bar,
                            advance=1,
                            description=f"Tuning {name} · best {ctx.metric.label} {shown}",
                        )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if method.startswith("Optuna"):
                            params, _, done = _search_optuna(
                                ctx, base, space, prefix, trials, per_model, on_trial
                            )
                        else:
                            params, _, done = _search_random(ctx, base, space, prefix, trials)
            except KeyboardInterrupt:
                warn("Stopped tuning early.")
                break
            except Exception as exc:
                warn(f"Tuning {esc(name)} failed: {esc(str(exc).splitlines()[0][:120])}")
                continue
            tuned = Candidate(
                spec.key,
                f"{name} (tuned)",
                template=clone(base).set_params(**params),
                params=params,
            )
            cross_validate_candidate(ctx, tuned)
            candidates.append(tuned)
            before = default.scores.get(ctx.metric.key)
            after = tuned.scores.get(ctx.metric.key)
            summary.append(
                {
                    "key": spec.key,
                    "name": name,
                    "before": before,
                    "after": after,
                    "trials": done,
                    "seconds": round(time.time() - started, 1),
                    "params": {k[len(prefix) :]: _plain(v) for k, v in params.items()},
                }
            )
            console.print(
                f"[good]✓[/] {name:<30} {ctx.metric.label} {fmt_num(before)} → [bold]{fmt_num(after)}[/] "
                f"[muted]({done} settings, {time.time() - started:.0f}s)[/]"
            )
        if not summary:
            raise PlainMLError("No model finished tuning.")
        _render_summary(summary, ctx)
        extra = {
            "tuning": {
                "method": method,
                "trials": trials,
                "source_run": run.name if run is not None else None,
                "models": summary,
            }
        }
        return finalize(ctx, candidates, kind="tune", extra=extra)


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _render_summary(summary: list[dict[str, Any]], ctx: Context) -> None:
    heading("Tuning result")
    table = Table(box=None, header_style="muted", pad_edge=False)
    table.add_column("Model")
    table.add_column("Default", justify="right")
    table.add_column("Tuned", justify="right", style="bold")
    table.add_column("Change", justify="right")
    table.add_column("Best settings", overflow="fold", style="muted")
    for entry in summary:
        before, after = entry["before"], entry["after"]
        change = ""
        if before is not None and after is not None and np.isfinite(before) and np.isfinite(after):
            delta = after - before
            improved = (delta > 0) == ctx.metric.greater_is_better and abs(delta) > 1e-12
            change = (
                f"[{'good' if improved else 'muted'}]{'+' if delta >= 0 else ''}{fmt_num(delta)}[/]"
            )
        settings = ", ".join(
            f"{k}={fmt_num(v) if isinstance(v, float) else v}" for k, v in entry["params"].items()
        )
        table.add_row(esc(entry["name"]), fmt_num(before), fmt_num(after), change, esc(settings))
    console.print(table)
    info("Tuned and default versions go on one leaderboard below; the best overall is saved.")
