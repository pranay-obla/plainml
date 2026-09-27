"""The ``plainml`` command line tool."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

from plainml import __version__
from plainml.console import console, error, esc, heading, info, note, set_quiet, success
from plainml.errors import PlainMLError, did_you_mean

ISSUES_URL = "https://github.com/pranay-obla/plainml/issues"
CONTEXT = {"help_option_names": ["-h", "--help"], "max_content_width": 100}

SECTIONS = [
    ("Start here", ["train", "profile", "clean"]),
    ("Use a trained model", ["predict", "evaluate", "explain", "drift", "report"]),
    ("Improve a model", ["tune", "importance", "select"]),
    ("Other kinds of problem", ["cluster", "anomaly", "forecast"]),
    ("Deploy and share", ["web", "serve", "deploy", "export"]),
    ("Housekeeping", ["runs", "compare", "models", "init"]),
]


class PlainMLGroup(click.Group):
    """Adds grouped help, 'did you mean' for commands, and friendly error handling."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        for title, names in SECTIONS:
            rows = []
            for name in names:
                command = self.get_command(ctx, name)
                if command is None or command.hidden:
                    continue
                rows.append((name, command.get_short_help_str(limit=70)))
            if rows:
                with formatter.section(title):
                    formatter.write_dl(rows)

    def resolve_command(self, ctx: click.Context, args: list[str]) -> Any:
        name = args[0] if args else None
        if name and name not in self.commands and not name.startswith("-"):
            visible = [n for n, c in self.commands.items() if not c.hidden]
            if Path(name).suffix.lower() in (".csv", ".xlsx", ".tsv", ".parquet", ".json"):
                raise click.UsageError(
                    f"'{name}' looks like a data file. Did you mean: plainml train {name} --target COLUMN"
                )
            raise click.UsageError(f"No command '{name}'.{did_you_mean(name, visible)}")
        return super().resolve_command(ctx, args)

    def invoke(self, ctx: click.Context) -> Any:
        debug = bool(ctx.params.get("debug")) or os.environ.get("PLAINML_DEBUG") == "1"
        try:
            return super().invoke(ctx)
        except PlainMLError as exc:
            if debug:
                raise
            error(exc.message, exc.hint)
            ctx.exit(1)
        except KeyboardInterrupt:
            console.print("\n[muted]Cancelled.[/]")
            ctx.exit(130)
        except (click.exceptions.Exit, click.exceptions.Abort, click.ClickException):
            raise
        except Exception as exc:
            if debug:
                raise
            error(
                f"Unexpected error: {type(exc).__name__}: {exc}",
                f"Run again with --debug for the full traceback, and please report it at {ISSUES_URL}",
            )
            ctx.exit(1)


def _explicit(ctx: click.Context, names: list[str] | None = None) -> dict[str, Any]:
    """Only the options the user actually typed (so config files can fill in the rest)."""
    values = {}
    for name, value in ctx.params.items():
        if names is not None and name not in names:
            continue
        if ctx.get_parameter_source(name) in (ParameterSource.DEFAULT, None):
            continue
        if isinstance(value, tuple):
            if not value:
                continue
            value = [v for item in value for v in str(item).split(",") if v.strip()]
        values[name] = value
    return values


def load_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Options for reading spreadsheets and databases, shared by data commands."""
    func = click.option(
        "--engine",
        type=click.Choice(["auto", "pandas", "polars"]),
        default="auto",
        show_default=True,
        help="Reader for big CSV/Parquet files.",
    )(func)
    func = click.option(
        "--table",
        "table_name",
        metavar="NAME",
        help="Database table to read (with a database URL).",
    )(func)
    func = click.option("--query", metavar="SQL", help="SQL query to run (with a database URL).")(
        func
    )
    func = click.option(
        "--sheet", metavar="NAME", help="Excel sheet to read (default: the first)."
    )(func)
    return func


def _load_kwargs(
    sheet: str | None, query: str | None, table_name: str | None, engine: str
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"engine": engine}
    if sheet is not None:
        kwargs["sheet"] = int(sheet) if sheet.isdigit() else sheet
    if query:
        kwargs["query"] = query
    if table_name:
        kwargs["table"] = table_name
    return kwargs


def _sample(value: float | None) -> float | int | None:
    if value is None:
        return None
    return int(value) if value >= 1 else float(value)


def _open(path: Path) -> None:
    try:
        click.launch(str(path))
    except Exception:  # pragma: no cover - no browser available
        note(f"Open {esc(path)} in a browser.")


# --- main group -----------------------------------------------------------------------------


@click.group(
    cls=PlainMLGroup,
    context_settings=CONTEXT,
    invoke_without_command=True,
    help="""Machine learning in plain English: from a spreadsheet to a trained, explained model.

\b
Quick start:
  plainml train data.csv --target price       train and compare models
  plainml predict latest new.csv -o out.csv   predict with the best model
  plainml                                     guided mode: answers a few questions

Run 'plainml COMMAND --help' for a command's options and examples.""",
)
@click.version_option(__version__, "-V", "--version", prog_name="plainml")
@click.option("--debug", is_flag=True, help="Show full tracebacks on errors.")
@click.option("-q", "--quiet", is_flag=True, help="Only print errors.")
@click.option("--no-color", is_flag=True, help="Plain text output.")
@click.pass_context
def cli(ctx: click.Context, debug: bool, quiet: bool, no_color: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        # Old Windows consoles (cp1252) can't print ✓ or ★: show '?' rather than crash.
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding and encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    set_quiet(quiet)
    if no_color or os.environ.get("NO_COLOR"):
        console.no_color = True
    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            from plainml.wizard import run_wizard

            run_wizard()
        else:
            click.echo(ctx.get_help())


# --- train ----------------------------------------------------------------------------------

TRAIN_EXAMPLES = """\b
Examples:
  plainml train churn.csv --target churned
  plainml train houses.xlsx -t price --metric mae --quick
  plainml train data.csv -t label --models rf,lightgbm,xgboost --time-budget 10m
  plainml train data.csv -t label --drop customer_id --balance smote
  plainml train reviews.csv -t is_spam,is_urgent        (multi-label)
  plainml train --config plainml.yaml
"""


@cli.command(
    epilog=TRAIN_EXAMPLES, short_help="Train and compare models; save the best with a report."
)
@click.argument("data", required=False)
@click.option(
    "-t", "--target", multiple=True, help="Column to predict. Repeat or comma-separate for several."
)
@click.option(
    "--task",
    type=click.Choice(["auto", "classification", "regression"]),
    help="Override the auto-detected task.",
)
@click.option(
    "-m",
    "--metric",
    help="Metric to rank models by (default: f1 or rmse). E.g. accuracy, roc_auc, r2, mae.",
)
@click.option(
    "--models", multiple=True, help="Only try these models, e.g. rf,xgboost. See: plainml models"
)
@click.option("--exclude", multiple=True, help="Skip these models.")
@click.option("--quick", is_flag=True, help="Only fast models, no ensemble.")
@click.option(
    "--thorough",
    is_flag=True,
    help="Also try the extra models (AdaBoost, Gaussian process, PyTorch...) and a stacked ensemble.",
)
@click.option("--cv", type=click.IntRange(2, 20), help="Cross-validation folds.  [default: 5]")
@click.option(
    "--test-size",
    type=click.FloatRange(0.05, 0.5),
    help="Share of rows held out for the final test.  [default: 0.2]",
)
@click.option("--seed", type=int, help="Random seed for reproducible results.  [default: 42]")
@click.option(
    "--time-budget",
    metavar="DURATION",
    help="Stop starting new models after this long, e.g. 90s, 5m, 1h.",
)
@click.option(
    "--balance",
    type=click.Choice(["auto", "none", "weights", "smote"]),
    help="Handle imbalanced classes.  [default: auto]",
)
@click.option(
    "--threshold",
    type=click.Choice(["auto", "on", "off"]),
    help="Tune the yes/no decision threshold.  [default: auto]",
)
@click.option(
    "--log-target",
    is_flag=True,
    help="Model log(target): helps with skewed, positive amounts like prices.",
)
@click.option(
    "--calibrate",
    is_flag=True,
    help="Adjust predicted probabilities so '70% sure' really means right 70% of the time.",
)
@click.option(
    "--mlflow",
    is_flag=True,
    help="Also log the run to MLflow (the server in MLFLOW_TRACKING_URI, or a local store).",
)
@click.option(
    "--ensemble/--no-ensemble",
    default=None,
    help="Also try averaging the top 3 models.  [default: on]",
)
@click.option(
    "--refit/--no-refit",
    default=None,
    help="Retrain the winner on all rows before saving.  [default: on]",
)
@click.option("--save-all", is_flag=True, help="Also save every model, not just the best.")
@click.option("--zip", "zip", is_flag=True, help="Also zip the run folder.")
@click.option("--drop", multiple=True, metavar="COLUMNS", help="Columns to ignore.")
@click.option(
    "--keep", multiple=True, metavar="COLUMNS", help="Columns to use even if they look like IDs."
)
@click.option(
    "--sample",
    type=float,
    metavar="N",
    help="Train on a random sample: a row count, or a fraction like 0.1.",
)
@click.option("--n-jobs", type=int, help="CPU cores to use (-1 = all).  [default: -1]")
@click.option("-o", "--out", "out_dir", metavar="DIR", help="Where to save runs.  [default: runs]")
@click.option("--name", help="Name for the run folder.")
@click.option("--report/--no-report", default=None, help="Write report.html.  [default: on]")
@click.option(
    "--private",
    is_flag=True,
    default=None,
    help="Keep raw data out of the report, model and run folder (for sharing results).",
)
@click.option("--open", "open_report", is_flag=True, help="Open the report when done.")
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False),
    help="YAML file of options (see: plainml init).",
)
@load_options
@click.pass_context
def train(
    ctx: click.Context,
    data: str | None,
    config: str | None,
    open_report: bool,
    sheet: str | None,
    query: str | None,
    table_name: str | None,
    engine: str,
    **_: Any,
) -> None:
    """Train many models, rank them with cross-validation, and save the best one.

    DATA is a CSV/Excel/Parquet/JSON file, a URL, or a database URL.
    """
    from plainml.training import train as run_train

    options = _explicit(
        ctx,
        [
            "task",
            "metric",
            "models",
            "exclude",
            "quick",
            "thorough",
            "cv",
            "test_size",
            "seed",
            "time_budget",
            "balance",
            "threshold",
            "log_target",
            "calibrate",
            "mlflow",
            "ensemble",
            "refit",
            "save_all",
            "zip",
            "drop",
            "keep",
            "sample",
            "n_jobs",
            "out_dir",
            "name",
            "report",
            "private",
        ],
    )
    targets = ctx.params.get("target") or ()
    target = [t for item in targets for t in item.split(",") if t.strip()] or None
    if "sample" in options:
        options["sample"] = _sample(options["sample"])
    options.update(
        {
            k: v
            for k, v in _load_kwargs(sheet, query, table_name, engine).items()
            if k != "engine" or v != "auto"
        }
    )
    if data is None and config is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            from plainml.wizard import run_wizard

            run_wizard(start="train")
            return
        raise click.UsageError(
            "Give a data file (plainml train DATA --target COLUMN) or --config FILE."
        )
    if data is not None and target is None and config is None:
        _prompt_for_target(data, options)
        return
    result = run_train(data, target, config=config, **options)
    if open_report and result.report_path.is_file():
        _open(result.report_path)


def _prompt_for_target(data: str, options: dict[str, Any]) -> None:
    """`plainml train data.csv` without --target: ask (interactive) or explain (scripts)."""
    from plainml.io import load_data

    df = load_data(
        data, **{k: v for k, v in options.items() if k in ("sheet", "query", "table", "engine")}
    )
    columns = list(df.columns)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise PlainMLError(
            "Which column should the model predict?",
            hint=f"Add --target COLUMN. Columns: {', '.join(columns[:20])}{' …' if len(columns) > 20 else ''}",
        )
    from plainml.wizard import ask_target

    target = ask_target(df)
    if target is None:
        raise click.Abort()
    from plainml.training import train as run_train

    run_train(df if not isinstance(data, str) else data, target, **options)


# --- profile / clean ------------------------------------------------------------------------


@cli.command(short_help="Summarise a dataset and flag problems, without training.")
@click.argument("data")
@click.option("-t", "--target", help="Also check this target column (imbalance, leakage, task).")
@click.option("--drop", multiple=True, metavar="COLUMNS", help="Columns to ignore.")
@click.option(
    "--keep", multiple=True, metavar="COLUMNS", help="Columns to use even if they look like IDs."
)
@click.option(
    "--html", "html_path", type=click.Path(dir_okay=False), help="Also write an HTML version."
)
@click.option("--open", "open_report", is_flag=True, help="Open the HTML version.")
@load_options
def profile(
    data: str,
    target: str | None,
    drop: tuple[str, ...],
    keep: tuple[str, ...],
    html_path: str | None,
    open_report: bool,
    **load: Any,
) -> None:
    """Show each column's type, gaps and values, plus warnings (imbalance, leaks, IDs...).

    \b
    Examples:
      plainml profile sales.csv
      plainml profile churn.csv --target churned --html profile.html --open
    """
    from plainml.profiling import profile as run_profile

    split = lambda values: [c for v in values for c in v.split(",") if c.strip()]  # noqa: E731
    prof = run_profile(data, target, drop=split(drop), keep=split(keep), **_load_kwargs(**load))
    if html_path or open_report:
        from plainml.report import write_profile_report

        path = write_profile_report(html_path or "profile.html", prof)
        success(f"Saved {esc(path)}")
        if open_report:
            _open(path)


@cli.command(short_help="Fix common data problems and save a cleaned copy.")
@click.argument("data")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Where to save the cleaned data.  [default: DATA_clean.csv]",
)
@click.option(
    "-t",
    "--target",
    help="Target column: rows missing it are dropped; it's never imputed or encoded.",
)
@click.option(
    "--duplicates/--keep-duplicates", default=True, show_default=True, help="Remove duplicate rows."
)
@click.option(
    "--impute/--no-impute",
    default=False,
    show_default=True,
    help="Fill blanks (median / most common value).",
)
@click.option(
    "--outliers",
    type=click.Choice(["none", "clip", "remove"]),
    default="none",
    show_default=True,
    help="Handle extreme numeric values (1.5×IQR rule).",
)
@click.option(
    "--encode", is_flag=True, help="One-hot encode text categories (for tools that need numbers)."
)
@click.option(
    "--drop-ids/--keep-ids",
    default=True,
    show_default=True,
    help="Remove ID-like and constant columns.",
)
@click.option("--dry-run", is_flag=True, help="Show what would change without saving.")
@load_options
def clean(
    data: str,
    output: str | None,
    target: str | None,
    duplicates: bool,
    impute: bool,
    outliers: str,
    encode: bool,
    drop_ids: bool,
    dry_run: bool,
    **load: Any,
) -> None:
    """Standardise blanks, fix numbers stored as text, parse dates, remove duplicates and more.

    \b
    Examples:
      plainml clean raw.csv -o clean.csv
      plainml clean raw.xlsx --impute --outliers clip --target price
    """
    from plainml.cleaning import clean as run_clean

    run_clean(
        data,
        output=None if dry_run else (output or _default_clean_path(data)),
        target=target,
        drop_duplicates=duplicates,
        impute=impute,
        outliers=outliers,
        encode=encode,
        drop_ids=drop_ids,
        **_load_kwargs(**load),
    )


def _default_clean_path(data: str) -> str:
    path = Path(data)
    return str(path.with_name(f"{path.stem}_clean.csv")) if path.suffix else "clean.csv"


# --- using a model --------------------------------------------------------------------------


@cli.command(short_help="Predict on new data with a trained model.")
@click.argument("model")
@click.argument("data", required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Save predictions (.csv, .xlsx, .json, .parquet).",
)
@click.option("--proba", is_flag=True, help="Add a probability column per class.")
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if any input column is missing (instead of treating it as blank).",
)
@click.option("--horizon", type=int, help="Forecast models: how many periods ahead.")
@click.option(
    "--chunk-size",
    type=click.IntRange(1000),
    help="Stream a big CSV/Parquet file this many rows at a time (needs -o).",
)
@click.option("--show", type=int, default=10, show_default=True, help="Rows to print.")
@click.option("--runs-dir", default="runs", show_default=True, help="Where runs are saved.")
@load_options
def predict(
    model: str,
    data: str | None,
    output: str | None,
    proba: bool,
    strict: bool,
    horizon: int | None,
    chunk_size: int | None,
    show: int,
    runs_dir: str,
    **load: Any,
) -> None:
    """Predict with MODEL: a .joblib file, a run folder, part of a run name, or 'latest'.

    \b
    Examples:
      plainml predict latest new_customers.csv -o predictions.csv
      plainml predict runs/20260924-101500_churn new.xlsx --proba
      plainml predict latest --horizon 30          (forecast models)
      plainml predict latest huge.csv -o out.csv --chunk-size 200000
    """
    from plainml.predicting import predict as run_predict
    from plainml.predicting import render_predictions

    frame = run_predict(
        model,
        data,
        output=output,
        proba=proba,
        strict=strict,
        horizon=horizon,
        out_dir=runs_dir,
        chunk_size=chunk_size,
        **_load_kwargs(**load),
    )
    render_predictions(frame, show, saved=bool(output))
    if output:
        success(f"Saved {frame.attrs.get('rows_written', len(frame)):,} rows to {esc(output)}")


@cli.command(short_help="Score a trained model on new labelled data (spot drift).")
@click.argument("model")
@click.argument("data")
@click.option(
    "--report", "report_path", type=click.Path(dir_okay=False), help="Also write an HTML report."
)
@click.option("--runs-dir", default="runs", show_default=True)
@load_options
def evaluate(model: str, data: str, report_path: str | None, runs_dir: str, **load: Any) -> None:
    """Check how MODEL does on DATA (which must include the target column).

    Compares against the scores measured at training time, so you can tell when the
    world has changed and the model needs retraining.

    \b
    Example:
      plainml evaluate latest march_labelled.csv --report march.html
    """
    from plainml.predicting import evaluate as run_evaluate

    run_evaluate(model, data, report=report_path, out_dir=runs_dir, **_load_kwargs(**load))


@cli.command(short_help="Explain what a model relies on, in plain English.")
@click.argument("model", default="latest")
@click.argument("data", required=False)
@click.option("--top", type=int, default=15, show_default=True, help="How many columns to show.")
@click.option(
    "--shap",
    "use_shap",
    is_flag=True,
    help='Also compute SHAP values (pip install "plainml[explain]").',
)
@click.option(
    "--row", type=int, help="Explain the prediction for this row number (0-based) of DATA."
)
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Save the importance table.")
@click.option("--runs-dir", default="runs", show_default=True)
@load_options
def explain(
    model: str,
    data: str | None,
    top: int,
    use_shap: bool,
    row: int | None,
    output: str | None,
    runs_dir: str,
    **load: Any,
) -> None:
    """Which columns matter, which way they push predictions, and why one row got its prediction.

    Uses the run's held-out test rows unless you pass DATA.

    \b
    Examples:
      plainml explain
      plainml explain latest new.csv --row 3
      plainml explain runs/20260924-101500_churn --shap
    """
    from plainml.explaining import explain as run_explain

    run_explain(
        model,
        data,
        top=top,
        use_shap=use_shap,
        row=row,
        output=output,
        out_dir=runs_dir,
        **_load_kwargs(**load),
    )


@cli.command(short_help="Check whether new data has drifted from the training data.")
@click.argument("reference")
@click.argument("data")
@click.option("-t", "--target", help="Target column to leave out (when REFERENCE is a data file).")
@click.option("--out", "out_dir", default="runs", show_default=True, help="Where to save the run.")
@click.option("--name", help="Name for the run folder.")
@click.option("--report/--no-report", default=True, show_default=True, help="Write report.html.")
@click.option("--open", "open_report", is_flag=True, help="Open the report when done.")
@load_options
def drift(
    reference: str,
    data: str,
    target: str | None,
    out_dir: str,
    name: str | None,
    report: bool,
    open_report: bool,
    **load: Any,
) -> None:
    """Compare DATA with a model's training data (REFERENCE = 'latest', a run, a .joblib) or with
    an older data file. No labels needed.

    \b
    Examples:
      plainml drift latest this_month.csv
      plainml drift january.csv june.csv --target churned
    """
    from plainml.drift import check_drift

    result = check_drift(
        reference,
        data,
        target=target,
        out_dir=out_dir,
        name=name,
        report=report,
        **_load_kwargs(**load),
    )
    if open_report and result.report_path and result.report_path.is_file():
        _open(result.report_path)


@cli.command(short_help="Rebuild (and open) a run's HTML report.")
@click.argument("run", default="latest")
@click.option("--open/--no-open", "open_report", default=True, show_default=True)
@click.option("--runs-dir", default="runs", show_default=True)
def report(run: str, open_report: bool, runs_dir: str) -> None:
    """Regenerate report.html for RUN ('latest', a folder, or part of a run name)."""
    from plainml.report import write_report
    from plainml.runs import resolve_run

    path = write_report(resolve_run(run, runs_dir))
    success(f"Report: {esc(path)}")
    if open_report:
        _open(path)


# --- improving ------------------------------------------------------------------------------


@cli.command(short_help="Search hyperparameters to squeeze out more accuracy.")
@click.argument("source", default="latest")
@click.option("-t", "--target", help="Target column (when SOURCE is a data file).")
@click.option("--models", multiple=True, help="Models to tune (default: the run's best models).")
@click.option(
    "--top", type=int, default=3, show_default=True, help="Tune this many of the run's best models."
)
@click.option(
    "--trials", type=int, default=30, show_default=True, help="Settings to try per model."
)
@click.option("--timeout", metavar="DURATION", help="Overall time limit, e.g. 10m.")
@click.option("-m", "--metric", help="Metric to optimise (default: the run's).")
@click.option("--cv", type=click.IntRange(2, 20), help="Cross-validation folds.")
@click.option("--seed", type=int, help="Random seed.")
@click.option("--open", "open_report", is_flag=True, help="Open the report when done.")
@click.option("--runs-dir", default="runs", show_default=True)
def tune(
    source: str,
    target: str | None,
    models: tuple[str, ...],
    top: int,
    trials: int,
    timeout: str | None,
    metric: str | None,
    cv: int | None,
    seed: int | None,
    open_report: bool,
    runs_dir: str,
) -> None:
    """Tune the best models from a run (default: latest), or models on a data file.

    Uses Optuna when installed (pip install "plainml[tune]"), otherwise random search.

    \b
    Examples:
      plainml tune
      plainml tune latest --trials 100 --timeout 20m
      plainml tune data.csv -t price --models lightgbm,rf
    """
    from plainml.tuning import tune as run_tune

    model_list = [m for item in models for m in item.split(",") if m.strip()] or None
    result = run_tune(
        source,
        target=target,
        models=model_list,
        top=top,
        trials=trials,
        timeout=timeout,
        metric=metric,
        cv=cv,
        seed=seed,
        out_dir=runs_dir,
    )
    if open_report and result.report_path.is_file():
        _open(result.report_path)


IMPORTANCE_METHODS_HELP = (
    "Methods (comma-separated), or 'all'. Default: mutual_info, f_test, correlation, random_forest, "
    "l1, permutation, rfe, boruta. Also: chi2, variance, extra_trees, gradient_boosting, linear, "
    "drop_column, shap, forward, backward, exhaustive, stability."
)


@cli.command(short_help="Rank columns with up to 19 methods; find the ones that matter.")
@click.argument("data")
@click.option("-t", "--target", required=True, help="Column to predict.")
@click.option("--methods", multiple=True, help=IMPORTANCE_METHODS_HELP)
@click.option(
    "-k",
    "--keep-top",
    "k",
    type=int,
    help="How many columns to recommend (default: those that clearly help).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Also save DATA with only the recommended columns (+ target).",
)
@click.option("--drop", multiple=True, metavar="COLUMNS", help="Columns to ignore.")
@click.option(
    "--sample",
    type=float,
    metavar="N",
    help="Use a random sample: a row count, or a fraction like 0.1.",
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--out", "out_dir", default="runs", show_default=True, help="Where to save the run.")
@click.option("--name", help="Name for the run folder.")
@click.option("--private", is_flag=True, help="Keep raw values out of the report.")
@click.option("--open", "open_report", is_flag=True, help="Open the report when done.")
@load_options
def importance(
    data: str,
    target: str,
    methods: tuple[str, ...],
    k: int | None,
    output: str | None,
    drop: tuple[str, ...],
    sample: float | None,
    seed: int,
    out_dir: str,
    name: str | None,
    private: bool,
    open_report: bool,
    **load: Any,
) -> None:
    """Which columns matter for predicting the target, measured many different ways.

    Filters (mutual information, F-test, correlation...), model-based importances (random
    forest, L1...), model-agnostic ones (permutation, drop-column, SHAP) and searches (RFE,
    forward/backward selection, exhaustive search, Boruta, stability selection) are combined
    into one consensus ranking, with a recommended set and a report.

    \b
    Examples:
      plainml importance data.csv -t price
      plainml importance data.csv -t churned --methods all --open
      plainml importance data.csv -t churned --methods boruta,exhaustive -o reduced.csv
    """
    from plainml.importance import feature_importance

    result = feature_importance(
        data,
        target,
        methods=[m for item in methods for m in item.split(",") if m.strip()] or None,
        k=k,
        output=output,
        drop=[c for v in drop for c in v.split(",") if c.strip()],
        sample=_sample(sample),
        seed=seed,
        out_dir=out_dir,
        name=name,
        private=private,
        **_load_kwargs(**load),
    )
    if open_report and result.report_path and result.report_path.is_file():
        _open(result.report_path)


@cli.command(name="select", short_help="Keep only the columns that matter; save a smaller dataset.")
@click.argument("data")
@click.option("-t", "--target", required=True, help="Column to predict.")
@click.option(
    "-k",
    "--keep-top",
    "k",
    type=int,
    help="How many columns to keep (default: those that clearly help).",
)
@click.option("--methods", multiple=True, help=IMPORTANCE_METHODS_HELP)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Save DATA with only the selected columns (+ target).",
)
@click.option("--drop", multiple=True, metavar="COLUMNS", help="Columns to ignore.")
@load_options
def select_command(
    data: str,
    target: str,
    k: int | None,
    methods: tuple[str, ...],
    output: str | None,
    drop: tuple[str, ...],
    **load: Any,
) -> None:
    """Like 'importance', without saving a run: rank the columns and keep the best ones.

    \b
    Examples:
      plainml select data.csv -t price
      plainml select data.csv -t churned -k 10 -o reduced.csv
    """
    from plainml.importance import select_features

    select_features(
        data,
        target,
        k=k,
        methods=[m for item in methods for m in item.split(",") if m.strip()] or None,
        output=output,
        drop=[c for v in drop for c in v.split(",") if c.strip()],
        **_load_kwargs(**load),
    )


# --- other problem types --------------------------------------------------------------------


@cli.command(short_help="Group similar rows (no target needed).")
@click.argument("data")
@click.option(
    "-k",
    "--clusters",
    "k",
    default="auto",
    show_default=True,
    help="Number of groups, a range like 2-8, or auto.",
)
@click.option(
    "--algorithms", multiple=True, help="Subset of: kmeans, agglomerative, gmm, hdbscan, kmedoids."
)
@click.option("--drop", multiple=True, metavar="COLUMNS", help="Columns to ignore.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False), help="Save DATA with a 'cluster' column."
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--out", "out_dir", default="runs", show_default=True, help="Where to save the run.")
@click.option("--name", help="Name for the run folder.")
@click.option("--private", is_flag=True, help="Keep raw values out of the report and run folder.")
@click.option("--open", "open_report", is_flag=True, help="Open the report when done.")
@load_options
def cluster(
    data: str,
    k: str,
    algorithms: tuple[str, ...],
    drop: tuple[str, ...],
    output: str | None,
    seed: int,
    out_dir: str,
    name: str | None,
    private: bool,
    open_report: bool,
    **load: Any,
) -> None:
    """Find natural groups (segments) in DATA and describe what makes each one different.

    \b
    Examples:
      plainml cluster customers.csv
      plainml cluster customers.csv -k 4 --drop customer_id -o segments.csv
    """
    from plainml.clustering import cluster as run_cluster

    result = run_cluster(
        data,
        k=k,
        algorithms=[a for item in algorithms for a in item.split(",") if a.strip()] or None,
        drop=[c for v in drop for c in v.split(",") if c.strip()],
        output=output,
        seed=seed,
        out_dir=out_dir,
        name=name,
        private=private,
        **_load_kwargs(**load),
    )
    if open_report and result.report_path.is_file():
        _open(result.report_path)


@cli.command(short_help="Find unusual rows (fraud, errors, outliers).")
@click.argument("data")
@click.option(
    "--contamination",
    default="auto",
    show_default=True,
    help="Expected share of anomalies, e.g. 0.02, or auto.",
)
@click.option(
    "--label", help="Optional column marking known anomalies (1/0), used to score the detectors."
)
@click.option("--algorithms", multiple=True, help="Subset of: iforest, lof, ocsvm, robust.")
@click.option("--drop", multiple=True, metavar="COLUMNS", help="Columns to ignore.")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False), help="Save DATA with anomaly scores."
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--out", "out_dir", default="runs", show_default=True, help="Where to save the run.")
@click.option("--name", help="Name for the run folder.")
@click.option("--private", is_flag=True, help="Keep raw values out of the report and run folder.")
@click.option("--open", "open_report", is_flag=True, help="Open the report when done.")
@load_options
def anomaly(
    data: str,
    contamination: str,
    label: str | None,
    algorithms: tuple[str, ...],
    drop: tuple[str, ...],
    output: str | None,
    seed: int,
    out_dir: str,
    name: str | None,
    private: bool,
    open_report: bool,
    **load: Any,
) -> None:
    """Score every row by how unusual it is and explain what's odd about the top ones.

    \b
    Examples:
      plainml anomaly transactions.csv -o scored.csv
      plainml anomaly transactions.csv --label is_fraud --contamination 0.01
    """
    from plainml.anomaly import detect_anomalies

    result = detect_anomalies(
        data,
        contamination=contamination,
        label=label,
        algorithms=[a for item in algorithms for a in item.split(",") if a.strip()] or None,
        drop=[c for v in drop for c in v.split(",") if c.strip()],
        output=output,
        seed=seed,
        out_dir=out_dir,
        name=name,
        private=private,
        **_load_kwargs(**load),
    )
    if open_report and result.report_path.is_file():
        _open(result.report_path)


@cli.command(short_help="Forecast a value over time (sales, demand, traffic...).")
@click.argument("data")
@click.option("-t", "--target", required=True, help="Column to forecast.")
@click.option("--date", "date_column", help="Date/time column (auto-detected if left out).")
@click.option("--horizon", type=int, help="How many periods ahead (default: depends on frequency).")
@click.option("--freq", help="Frequency: D (daily), W, M, H... (auto-detected if left out).")
@click.option(
    "--agg",
    type=click.Choice(["sum", "mean", "last"]),
    default="sum",
    show_default=True,
    help="How to combine several rows with the same date.",
)
@click.option(
    "--models",
    multiple=True,
    help="Subset of: naive, seasonal_naive, ridge, rf, histgb, lightgbm, xgboost.",
)
@click.option(
    "--group",
    help="Forecast each value of this column separately (store, product, region...).",
)
@click.option(
    "--inputs",
    multiple=True,
    help="Columns known in advance that affect the target (promo, price...). Comma-separated.",
)
@click.option(
    "--future",
    type=click.Path(exists=True, dir_okay=False),
    help="File with the inputs' planned future values (date, inputs, and group if used).",
)
@click.option("--country", help="Add public holidays for this country code (US, GB, IN, DE...).")
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Save the forecast.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--out", "out_dir", default="runs", show_default=True, help="Where to save the run.")
@click.option("--name", help="Name for the run folder.")
@click.option("--open", "open_report", is_flag=True, help="Open the report when done.")
@load_options
def forecast(
    data: str,
    target: str,
    date_column: str | None,
    horizon: int | None,
    freq: str | None,
    agg: str,
    models: tuple[str, ...],
    group: str | None,
    inputs: tuple[str, ...],
    future: str | None,
    country: str | None,
    output: str | None,
    seed: int,
    out_dir: str,
    name: str | None,
    open_report: bool,
    **load: Any,
) -> None:
    """Backtest several forecasting models on DATA's history and forecast the future.

    Rows dated after the last known target value are read as plans for --inputs,
    so a promo calendar can sit in the same file as the history.

    \b
    Examples:
      plainml forecast sales.csv -t revenue --horizon 30
      plainml forecast visits.csv -t visitors --date day --freq D -o next_month.csv
      plainml forecast stores.csv -t sales --group store --inputs promo --country US
      plainml forecast sales.csv -t sales --inputs promo,price --future plans.csv
    """
    from plainml.forecasting import forecast as run_forecast

    result = run_forecast(
        data,
        target,
        date=date_column,
        horizon=horizon,
        freq=freq,
        agg=agg,
        models=[m for item in models for m in item.split(",") if m.strip()] or None,
        group=group,
        inputs=[c.strip() for item in inputs for c in item.split(",") if c.strip()] or None,
        future=future,
        country=country,
        output=output,
        seed=seed,
        out_dir=out_dir,
        name=name,
        **_load_kwargs(**load),
    )
    if open_report and result.report_path.is_file():
        _open(result.report_path)


# --- deploy ---------------------------------------------------------------------------------


@cli.command(short_help="Serve a model as a REST API (FastAPI).")
@click.argument("model", default="latest")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Use 0.0.0.0 to accept outside connections.",
)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option(
    "--api-key",
    envvar="PLAINML_API_KEY",
    help="Require this key in the X-API-Key header (or set PLAINML_API_KEY).",
)
@click.option("--runs-dir", default="runs", show_default=True)
def serve(model: str, host: str, port: int, api_key: str | None, runs_dir: str) -> None:
    """Start a prediction API for MODEL, with interactive docs at /docs.

    \b
    Example:
      plainml serve latest --port 8000
      curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \\
           -d '{"rows": [{"age": 42, "plan": "pro"}]}'
    """
    from plainml.serve import serve as run_serve

    run_serve(model, host=host, port=port, out_dir=runs_dir, api_key=api_key)


@cli.command(short_help="Package a model as a Docker image that serves it.")
@click.argument("model", default="latest")
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False),
    help="Folder to write.  [default: deploy/<run>]",
)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--python", "python_version", help="Python version for the image, e.g. 3.12.")
@click.option("--runs-dir", default="runs", show_default=True)
def deploy(
    model: str, output: str | None, port: int, python_version: str | None, runs_dir: str
) -> None:
    """Write a folder with a Dockerfile, pinned requirements and MODEL, ready to build.

    \b
    Example:
      plainml deploy latest
      docker build -t churn deploy/20260101-120000_churn
      docker run -p 8000:8000 -e PLAINML_API_KEY=secret churn
    """
    from plainml.deploy import deploy as run_deploy

    run_deploy(model, output=output, out_dir=runs_dir, port=port, python=python_version)


@cli.command(short_help="Export a model to ONNX or MLflow format.")
@click.argument("model", default="latest")
@click.option(
    "--format", "fmt", type=click.Choice(["onnx", "mlflow"]), default="onnx", show_default=True
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Output file.  [default: next to the model]",
)
@click.option("--runs-dir", default="runs", show_default=True)
def export(model: str, fmt: str, output: str | None, runs_dir: str) -> None:
    """Convert MODEL to ONNX (checked against the original) or save it as an MLflow model.

    \b
    Examples:
      plainml export latest -o churn.onnx
      plainml export latest --format mlflow -o churn_mlflow
    """
    if fmt == "mlflow":
        from plainml.tracking import export_mlflow

        export_mlflow(model, output=output, out_dir=runs_dir)
        return
    from plainml.export import export_onnx

    export_onnx(model, output=output, out_dir=runs_dir)


@cli.command(short_help="Open the point-and-click web app (Streamlit).")
@click.option("--port", type=int, default=8501, show_default=True)
@click.option("--runs-dir", default="runs", show_default=True)
def ui(port: int, runs_dir: str) -> None:
    """Upload data, pick a target, train, and download results, all in the browser."""
    from plainml.errors import require

    require("streamlit", "The web app")
    import subprocess

    app = Path(__file__).with_name("ui_app.py")
    info(f"Starting the plainml app on http://localhost:{port} (Ctrl-C to stop)")
    env = {**os.environ, "PLAINML_RUNS_DIR": runs_dir}
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(subprocess.call(command, env=env))


# --- housekeeping ---------------------------------------------------------------------------


@cli.command(short_help="List past runs, or delete old ones.")
@click.option("--runs-dir", default="runs", show_default=True)
@click.option("-n", "--limit", type=int, default=20, show_default=True)
@click.option(
    "--prune", is_flag=True, help="Delete old runs (asks first). Use with --keep / --older-than."
)
@click.option("--keep", type=click.IntRange(0), help="With --prune: keep the newest N runs.")
@click.option(
    "--older-than", metavar="AGE", help="With --prune: only runs older than this, e.g. 30d, 2w."
)
@click.option(
    "--kind", help="With --prune: only this kind of run (train, forecast, importance...)."
)
@click.option("-y", "--yes", is_flag=True, help="Don't ask for confirmation.")
def runs(
    runs_dir: str,
    limit: int,
    prune: bool,
    keep: int | None,
    older_than: str | None,
    kind: str | None,
    yes: bool,
) -> None:
    """List saved runs, newest last.

    \b
    Examples:
      plainml runs
      plainml runs --prune --keep 10
      plainml runs --prune --older-than 30d --kind drift
    """
    from rich.table import Table

    from plainml.console import fmt_num, parse_duration
    from plainml.runs import list_runs

    if prune or keep is not None or older_than:
        if not prune:
            raise click.UsageError("--keep and --older-than go with --prune.")
        _prune(runs_dir, keep, parse_duration(older_than), kind, yes)
        return

    frame = list_runs(runs_dir)
    if frame.empty:
        info(
            f"No runs in '{esc(runs_dir)}' yet. Train one with: plainml train DATA --target COLUMN"
        )
        return
    table = Table(header_style="muted", box=None, pad_edge=False)
    for column in ("Run", "Kind", "Data", "Target", "Best model", "Score", "Rows"):
        table.add_column(
            column, justify="right" if column in ("Score", "Rows") else "left", overflow="fold"
        )
    for _, row in frame.tail(limit).iterrows():
        score = (
            f"{row['metric']} {fmt_num(row['score'])}"
            if row["score"] is not None and row["metric"]
            else ""
        )
        rows = f"{int(row['rows']):,}" if row["rows"] == row["rows"] and row["rows"] else ""
        table.add_row(
            *(
                esc(v)
                for v in (
                    row["run"],
                    row["kind"],
                    row["data"],
                    row["target"] or "",
                    row["best_model"] or "",
                    score,
                    rows,
                )
            )
        )
    console.print(table)


def _prune(
    runs_dir: str, keep: int | None, older_than: float | None, kind: str | None, yes: bool
) -> None:
    import shutil

    from plainml.runs import runs_to_prune

    doomed = runs_to_prune(runs_dir, keep=keep, older_than=older_than, kind=kind)
    if not doomed:
        info("Nothing to delete.")
        return
    size = sum(f.stat().st_size for p in doomed for f in p.rglob("*") if f.is_file())
    heading(f"{len(doomed)} run(s) to delete ({size / 1e6:,.1f} MB)")
    for path in doomed[:15]:
        console.print(f"  {esc(path.name)}")
    if len(doomed) > 15:
        console.print(f"  [muted]… and {len(doomed) - 15} more[/]")
    if not yes and not click.confirm("Delete them? This can't be undone", default=False):
        info("Nothing deleted.")
        return
    for path in doomed:
        shutil.rmtree(path)
    success(f"Deleted {len(doomed)} run(s).")


@cli.command(short_help="Compare runs side by side.")
@click.argument("run_refs", nargs=-1)
@click.option("--runs-dir", default="runs", show_default=True)
def compare(run_refs: tuple[str, ...], runs_dir: str) -> None:
    """Compare RUNs (default: the last five): data, target, best model and scores.

    \b
    Examples:
      plainml compare
      plainml compare runs/*churn*
    """
    from rich.table import Table

    from plainml.console import fmt_num
    from plainml.runs import list_runs, load_run

    if run_refs:
        refs = list(run_refs)
    else:
        existing = list_runs(runs_dir)
        refs = list(existing["run"].tail(5)) if not existing.empty else []
    if not refs:
        info("No runs to compare yet.")
        return
    loaded = [load_run(ref, runs_dir) for ref in refs]
    table = Table(header_style="muted", box=None, pad_edge=False)
    table.add_column("")
    for run in loaded:
        table.add_column(esc(run.name), overflow="fold")

    def row(label: str, getter: Callable[[Any], Any]) -> None:
        values = []
        for run in loaded:
            try:
                values.append(esc(getter(run.info)))
            except (KeyError, TypeError):
                values.append("—")
        table.add_row(label, *values)

    row("Kind", lambda i: i.get("kind", "train"))
    row("Data", lambda i: Path(str(i["data"]["source"])).name)
    row("Rows", lambda i: f"{i['data']['rows']:,}")
    row("Target", lambda i: i.get("target"))
    row("Task", lambda i: i.get("task"))
    row("Best model", lambda i: i["best"]["name"])
    row("Metric", lambda i: i.get("metric_label", i.get("metric")))
    row(
        "CV score",
        lambda i: f"{fmt_num(i['best']['cv_score'])} ± {fmt_num(i['best'].get('cv_std'))}",
    )
    row("Test score", lambda i: fmt_num(i["best"]["holdout"].get(i["metric"])))
    row("Data fingerprint", lambda i: i["data"].get("fingerprint", "—"))
    row("Time", lambda i: f"{i['timings']['total_seconds']:.0f}s")
    console.print(table)
    fingerprints = {r.info.get("data", {}).get("fingerprint") for r in loaded}
    if (
        len(fingerprints) > 1
        and len(
            {
                r.info.get("target")
                if isinstance(r.info.get("target"), str)
                else str(r.info.get("target"))
                for r in loaded
            }
        )
        == 1
    ):
        note(
            "These runs used different data (fingerprints differ), so scores aren't directly comparable."
        )


@cli.command(name="models", short_help="List the models plainml can train.")
@click.option(
    "--task", type=click.Choice(["classification", "regression"]), help="Only models for this task."
)
def models_command(task: str | None) -> None:
    """Show every model, which tasks it supports, and whether it's installed."""
    from rich.table import Table

    from plainml.registry import MODELS

    table = Table(header_style="muted", box=None, pad_edge=False)
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Model", max_width=30, overflow="fold")
    table.add_column("For", no_wrap=True)
    table.add_column("Runs", no_wrap=True)
    table.add_column("Ready", justify="center", no_wrap=True)
    table.add_column("About", ratio=1, overflow="fold")
    for spec in MODELS:
        tasks = [t for t in ("classification", "regression") if t in spec.names]
        if task and task not in tasks:
            continue
        names = spec.names.get("classification") or spec.names.get("regression")
        if len(set(spec.names.values())) > 1:
            names = " / ".join(dict.fromkeys(spec.names.values()))
        short = (
            "both" if len(tasks) == 2 else ("class." if tasks == ["classification"] else "regr.")
        )
        ready = "[good]✓[/]" if spec.available else f"[warn]✗[/] [muted]needs {spec.requires}[/]"
        runs = (
            "also --quick" if spec.quick else ("always" if spec.tier == "default" else "--thorough")
        )
        table.add_row(spec.key, names, short, runs, ready, spec.description)
    console.print(table)
    note(
        esc(
            "For = classification / regression / both. Runs = when it's tried: always (and in --quick), "
            'or only with --thorough or when named with --models. Missing ones: pip install "plainml[all]"'
        )
    )
    note("Use the keys with --models and --exclude, e.g. --models rf,lightgbm")


@cli.command(short_help="Create a starter config file.")
@click.argument("path", default="plainml.yaml")
@click.option("--data", default="data.csv", help="Data file to put in the config.")
@click.option("-t", "--target", default="target", help="Target column to put in the config.")
def init(path: str, data: str, target: str) -> None:
    """Write a commented YAML config you can edit and run with: plainml train --config PATH"""
    from plainml.config import write_template

    written = write_template(path, data=data, target=target)
    success(f"Created {esc(written)}. Edit it, then run: plainml train --config {esc(written)}")


def main() -> None:
    cli(prog_name="plainml")


if __name__ == "__main__":  # pragma: no cover
    main()
