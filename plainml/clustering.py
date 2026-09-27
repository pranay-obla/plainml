"""``plainml cluster``: find natural groups in data that has no target column.

Several algorithms are tried over a range of group counts and scored by how cleanly the
groups separate (silhouette). Each group is then described in plain English by what
makes it different from the rest ("higher income, mostly on the pro plan").
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.base import BaseEstimator
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from plainml import __version__
from plainml.console import (
    console,
    esc,
    fmt_num,
    fmt_pct,
    heading,
    info,
    note,
    quiet,
    warn,
)
from plainml.errors import PlainMLError, did_you_mean, find_column, is_installed
from plainml.io import describe_source, fingerprint, load_data, save_table, source_stem
from plainml.preprocessing import build_preprocessor
from plainml.profiling import build_profile, redact_columns, shown_source
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
from plainml.schema import Schema, coerce_numeric, infer_schema
from plainml.tasks import CLUSTERING

ALGORITHMS = {
    "kmeans": "K-Means",
    "agglomerative": "Agglomerative (Ward)",
    "gmm": "Gaussian Mixture",
    "hdbscan": "HDBSCAN (finds the count itself)",
    "kmedoids": "K-Medoids",
}
MAX_ROWS = {"agglomerative": 15_000, "kmedoids": 5_000}
SILHOUETTE_SAMPLE = 5_000
ASSIGNMENTS_FILE = "clustered.csv"


class ClusterModel(BaseEstimator):
    """Assigns new rows to the discovered groups (numbered largest first)."""

    def __init__(self, pipeline: Any = None, mapping: dict[int, int] | None = None):
        self.pipeline = pipeline
        self.mapping = mapping

    def predict(self, X: Any) -> np.ndarray:
        raw = np.asarray(self.pipeline.predict(X)).astype(int)
        mapping = self.mapping or {}
        return np.array([mapping.get(int(v), int(v)) for v in raw])


@dataclass
class ClusterResult:
    run_dir: Path
    algorithm: str
    k: int
    labels: pd.Series
    leaderboard: pd.DataFrame
    descriptions: list[str]
    model: ClusterModel

    @property
    def report_path(self) -> Path:
        return self.run_dir / REPORT_FILE

    def __repr__(self) -> str:
        return f"ClusterResult({self.k} groups via {self.algorithm}, run_dir='{self.run_dir}')"


def _k_values(k: Any, n_rows: int) -> list[int]:
    upper = max(2, min(10, n_rows - 1))
    if k is None or str(k).strip().lower() == "auto":
        return list(range(2, upper + 1))
    text = str(k).strip()
    try:
        if "-" in text:
            low, high = (int(v) for v in text.split("-", 1))
            values = list(range(max(2, low), min(high, n_rows - 1) + 1))
        else:
            values = [int(text)]
    except ValueError:
        raise PlainMLError(
            f"Couldn't understand -k '{k}'.", hint="Use a number (4), a range (2-8) or auto."
        ) from None
    values = [v for v in values if 2 <= v < n_rows]
    if not values:
        raise PlainMLError(
            "The number of groups must be at least 2 and less than the number of rows."
        )
    return values


def _estimator(algorithm: str, k: int, seed: int, n_rows: int, n_features: int) -> Any:
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward")
    if algorithm == "gmm":
        covariance = "full" if n_features <= 20 else "diag"
        return GaussianMixture(
            n_components=k, covariance_type=covariance, n_init=2, random_state=seed
        )
    if algorithm == "hdbscan":
        return HDBSCAN(min_cluster_size=int(min(500, max(5, n_rows // 50))))
    if algorithm == "kmedoids":
        from sklearn_extra.cluster import KMedoids

        return KMedoids(n_clusters=k, random_state=seed, init="k-medoids++")
    raise AssertionError(algorithm)


def _choose_algorithms(
    algorithms: list[str] | None, n_rows: int
) -> tuple[list[str], dict[str, str]]:
    skipped: dict[str, str] = {}
    if algorithms:
        chosen = []
        for raw in algorithms:
            key = raw.strip().lower()
            if key not in ALGORITHMS:
                raise PlainMLError(
                    f"Unknown algorithm '{raw}'.{did_you_mean(key, ALGORITHMS)}",
                    hint="Choose from: " + ", ".join(ALGORITHMS),
                )
            chosen.append(key)
    else:
        chosen = ["kmeans", "agglomerative", "gmm", "hdbscan", "kmedoids"]
    final = []
    for key in chosen:
        if key == "kmedoids" and not is_installed("sklearn_extra"):
            if algorithms:
                skipped[key] = 'needs scikit-learn-extra (pip install "plainml[cluster]")'
            continue
        if key in MAX_ROWS and n_rows > MAX_ROWS[key]:
            skipped[key] = f"too slow for {n_rows:,} rows"
            continue
        final.append(key)
    if not final:
        raise PlainMLError(
            "No clustering algorithm left to run.",
            hint="; ".join(f"{k}: {v}" for k, v in skipped.items()),
        )
    return final, skipped


def _describe(
    df: pd.DataFrame, labels: np.ndarray, schema: Schema
) -> tuple[list[dict[str, Any]], list[str]]:
    """What sets each group apart: numeric columns in standard deviations, categories by lift."""
    profiles, sentences = [], []
    numeric = {c: coerce_numeric(df[c]) for c in schema.numeric}
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        share = float(mask.mean())
        traits: list[dict[str, Any]] = []
        for column, values in numeric.items():
            std = values.std()
            if not std or np.isnan(std):
                continue
            z = float((values[mask].mean() - values.mean()) / std)
            traits.append(
                {
                    "column": column,
                    "kind": "numeric",
                    "z": z,
                    "mean": float(values[mask].mean()),
                    "overall": float(values.mean()),
                    "strength": abs(z),
                }
            )
        for column in schema.categorical:
            values = df[column].astype(str)
            inside = values[mask].value_counts(normalize=True)
            if inside.empty:
                continue
            top, top_share = inside.index[0], float(inside.iloc[0])
            overall = float((values == top).mean())
            lift = top_share / overall if overall else 0.0
            if top_share >= 0.3 and lift >= 1.4:
                traits.append(
                    {
                        "column": column,
                        "kind": "categorical",
                        "value": top,
                        "share": top_share,
                        "overall": overall,
                        "strength": min(lift - 1, 3.0),
                    }
                )
        traits.sort(key=lambda t: -float(t["strength"]))
        top_traits = [t for t in traits if t["strength"] >= 0.3][:3]
        parts = []
        for trait in top_traits:
            if trait["kind"] == "numeric":
                word = "higher" if trait["z"] > 0 else "lower"
                parts.append(
                    f"{word} {trait['column']} ({'+' if trait['z'] > 0 else '−'}{abs(trait['z']):.1f} sd)"
                )
            else:
                parts.append(
                    f"mostly {trait['column']} = {trait['value']} ({fmt_pct(trait['share'], 0)} vs {fmt_pct(trait['overall'], 0)} overall)"
                )
        label = "Noise (fits no group)" if cluster == -1 else f"Group {cluster}"
        text = f"{label} · {fmt_pct(share, 0)} of rows: " + (
            ", ".join(parts) if parts else "close to average on everything"
        )
        sentences.append(text)
        profiles.append(
            {
                "cluster": int(cluster),
                "label": label,
                "size": int(mask.sum()),
                "share": share,
                "traits": top_traits,
                "summary": text,
            }
        )
    return profiles, sentences


def cluster(
    data: Any,
    *,
    k: Any = "auto",
    algorithms: list[str] | None = None,
    drop: list[str] | None = None,
    output: str | Path | None = None,
    seed: int = 42,
    out_dir: str | Path = "runs",
    name: str | None = None,
    report: bool = True,
    private: bool = False,
    verbose: bool = True,
    **load_options: Any,
) -> ClusterResult:
    """Group similar rows. Returns the groups, scores for every algorithm tried, and descriptions."""
    started = time.time()
    with quiet(not verbose):
        df = load_data(data, seed=seed, **load_options)
        drop_columns = [find_column(c, df.columns) for c in drop or []]
        schema, infos = infer_schema(df, drop=drop_columns)
        if not schema.features:
            raise PlainMLError("No usable columns to cluster on.")
        source = describe_source(data)
        prof = build_profile(df, schema, infos, source)
        n_rows = len(df)
        if n_rows < 10:
            raise PlainMLError("Clustering needs at least 10 rows.")
        chosen, skipped = _choose_algorithms(algorithms, n_rows)
        ks = _k_values(k, n_rows)
        preprocess = build_preprocessor(schema)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = np.asarray(preprocess.fit_transform(df), dtype=float)
        heading(f"Clustering {n_rows:,} rows on {len(schema.features)} columns")
        for key, reason in skipped.items():
            note(f"Skipping {key}: {esc(reason)}")

        rows: list[dict[str, Any]] = []
        results: dict[tuple[str, int], np.ndarray] = {}
        with console.status("Trying algorithms…") as status:
            for algorithm in chosen:
                for count in [0] if algorithm == "hdbscan" else ks:
                    status.update(
                        f"Trying {ALGORITHMS[algorithm]}"
                        + (f" with {count} groups" if count else "")
                    )
                    t0 = time.perf_counter()
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            labels = np.asarray(
                                _estimator(algorithm, count, seed, n_rows, X.shape[1]).fit_predict(
                                    X
                                )
                            )
                    except Exception as exc:
                        note(
                            f"{ALGORITHMS[algorithm]} failed: {esc(str(exc).splitlines()[0][:100])}"
                        )
                        break
                    valid = labels != -1
                    found = len(set(labels[valid]))
                    if found < 2 or valid.sum() < 3:
                        continue
                    sample = min(SILHOUETTE_SAMPLE, int(valid.sum()))
                    silhouette = float(
                        silhouette_score(
                            X[valid], labels[valid], sample_size=sample, random_state=seed
                        )
                    )
                    sizes = (
                        np.bincount(labels[valid]) if labels[valid].min() >= 0 else np.array([1])
                    )
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "model": ALGORITHMS[algorithm].split(" (")[0],
                            "k": found,
                            "silhouette": silhouette,
                            "calinski_harabasz": float(
                                calinski_harabasz_score(X[valid], labels[valid])
                            ),
                            "davies_bouldin": float(davies_bouldin_score(X[valid], labels[valid])),
                            "smallest_share": float(sizes.min() / valid.sum()),
                            "noise_share": float(1 - valid.mean()),
                            "seconds": round(time.perf_counter() - t0, 2),
                        }
                    )
                    results[(algorithm, found)] = labels
        if not rows:
            raise PlainMLError("No algorithm found more than one group in this data.")
        leaderboard = (
            pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
        )
        best = leaderboard.iloc[0]
        algorithm, count = str(best["algorithm"]), int(best["k"])

        # final model, with groups renumbered largest-first
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if algorithm in ("kmeans", "gmm", "kmedoids"):
                pipeline = Pipeline(
                    [
                        ("preprocess", build_preprocessor(schema)),
                        ("model", _estimator(algorithm, count, seed, n_rows, X.shape[1])),
                    ]
                )
                raw = np.asarray(pipeline.fit(df).predict(df))
            else:
                raw = results[(algorithm, count)]
                pipeline = Pipeline(
                    [
                        ("preprocess", build_preprocessor(schema)),
                        ("model", KNeighborsClassifier(n_neighbors=5)),
                    ]
                ).fit(df, raw)
        counts = pd.Series(raw[raw != -1]).value_counts()
        mapping = {int(old): new for new, old in enumerate(counts.index)}
        mapping[-1] = -1
        model = ClusterModel(pipeline, mapping)
        labels = np.array([mapping[int(v)] for v in raw])
        profiles, sentences = _describe(df, labels, schema)

        run_dir = create_run_dir(out_dir, name or f"{source_stem(data)}-clusters")
        created = datetime.now().isoformat(timespec="seconds")
        model.plainml_meta_ = {
            "plainml_version": __version__,
            "packages": environment()["packages"],
            "created": created,
            "run": run_dir.name,
            "task": CLUSTERING,
            "targets": [],
            "features": schema.features,
            "schema": schema.to_dict(),
            "model": ALGORITHMS[algorithm],
            "k": count,
        }
        joblib.dump(model, run_dir / MODEL_FILE, compress=3)
        assigned = df.copy()
        assigned["cluster"] = labels
        assignments_path = (
            save_table(assigned, output)
            if output
            else save_table(assigned, run_dir / ASSIGNMENTS_FILE)
        )
        if output:
            save_table(assigned, run_dir / ASSIGNMENTS_FILE)
        leaderboard.to_csv(run_dir / LEADERBOARD_FILE, index=False)

        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(n_rows, min(n_rows, 1200), replace=False))
        projection = (
            PCA(n_components=2, random_state=seed).fit_transform(X)
            if X.shape[1] >= 2
            else np.c_[X[:, 0], np.zeros(n_rows)]
        )
        evaluation = {
            "profiles": profiles,
            "sentences": sentences,
            "projection": {
                "x": projection[index, 0].tolist(),
                "y": projection[index, 1].tolist(),
                "labels": labels[index].tolist(),
            },
            "means": _group_means(df, labels, schema),
        }
        write_json(run_dir / EVALUATION_FILE, evaluation)
        write_json(
            run_dir / RUN_FILE,
            {
                "kind": "cluster",
                "plainml_version": __version__,
                "environment": environment(),
                "created": created,
                "task": CLUSTERING,
                "task_label": "clustering",
                "target": None,
                "metric": "silhouette",
                "metric_label": "Silhouette",
                "data": {
                    "source": shown_source(source, private),
                    "rows": n_rows,
                    "raw_rows": n_rows,
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
                "best": {
                    "key": algorithm,
                    "name": ALGORITHMS[algorithm],
                    "k": count,
                    "score": float(best["silhouette"]),
                    "cv_score": float(best["silhouette"]),
                },
                "options": {"k": str(k), "algorithms": chosen, "seed": seed, "drop": drop_columns},
                "skipped": skipped,
                "timings": {"total_seconds": round(time.time() - started, 2)},
                "files": {
                    "model": MODEL_FILE,
                    "assignments": ASSIGNMENTS_FILE,
                    "report": REPORT_FILE if report else None,
                },
            },
        )
        if report:
            try:
                from plainml.report import write_report

                write_report(run_dir)
            except Exception as exc:
                warn(f"Couldn't write the HTML report: {esc(str(exc)[:120])}")

    result = ClusterResult(
        run_dir,
        ALGORITHMS[algorithm],
        count,
        pd.Series(labels, index=df.index, name="cluster"),
        leaderboard,
        sentences,
        model,
    )
    if verbose:
        _render(result, best, assignments_path)
    return result


def _group_means(
    df: pd.DataFrame, labels: np.ndarray, schema: Schema, limit: int = 10
) -> dict[str, Any]:
    numeric = pd.DataFrame({c: coerce_numeric(df[c]) for c in schema.numeric})
    if numeric.empty:
        return {"columns": [], "rows": []}
    grouped = numeric.groupby(labels).mean()
    spread = (
        (grouped - numeric.mean())
        .abs()
        .div(numeric.std().replace(0, np.nan))
        .max()
        .sort_values(ascending=False)
    )
    columns = [c for c in spread.index if not np.isnan(spread[c])][:limit]
    rows: list[list[Any]] = [
        [int(g), *[float(grouped.loc[g, c]) for c in columns]] for g in grouped.index
    ]
    rows.append(["all", *[float(numeric[c].mean()) for c in columns]])
    return {"columns": columns, "rows": rows}


def _quality(score: float) -> str:
    if score >= 0.5:
        return "clearly separated"
    if score >= 0.25:
        return "reasonably separated"
    return "overlapping (the groups blend into each other)"


def _render(result: ClusterResult, best: pd.Series, assignments: Path) -> None:
    heading("Algorithms compared")
    table = Table(box=None, header_style="muted", pad_edge=False)
    headers: tuple[tuple[str, Any], ...] = (
        ("Algorithm", "left"),
        ("Groups", "right"),
        ("Silhouette ↑", "right"),
        ("Smallest group", "right"),
        ("Time", "right"),
    )
    for column, justify in headers:
        table.add_column(column, justify=justify)
    for i, row in result.leaderboard.head(12).iterrows():
        style = "best" if i == 0 else ""
        table.add_row(
            ("★ " if i == 0 else "") + row["model"],
            str(row["k"]),
            fmt_num(row["silhouette"]),
            fmt_pct(row["smallest_share"], 0),
            f"{row['seconds']:.1f}s",
            style=style,
        )
    console.print(table)
    note(
        "Silhouette: how clearly separated the groups are (−1 to 1; above 0.5 is strong, below 0.25 is weak)."
    )
    heading("Result")
    console.print(
        f"[best]★ {result.k} groups found by {result.algorithm}[/]  [muted]silhouette {fmt_num(best['silhouette'])}: {_quality(float(best['silhouette']))}[/]"
    )
    for sentence in result.descriptions:
        console.print(f"• {esc(sentence)}")
    heading("Saved")
    console.print(f"[bold]{esc(result.run_dir)}[/]")
    console.print(f"  {'report.html':<22}[muted]charts and group descriptions[/]")
    console.print(f"  [muted]Rows with their group:[/] {esc(assignments)}")
    info("Assign new rows to these groups with: plainml predict latest NEW_DATA.csv")


def render_cluster_report(run: Any) -> str:
    from plainml.report import (
        as_table_view,
        bar_list,
        card,
        chart,
        grid,
        page,
        profile_section,
        section,
        sentences_list,
        stat_tiles,
        table,
    )

    info_json, evaluation, leaderboard = run.info, run.evaluation, run.leaderboard
    best = info_json["best"]
    charts: dict[str, Any] = {}
    profiles = evaluation.get("profiles", [])
    tiles = [
        ("Silhouette", fmt_num(best["score"]), _quality(best["score"]).split(" (")[0]),
        ("Groups", str(best["k"]), best["name"]),
        ("Rows", f"{info_json['data']['rows']:,}", None),
        (
            "Columns used",
            str(
                sum(
                    len(info_json["schema"].get(k, []))
                    for k in ("numeric", "categorical", "datetime", "text")
                )
            ),
            None,
        ),
    ]
    body = (
        stat_tiles(tiles)
        + f'<div class="summary">{sentences_list(evaluation.get("sentences", []))}</div>'
    )
    body = section("Results", body, anchor="results")

    items = [{"label": p["label"], "value": p["share"]} for p in profiles]
    sizes = bar_list(items, value_format="pct", emphasis=set(range(len(items))))
    projection = evaluation.get("projection", {})
    minis = []
    labels = projection.get("labels", [])
    for p in profiles[:8]:
        chart_id = f"group{p['cluster']}"
        charts[chart_id] = {
            "type": "scatter",
            "x": projection.get("x", []),
            "y": projection.get("y", []),
            "highlight": [label == p["cluster"] for label in labels],
            "labels": [("noise" if v == -1 else f"group {v}") for v in labels],
            "xDomain": _domain(projection.get("x", [])),
            "yDomain": _domain(projection.get("y", [])),
            "xLabel": "component 1",
            "yLabel": "component 2",
            "height": 220,
        }
        minis.append(
            f"<div><h3>{p['label']} · {fmt_pct(p['share'], 0)}</h3>{chart(chart_id, 220)}</div>"
        )
    means = evaluation.get("means", {})
    means_table = (
        table(
            ["Group", *means.get("columns", [])],
            means.get("rows", []),
            numeric=set(range(1, len(means.get("columns", [])) + 1)),
        )
        if means.get("columns")
        else ""
    )
    body += section(
        "The groups",
        grid(
            card(sizes, "Group sizes"),
            card(
                means_table or "<p class='note'>No numeric columns.</p>",
                "Average values per group",
                note="Columns that differ most between groups.",
            ),
            card(
                f'<div class="minis">{"".join(minis)}</div>',
                "Where each group sits",
                note="Every row squeezed onto two axes (PCA); each panel highlights one group in blue. Overlap here doesn't always mean overlap in the full data.",
                wide=True,
            ),
        ),
        anchor="groups",
    )
    board_rows = [
        [
            r["model"],
            r["k"],
            r["silhouette"],
            r["calinski_harabasz"],
            r["davies_bouldin"],
            fmt_pct(r["smallest_share"], 1),
            r["seconds"],
        ]
        for _, r in leaderboard.iterrows()
    ]
    kmeans = leaderboard[leaderboard["algorithm"] == "kmeans"].sort_values("k")
    comparison = [
        card(
            table(
                [
                    "Algorithm",
                    "Groups",
                    "Silhouette ↑",
                    "Calinski-Harabasz ↑",
                    "Davies-Bouldin ↓",
                    "Smallest group",
                    "Seconds",
                ],
                board_rows,
                numeric={1, 2, 3, 4, 6},
                highlight=0,
            ),
            "Every attempt",
            wide=True,
        )
    ]
    if len(kmeans) > 1:
        charts["elbow"] = {
            "type": "line",
            "series": [{"x": kmeans["k"].tolist(), "y": kmeans["silhouette"].tolist()}],
            "xDomain": [int(kmeans["k"].min()), int(kmeans["k"].max())],
            "yDomain": _domain([*kmeans["silhouette"].tolist(), 0.0]),
            "xLabel": "number of groups (K-Means)",
            "yLabel": "silhouette",
        }
        elbow_rows = [[int(r.k), float(r.silhouette)] for r in kmeans.itertuples()]
        comparison.insert(
            0,
            card(
                chart("elbow")
                + as_table_view(table(["Groups", "Silhouette"], elbow_rows, numeric={0, 1})),
                "How many groups?",
                note="Higher is better; a peak suggests a natural number of groups.",
            ),
        )
    body += section("Algorithms compared", grid(*comparison), anchor="algorithms")
    body += section(
        "The data",
        profile_section(info_json.get("profile", {}), info_json.get("schema")),
        anchor="data",
    )
    title = f"Groups in {Path(str(info_json['data']['source'])).name}"
    subtitle = f"clustering · {info_json['data']['rows']:,} rows · {info_json['created'][:16].replace('T', ' ')}"
    return page(
        title,
        subtitle,
        body,
        charts,
        [
            ("results", "Results"),
            ("groups", "Groups"),
            ("algorithms", "Algorithms"),
            ("data", "Data"),
        ],
    )


def _domain(values: list[float]) -> list[float]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return [0.0, 1.0]
    low, high = min(finite), max(finite)
    pad = (high - low) * 0.05 or 1.0
    return [low - pad, high + pad]
