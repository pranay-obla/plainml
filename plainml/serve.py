"""``plainml serve``: a prediction API for a trained model (FastAPI + uvicorn).

Endpoints:
    GET  /          what the model is, the columns it expects, and an example request
    GET  /health    liveness check
    POST /predict   {"rows": [{...}, ...]}  (a single object or a bare list also works)
    POST /forecast  {"horizon": 30}          (forecast models only; add "groups": [...]
                                              to pick series from a grouped model)
    GET  /docs      interactive documentation

With an API key (``--api-key`` or the ``PLAINML_API_KEY`` environment variable), every
endpoint except /health and /docs needs the header ``X-API-Key: <key>``.
"""

from __future__ import annotations

import math
import os
import secrets
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plainml import __version__
from plainml.console import esc, info, note
from plainml.errors import PlainMLError, require
from plainml.runs import DEFAULT_RUNS_DIR
from plainml.tasks import FORECAST


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): _plain(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def _rows_from(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if isinstance(payload, dict):
        payload = [payload]
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(r, dict) for r in payload)
    ):
        raise ValueError('Send {"rows": [{"column": value, ...}, ...]}')
    return payload


API_KEY_ENV = "PLAINML_API_KEY"
API_KEY_HEADER = "X-API-Key"


def create_app(
    model: Any = "latest", out_dir: str | Path = DEFAULT_RUNS_DIR, api_key: str | None = None
) -> Any:
    """Build the FastAPI app for ``model`` (a path, run, 'latest' or a loaded model).

    ``api_key`` (default: the PLAINML_API_KEY environment variable) makes every endpoint
    except /health and /docs require the header ``X-API-Key``.
    """
    fastapi = require("fastapi", "Serving models")
    from fastapi import Body, Depends, HTTPException, Security
    from fastapi.security import APIKeyHeader

    api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV) or None
    header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

    def check_key(sent: str | None = Security(header)) -> None:
        if api_key is None:
            return
        if not sent or not secrets.compare_digest(sent.encode(), api_key.encode()):
            raise HTTPException(401, f"Missing or wrong API key: send the {API_KEY_HEADER} header.")

    guarded = [Depends(check_key)]

    from plainml.console import quiet
    from plainml.predicting import load_model, predict

    fitted, meta = load_model(model, out_dir=out_dir)
    task = meta.get("task")
    targets = meta.get("targets") or []
    schema = meta.get("schema") or {}
    columns = [
        {"name": column, "type": kind}
        for kind in ("numeric", "categorical", "datetime", "text")
        for column in schema.get(kind, [])
    ]
    example = meta.get("example") or {}
    app = fastapi.FastAPI(
        title=f"plainml model: {meta.get('model', 'model')}",
        version=__version__,
        description=(
            f"Predicts **{', '.join(targets) or task}** ({task}). "
            f"Send rows as JSON to `POST /predict`. Missing columns are treated as blank."
        ),
    )

    @app.get("/", dependencies=guarded)
    def describe() -> dict[str, Any]:
        return {
            "model": meta.get("model"),
            "task": task,
            "target": targets,
            "classes": meta.get("classes"),
            "features": columns or meta.get("features"),
            "trained": meta.get("created"),
            "test_scores": meta.get("holdout_scores"),
            "plainml_version": meta.get("plainml_version"),
            "example_request": {"rows": [example]} if example else None,
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    if task == FORECAST:

        @app.post("/forecast", dependencies=guarded)
        def forecast(
            payload: dict[str, Any] = Body(default={"horizon": meta.get("horizon", 30)}),
        ) -> dict[str, Any]:
            horizon = int(payload.get("horizon") or meta.get("horizon", 30))
            if not 1 <= horizon <= 10_000:
                raise HTTPException(422, "horizon must be between 1 and 10000")
            frame = fitted.forecast(horizon)
            group = meta.get("group")
            wanted = payload.get("groups")
            if group and wanted:  # grouped models: optionally just some of the series
                wanted = [str(g) for g in ([wanted] if isinstance(wanted, str) else wanted)]
                unknown = sorted(set(wanted) - set(frame[group].astype(str)))
                if unknown:
                    raise HTTPException(422, f"Unknown {group}: {', '.join(unknown[:10])}")
                frame = frame[frame[group].astype(str).isin(wanted)]
            return {"forecast": _records(frame)}

    else:

        @app.post("/predict", dependencies=guarded)
        def predict_rows(
            payload: Any = Body(..., examples=[{"rows": [example]}] if example else None),
        ) -> dict[str, Any]:
            try:
                rows = _rows_from(payload)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if len(rows) > 100_000:
                raise HTTPException(413, "Send at most 100,000 rows per request.")
            frame = pd.DataFrame(rows)
            known = set(meta.get("features") or [])
            if known and not known & set(frame.columns):
                raise HTTPException(
                    422,
                    f"None of the model's columns were sent. Expected some of: {sorted(known)[:20]}",
                )
            try:
                with quiet(True):
                    result = predict(fitted, frame, include_input=False, proba=True)
            except PlainMLError as exc:
                raise HTTPException(422, exc.message) from exc
            return {
                "predictions": _records(result),
                "model": meta.get("model"),
                "missing_columns": sorted(known - set(frame.columns)),
            }

    return app


def serve(
    model: Any = "latest",
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    api_key: str | None = None,
) -> None:
    uvicorn = require("uvicorn", "Serving models")
    app = create_app(model, out_dir, api_key)
    protected = bool(api_key or os.environ.get(API_KEY_ENV))
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    info(
        f"Serving {esc(app.title)} on http://{shown}:{port}  (docs: http://{shown}:{port}/docs, Ctrl-C to stop)"
    )
    if protected:
        note(f"Requests need the header {API_KEY_HEADER} (except /health and /docs).")
    if host == "0.0.0.0" and not protected:
        note(
            "Listening on all network interfaces: anyone who can reach this machine can call the "
            f"API. Add --api-key (or set {API_KEY_ENV}) to require a key."
        )
    uvicorn.run(app, host=host, port=port, log_level="info")
