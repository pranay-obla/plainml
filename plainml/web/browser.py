"""The website's API, answered by Python running inside the visitor's browser.

The static version of plainml's website (``plainml web --export``) has no server: a Web Worker
(``static/worker.js``) starts Pyodide, Python compiled to WebAssembly, installs plainml, and
passes every API request here. The answers come from the same functions the FastAPI server
uses (``plainml.web.server``), so both versions behave alike.

    backend = Backend(root, notify)            # notify(json) hears about running jobs
    backend.request(method, path, body_json)   # -> '{"status": 200, "body": ...}'
    backend.upload(filename, data)             # the same, for an uploaded file's bytes
    backend.read(path)                         # -> (bytes, media type), for downloads
    backend.run_pending()                      # runs queued jobs, one after another

Browsers don't give Python threads, so jobs run one at a time and parallel steps inside
scikit-learn and joblib run sequentially.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from plainml.errors import PlainMLError
from plainml.web.jobs import Job, run_job
from plainml.web.server import (
    TASK_TITLES,
    Workspace,
    build_work,
    finish_upload,
    info_payload,
    inside,
    media_type,
    models_payload,
    preview_payload,
    run_summary,
    runs_payload,
    sample_upload,
    start_upload,
    upload_payload,
)

MAX_UPLOAD_MB = 200  # the whole file sits in the browser's memory


class ApiError(Exception):
    def __init__(self, status: int, message: str, hint: str | None = None):
        super().__init__(message)
        self.status, self.message, self.hint = status, message, hint


@contextmanager
def one_thread() -> Iterator[None]:
    """Run joblib's (and so scikit-learn's) parallel steps one after another."""
    try:
        from joblib import parallel_config
    except ImportError:  # pragma: no cover - joblib older than 1.3
        yield
        return
    with parallel_config(backend="sequential"):
        yield


class Backend:
    """plainml's web API for one browser, with runs kept under ``root``."""

    def __init__(self, root: str | Path, notify: Callable[[str], None] | None = None):
        self.ws = Workspace(root)
        self.notify = notify
        self.jobs: dict[str, Job] = {}
        self.pending: list[tuple[Job, Callable[[Job], dict[str, Any]]]] = []

    # --- requests ------------------------------------------------------------------------------

    def request(self, method: str, path: str, body: str | None = None) -> str:
        try:
            payload = json.loads(body) if body else {}
            status, answer = 200, self._route(method.upper(), path, payload)
        except ApiError as exc:
            status, answer = exc.status, {"detail": {"message": exc.message, "hint": exc.hint}}
        except PlainMLError as exc:
            status, answer = 422, {"detail": {"message": exc.message, "hint": exc.hint}}
        return json.dumps({"status": status, "body": answer}, default=str)

    def _route(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        parts = urlsplit(path)
        route = unquote(parts.path)
        query = parse_qs(parts.query)
        if method == "GET" and route == "/api/info":
            return info_payload(self.ws, mode="browser", max_upload_mb=MAX_UPLOAD_MB)
        if method == "GET" and route == "/api/runs":
            return runs_payload(self.ws)
        if method == "GET" and route == "/api/models":
            return models_payload(self.ws)
        if method == "POST" and route == "/api/jobs":
            return self._start_job(payload)
        if method == "POST" and (m := re.fullmatch(r"/api/samples/([^/]+)", route)):
            key = m.group(1)
            return self._found(lambda: sample_upload(self.ws, key))
        if method == "GET" and (m := re.fullmatch(r"/api/uploads/([^/]+)", route)):
            upload_id = m.group(1)
            return self._found(lambda: upload_payload(self.ws, upload_id))
        if method == "GET" and (m := re.fullmatch(r"/api/jobs/([^/]+)", route)):
            job = self.jobs.get(m.group(1))
            if job is None:
                raise ApiError(404, "No such job (the page may have been reloaded).")
            return job.to_dict(max(0, int((query.get("log_from") or ["0"])[0])))
        if method == "GET" and (m := re.fullmatch(r"/api/runs/([^/]+)", route)):
            name = m.group(1)
            return self._found(lambda: run_summary(self.ws, name))
        if method == "GET" and (m := re.fullmatch(r"/api/(runs|jobs)/([^/]+)/preview/(.+)", route)):
            kind, owner, relative = m.groups()
            return self._found(lambda: preview_payload(self._file(kind, owner, relative)))
        raise ApiError(404, f"Not found: {method} {route}")

    @staticmethod
    def _found(answer: Callable[[], Any]) -> Any:
        try:
            return answer()
        except PlainMLError as exc:
            raise ApiError(404, exc.message, exc.hint) from exc

    def _file(self, kind: str, owner: str, relative: str) -> Path:
        if kind == "runs":
            return inside(self.ws.run_dir(owner), relative)
        if not re.fullmatch(r"[0-9a-f]{12}", owner):
            raise PlainMLError("No such job.")
        return inside(self.ws.outputs / owner, relative)

    def _start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task", ""))
        work = build_work(self.ws, task, payload.get("upload"), payload.get("options") or {})
        job = Job(id=uuid.uuid4().hex[:12], task=task, title=TASK_TITLES.get(task, task))
        self.jobs[job.id] = job
        self.pending.append((job, work))
        return job.to_dict()

    # --- files ---------------------------------------------------------------------------------

    def upload(self, filename: str, data: bytes | bytearray | memoryview) -> str:
        """Save an uploaded file's bytes and summarise it (the same answer as POST /api/uploads)."""
        if hasattr(data, "to_bytes"):  # a JavaScript Uint8Array, passed in by the worker
            data = data.to_bytes()
        try:
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise ApiError(
                    413,
                    f"The file is over {MAX_UPLOAD_MB} MB, too big to work on in a browser.",
                    'Install plainml and run it on your computer instead: pip install "plainml[web]"',
                )
            try:
                upload_id, path = start_upload(self.ws, filename)
            except PlainMLError as exc:
                raise ApiError(415, exc.message, exc.hint) from exc
            path.write_bytes(bytes(data))
            status, answer = 200, finish_upload(upload_id, path)
        except ApiError as exc:
            status, answer = exc.status, {"detail": {"message": exc.message, "hint": exc.hint}}
        except PlainMLError as exc:
            status, answer = 422, {"detail": {"message": exc.message, "hint": exc.hint}}
        return json.dumps({"status": status, "body": answer}, default=str)

    def read(self, path: str) -> tuple[bytes, str]:
        """A run's or job's file, for downloading or showing: (content, media type)."""
        route = unquote(urlsplit(path).path)
        m = re.fullmatch(r"/api/(runs|jobs)/([^/]+)/files/(.+)", route)
        if m is None:
            raise PlainMLError(f"Not a file: {route}")
        file = self._file(m.group(1), m.group(2), m.group(3))
        return file.read_bytes(), media_type(file) or "application/octet-stream"

    # --- jobs ----------------------------------------------------------------------------------

    def run_pending(self) -> bool:
        """Run every queued job in order. Returns whether anything ran."""
        ran = False
        while self.pending:
            job, work = self.pending.pop(0)
            job.listener = self._tell
            with one_thread():
                run_job(job, work)
            ran = True
        return ran

    def _tell(self, job: Job) -> None:
        if self.notify is not None:
            self.notify(json.dumps(job.to_dict(), default=str))
