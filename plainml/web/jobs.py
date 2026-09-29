"""Background jobs for the web app: one worker runs tasks in order and records their progress.

Tasks run one at a time. plainml's console is shared by every module, so while a job runs
its output is redirected into that job's log, and ``console.status`` messages ("Testing
Random forest…") become the job's current stage. Training and importance also report a
progress fraction.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text

from plainml.console import console, set_quiet
from plainml.errors import PlainMLError

MAX_LOG_LINES = 400
MAX_JOBS_KEPT = 200


@dataclass
class Job:
    id: str
    task: str
    title: str
    status: str = "queued"  # queued, running, done, failed
    stage: str = "Waiting for the previous job to finish"
    progress: float | None = None
    log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def to_dict(self, log_from: int = 0) -> dict[str, Any]:
        now = time.time()
        return {
            "id": self.id,
            "task": self.task,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "log": self.log[log_from:],
            "log_size": len(self.log),
            "result": self.result,
            "error": self.error,
            "elapsed": round((self.finished or now) - (self.started or now), 1),
        }


class _LogWriter(io.TextIOBase):
    """A file-like sink that turns console output into log lines for one job."""

    def __init__(self, job: Job):
        self.job = job
        self.partial = ""

    def write(self, text: str) -> int:
        self.partial += text
        *lines, self.partial = self.partial.split("\n")
        for line in lines:
            self._add(line)
        return len(text)

    def flush(self) -> None:
        if self.partial:
            self._add(self.partial)
            self.partial = ""

    def _add(self, line: str) -> None:
        line = line.rstrip()
        if not line and (not self.job.log or not self.job.log[-1]):
            return  # no runs of blank lines
        self.job.log.append(line)
        if len(self.job.log) > MAX_LOG_LINES:
            del self.job.log[: len(self.job.log) - MAX_LOG_LINES]

    def isatty(self) -> bool:
        return False


class _Status:
    """Stands in for rich's spinner: records the message as the job's stage."""

    def __init__(self, job: Job, message: Any):
        self.job = job
        self.update(message)

    def update(self, message: Any = None, **_: Any) -> None:
        if message is None:
            return
        plain = Text.from_markup(str(message)).plain.strip()
        if plain and plain != self.job.stage:
            self.job.stage = plain
            self.job.log.append(f"▸ {plain}")

    def __enter__(self) -> _Status:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class JobRunner:
    """Queues jobs and runs them one after another on a background thread."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[tuple[Job, Callable[[Job], dict[str, Any]]]] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, name="plainml-jobs", daemon=True)
        self._thread.start()

    def submit(self, task: str, title: str, work: Callable[[Job], dict[str, Any]]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], task=task, title=title)
        with self._lock:
            self.jobs[job.id] = job
            if len(self.jobs) > MAX_JOBS_KEPT:
                finished = [j for j in self.jobs.values() if j.status in ("done", "failed")]
                for old in sorted(finished, key=lambda j: j.created)[
                    : len(self.jobs) - MAX_JOBS_KEPT
                ]:
                    self.jobs.pop(old.id, None)
        self._queue.put((job, work))
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _work(self) -> None:
        while True:
            job, work = self._queue.get()
            try:
                self._run(job, work)
            finally:
                self._queue.task_done()

    def _run(self, job: Job, work: Callable[[Job], dict[str, Any]]) -> None:
        job.status, job.started, job.stage = "running", time.time(), "Starting"
        writer = _LogWriter(job)
        # rich keeps "not set" as None (then follows the terminal), so save the raw values
        saved_file, saved_width = console._file, console._width
        console.file = writer  # type: ignore[assignment]
        console.width = 100
        console.status = lambda message, **_: _Status(job, message)  # type: ignore[method-assign,assignment]
        set_quiet(False)
        try:
            job.result = work(job)
            job.status, job.stage, job.progress = "done", "Done", 1.0
        except PlainMLError as exc:
            job.status = "failed"
            job.error = {"message": exc.message, "hint": exc.hint}
        except Exception as exc:  # show the error in the browser instead of killing the worker
            job.status = "failed"
            job.error = {
                "message": f"Unexpected error: {type(exc).__name__}: {exc}",
                "hint": "The details are in the log below. Please report it if it looks like a bug.",
            }
            job.log.extend(traceback.format_exc().splitlines()[-12:])
        finally:
            writer.flush()
            del console.status  # back to the real spinner
            console._file, console._width = saved_file, saved_width
            job.finished = time.time()
