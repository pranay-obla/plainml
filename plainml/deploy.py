"""``plainml deploy``: package a trained model as a Docker image that serves it over HTTP.

The folder it writes is self-contained: the model, pinned requirements (the same versions
it was trained with, so the pickle loads cleanly), a small non-root Dockerfile with a
health check, and a README with the build, run and call commands.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib

from plainml import __version__
from plainml.card import CARD_FILE
from plainml.console import esc, heading, info, note, success, warn
from plainml.errors import PlainMLError
from plainml.runs import DEFAULT_RUNS_DIR, RUN_FILE, read_json, resolve_model_path

# import name -> pip name, for libraries a pickled model can need
PIP_NAMES = {
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "imblearn": "imbalanced-learn",
    "torch": "torch",
    "holidays": "holidays",
}
ALWAYS = ("scikit-learn", "pandas", "numpy", "joblib")
NEEDS_OPENMP = {"lightgbm", "xgboost", "catboost"}
_LEAVES = ("numpy", "pandas", "builtins", "scipy")


def model_libraries(model: Any) -> set[str]:
    """Top-level modules of every object inside a fitted model (to know what to install)."""
    found: set[str] = set()
    seen: set[int] = set()

    def walk(obj: Any, depth: int) -> None:
        if id(obj) in seen or depth > 25:
            return
        seen.add(id(obj))
        root = type(obj).__module__.split(".")[0]
        found.add(root)
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for value in obj:
                walk(value, depth + 1)
        elif root not in _LEAVES and hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value, depth + 1)

    walk(model, 0)
    return found


def _version(name: str, saved: dict[str, str]) -> str | None:
    if saved.get(name):
        return saved[name]
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_plainml_source() -> Path | None:
    """The source folder if plainml was installed from a local checkout (not from PyPI)."""
    try:
        raw = importlib.metadata.distribution("plainml").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    url = json.loads(raw).get("url", "")
    if url.startswith("file://") and "dir_info" in raw:
        path = Path(url.removeprefix("file://"))
        return path if (path / "pyproject.toml").is_file() else None
    return None


def _build_wheel(source: Path, into: Path) -> bool:
    """Build plainml's wheel from a clean copy of the checkout.

    Building in place would leave ``build/`` and ``*.egg-info`` folders in the user's source tree.
    """
    with tempfile.TemporaryDirectory(prefix="plainml-wheel-") as scratch:
        copy = Path(scratch) / "src"
        copy.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            if (source / name).is_file():
                shutil.copy2(source / name, copy / name)
        shutil.copytree(
            source / "plainml",
            copy / "plainml",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        built = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(into), str(copy)],
            capture_output=True,
            text=True,
        )
    return built.returncode == 0


def requirements(model: Any, saved: dict[str, str]) -> tuple[list[str], set[str]]:
    """Pinned requirement lines for serving ``model``, and the pip names they cover."""
    libraries = model_libraries(model)
    names = list(ALWAYS) + sorted({PIP_NAMES[m] for m in libraries if m in PIP_NAMES} - set(ALWAYS))
    lines = [f"plainml[serve]=={__version__}"]
    for name in names:
        version = _version(name, saved)
        lines.append(f"{name}=={version}" if version else name)
    return lines, set(names)


def _dockerfile(python: str, needs_openmp: bool, port: int) -> str:
    system = (
        "# LightGBM/XGBoost/CatBoost need the OpenMP runtime\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n\n"
        if needs_openmp
        else ""
    )
    return f"""# Built by `plainml deploy`. Build:  docker build -t my-model .
FROM python:{python}-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
WORKDIR /app

{system}COPY requirements.txt ./
COPY wheels/ ./wheels/
RUN pip install --no-cache-dir -r requirements.txt

COPY model.joblib ./

# Don't run as root inside the container
RUN useradd --create-home --uid 10001 plainml
USER plainml

EXPOSE {port}
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health')"

# Set PLAINML_API_KEY at `docker run` time to require an X-API-Key header on requests
CMD ["plainml", "serve", "model.joblib", "--host", "0.0.0.0", "--port", "{port}"]
"""


def _readme(name: str, meta: dict[str, Any], port: int, image: str, card: bool) -> str:
    forecast = meta.get("task") == "forecast"
    example = meta.get("example") or {}
    if forecast:
        call = (
            f"curl -X POST localhost:{port}/forecast -H 'X-API-Key: change-me' \\\n"
            "     -H 'Content-Type: application/json' -d '{\"horizon\": 14}'"
        )
    else:
        payload = json.dumps({"rows": [example or {"column": "value"}]}, default=str)
        call = (
            f"curl -X POST localhost:{port}/predict -H 'X-API-Key: change-me' \\\n"
            f"     -H 'Content-Type: application/json' -d '{payload}'"
        )
    target = ", ".join(meta.get("targets") or []) or meta.get("task", "")
    card_line = (
        f"- `{CARD_FILE}`: what the model does, how well, and its limitations\n" if card else ""
    )
    return f"""# {name}: {meta.get("model", "model")} predicting {target}

Made by `plainml deploy`. This folder builds a Docker image that serves the model over HTTP.

## Build and run

```bash
docker build -t {image} .
docker run --rm -p {port}:{port} -e PLAINML_API_KEY=change-me {image}
```

Leave out `-e PLAINML_API_KEY=...` to serve without a key (only on a trusted network).

## Call it

```bash
{call}
```

- `GET /` describes the model and the columns it expects
- `GET /health` is the health check (no key needed)
- `GET /docs` has interactive documentation where you can try requests

## Files

- `model.joblib`: the trained model
- `requirements.txt`: the exact library versions it was trained with; keep them, since
  pickled models don't reliably load under other versions
- `Dockerfile`: a slim, non-root image with a health check
{card_line}
## Deploying

The image runs anywhere that runs containers: Google Cloud Run, AWS App Runner or ECS,
Azure Container Apps, Fly.io, Render, or a Kubernetes cluster. Pass the API key as a
secret environment variable, not in the image.
"""


def deploy(
    model: Any = "latest",
    *,
    output: str | Path | None = None,
    out_dir: str | Path = DEFAULT_RUNS_DIR,
    port: int = 8000,
    python: str | None = None,
    verbose: bool = True,
) -> Path:
    """Write a Docker build folder for ``model`` (a run, 'latest' or a .joblib path)."""
    model_path = resolve_model_path(model, out_dir)
    run_dir = model_path.parent
    try:
        fitted = joblib.load(model_path)
    except Exception as exc:
        raise PlainMLError(f"Couldn't load {model_path}: {exc}") from exc
    meta = getattr(fitted, "plainml_meta_", {}) or {}
    run_info = read_json(run_dir / RUN_FILE) if (run_dir / RUN_FILE).is_file() else {}
    saved = meta.get("packages") or (run_info.get("environment") or {}).get("packages") or {}
    trained_python = (run_info.get("environment") or {}).get("python") or ".".join(
        map(str, sys.version_info[:2])
    )
    python = python or ".".join(trained_python.split(".")[:2])
    run_name = meta.get("run") or run_dir.name
    target = Path(output) if output else Path("deploy") / run_name
    if target.exists() and any(target.iterdir()):
        raise PlainMLError(
            f"'{target}' already exists and isn't empty.",
            hint="Pass another folder with -o, or remove this one first.",
        )
    target.mkdir(parents=True, exist_ok=True)
    wheels = target / "wheels"
    wheels.mkdir()

    lines, covered = requirements(fitted, saved)
    header = ["# Pinned to the versions the model was trained with.", "--find-links ./wheels"]
    if "torch" in covered:
        header.append("--extra-index-url https://download.pytorch.org/whl/cpu  # CPU-only torch")
    source = _local_plainml_source()
    # installed from a checkout: ship a wheel so the image build doesn't need plainml on PyPI
    if source is not None and not _build_wheel(source, wheels):
        warn(
            "Couldn't build a plainml wheel from your checkout; the image will install "
            f"plainml {__version__} from PyPI instead."
        )
    (wheels / ".keep").write_text("", encoding="utf-8")
    (target / "requirements.txt").write_text("\n".join([*header, *lines]) + "\n", encoding="utf-8")
    shutil.copy2(model_path, target / "model.joblib")
    if (run_dir / CARD_FILE).is_file():
        shutil.copy2(run_dir / CARD_FILE, target / CARD_FILE)
    image = run_name.partition("_")[2].lower().replace("_", "-") or "plainml-model"
    (target / "Dockerfile").write_text(
        _dockerfile(python, bool(covered & NEEDS_OPENMP), port), encoding="utf-8"
    )
    (target / ".dockerignore").write_text("*.md\n__pycache__/\n", encoding="utf-8")
    (target / "README.md").write_text(
        _readme(run_name, meta, port, image, (target / CARD_FILE).is_file()), encoding="utf-8"
    )

    if verbose:
        heading(f"Deployment folder for {esc(meta.get('model', 'the model'))}")
        success(f"Wrote {esc(target)}")
        for name, what in (
            ("Dockerfile", f"python:{python}-slim, non-root, health check on /health"),
            ("requirements.txt", ", ".join(lines[1:5]) + (" …" if len(lines) > 5 else "")),
            ("model.joblib", "the trained model"),
            ("README.md", "how to build, run and call it"),
        ):
            info(f"{name:<18}[muted]{esc(what)}[/]")
        note("Build and run it:")
        info(f"docker build -t {image} {esc(target)}")
        info(f"docker run --rm -p {port}:{port} -e PLAINML_API_KEY=change-me {image}")
    return target
