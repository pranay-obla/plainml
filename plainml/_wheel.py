"""Building plainml's own wheel from a source checkout (for `deploy` and `web --export`).

Standard library only, so it also works where plainml's dependencies aren't installed
(for example Vercel's build step).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def checkout_root() -> Path | None:
    """The source checkout this plainml runs from, or None when it's installed from PyPI."""
    root = Path(__file__).resolve().parents[1]
    return root if (root / "pyproject.toml").is_file() and (root / "plainml").is_dir() else None


def build_wheel(source: Path, into: Path) -> Path | None:
    """Build plainml's wheel from a clean copy of ``source`` into ``into``. None if it fails.

    Building in place would leave ``build/`` and ``*.egg-info`` folders in the source tree.
    """
    into.mkdir(parents=True, exist_ok=True)
    before = set(into.glob("plainml-*.whl"))
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
    if built.returncode != 0:
        return None
    new = sorted(set(into.glob("plainml-*.whl")) - before) or sorted(into.glob("plainml-*.whl"))
    return new[-1] if new else None
