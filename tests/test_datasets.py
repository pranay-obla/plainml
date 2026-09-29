"""The example datasets that ship with plainml, for trying it without your own data."""

from __future__ import annotations

from pathlib import Path

import pytest

from plainml import datasets
from plainml.errors import PlainMLError
from plainml.web.server import TASK_TITLES, summarize_upload

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("key", sorted(datasets.SAMPLES))
def test_each_sample_suits_its_task(key: str) -> None:
    sample = datasets.get(key)
    assert sample.task in TASK_TITLES
    summary = summarize_upload(datasets.path(key), "0123456789ab")
    assert summary["rows"] == sample.rows
    kinds = {c["name"]: c["kind"] for c in summary["columns"]}
    options = sample.options
    named = [options.get(k) for k in ("target", "group", "label")] + options.get("inputs", [])
    for column in named:
        assert column is None or column in kinds, f"{key}: no column '{column}'"
    if sample.task == "forecast":
        assert kinds[options["target"]] == "numeric"
        assert all(kinds[c] == "numeric" for c in options.get("inputs", []))
        assert options.get("group") is None or kinds[options["group"]] == "categorical"


def test_packaged_copies_match_the_examples_folder() -> None:
    if not EXAMPLES.is_dir():
        pytest.skip("not running from a source checkout")
    for sample in datasets.SAMPLES.values():
        packaged = datasets.path(sample.key).read_bytes().replace(b"\r\n", b"\n")
        original = (EXAMPLES / sample.file).read_bytes().replace(b"\r\n", b"\n")
        assert packaged == original, f"re-run: python examples/make_datasets.py ({sample.file})"


def test_unknown_sample() -> None:
    with pytest.raises(PlainMLError, match="Choose one of"):
        datasets.get("nope")


def test_reachable_from_the_package() -> None:
    import plainml

    assert plainml.datasets.path("churn").is_file()
