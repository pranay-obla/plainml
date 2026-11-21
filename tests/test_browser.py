"""The in-browser website: its Python backend (run natively here) and the static export."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from plainml.web.browser import Backend, one_thread
from plainml.web.static_site import export_static


def _call(backend: Backend, method: str, path: str, body: Any = None) -> dict[str, Any]:
    return json.loads(backend.request(method, path, json.dumps(body) if body is not None else None))


def test_backend_answers_like_the_server(messy_csv: Path, tmp_path: Path) -> None:
    updates: list[dict[str, Any]] = []
    backend = Backend(tmp_path / "runs", notify=lambda text: updates.append(json.loads(text)))
    assert _call(backend, "GET", "/api/info")["body"]["mode"] == "browser"

    upload = json.loads(backend.upload("messy.csv", messy_csv.read_bytes()))
    assert upload["status"] == 200 and upload["body"]["suggested_target"] == "churned"
    upload_id = upload["body"]["id"]
    assert (
        _call(backend, "GET", f"/api/uploads/{upload_id}")["body"]["rows"] == upload["body"]["rows"]
    )

    rejected = _call(backend, "POST", "/api/jobs", {"task": "train", "upload": upload_id})
    assert (
        rejected["status"] == 422 and "column to predict" in rejected["body"]["detail"]["message"]
    )
    job = _call(
        backend,
        "POST",
        "/api/jobs",
        {"task": "train", "upload": upload_id, "options": {"target": "churned", "speed": "quick"}},
    )["body"]
    assert job["status"] == "queued"
    assert backend.run_pending() and not backend.run_pending()
    assert updates and updates[-1]["status"] == "done"  # progress was pushed while it ran

    finished = _call(backend, "GET", f"/api/jobs/{job['id']}?log_from=0")["body"]
    run_name = finished["result"]["run"]
    run = _call(backend, "GET", f"/api/runs/{run_name}")["body"]
    files = {f["name"]: f for f in run["files"]}
    visual = run["visual"]  # the 3D view: every model's score, the winner marked
    assert visual["type"] == "bars" and any(b["tone"] == "best" for b in visual["bars"])
    content, kind = backend.read(files["report.html"]["url"])
    assert content.startswith(b"<!doctype html>") and kind.startswith("text/html")
    assert _call(backend, "GET", files["leaderboard.csv"]["preview"])["body"]["rows"]
    assert _call(backend, "GET", "/api/runs")["body"][0]["run"] == run_name

    assert _call(backend, "GET", "/api/runs/nope")["status"] == 404
    assert _call(backend, "GET", "/api/jobs/0123456789ab")["status"] == 404
    with pytest.raises(Exception, match="No file"):
        backend.read(f"/api/runs/{run_name}/files/../../secret.txt")


def test_backend_upload_checks(tmp_path: Path) -> None:
    backend = Backend(tmp_path / "runs")
    assert json.loads(backend.upload("virus.exe", b"MZ"))["status"] == 415

    class JsBytes:  # what a JavaScript Uint8Array looks like from Python in Pyodide
        def to_bytes(self) -> bytes:
            return b"a,b\n1,2\n3,4\n"

    answer = json.loads(backend.upload("tiny.csv", JsBytes()))  # type: ignore[arg-type]
    assert answer["status"] == 200 and answer["body"]["rows"] == 2


def test_backend_example_datasets(tmp_path: Path) -> None:
    backend = Backend(tmp_path / "runs")
    assert _call(backend, "GET", "/api/info")["body"]["samples"]
    upload = _call(backend, "POST", "/api/samples/customers")["body"]
    assert upload["name"] == "customers.csv" and upload["sample"]["task"] == "cluster"
    job = _call(backend, "POST", "/api/jobs", {"task": "cluster", "upload": upload["id"]})["body"]
    assert backend.run_pending()
    done = _call(backend, "GET", f"/api/jobs/{job['id']}")["body"]
    assert done["status"] == "done"
    visual = _call(backend, "GET", f"/api/runs/{done['result']['run']}")["body"]["visual"]
    assert visual["type"] == "points" and len(visual["groups"]) == 3
    assert len(visual["x"]) == len(visual["y"]) == len(visual["z"]) == len(visual["group"])
    assert _call(backend, "POST", "/api/samples/nope")["status"] == 404


def test_jobs_run_on_one_thread() -> None:
    from joblib.parallel import SequentialBackend, get_active_backend

    with one_thread():
        assert isinstance(get_active_backend()[0], SequentialBackend)


def _built_site(folder: Path) -> Path:
    """A stand-in for the built website (frontend/ needs Node.js to build)."""
    (folder / "runs").mkdir(parents=True)
    (folder / "_next").mkdir()
    page = '<!DOCTYPE html><html lang="en"><head><meta charSet="utf-8"/></head><body></body></html>'
    (folder / "index.html").write_text(page)
    (folder / "runs" / "index.html").write_text(page)
    (folder / "_next" / "app.js").write_text("1")
    (folder / "worker.js").write_text("// worker")
    return folder


def test_export_writes_a_static_site(tmp_path: Path) -> None:
    site = _built_site(tmp_path / "built")
    out = export_static(tmp_path / "site", bundle_wheel=False, site_files=site)
    assert (out / "_next" / "app.js").is_file() and (out / "worker.js").is_file()
    marker = '<meta name="plainml-mode" content="browser">'
    assert marker in (out / "index.html").read_text()
    assert marker in (out / "runs" / "index.html").read_text()  # on every page
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["wheel"] is None and manifest["version"]  # installs that version from PyPI


def test_export_needs_the_built_site(tmp_path: Path) -> None:
    from plainml.web.static_site import ExportError

    with pytest.raises(ExportError, match="isn't built"):
        export_static(tmp_path / "site", bundle_wheel=False, site_files=tmp_path)


def test_export_bundles_a_wheel_from_a_checkout(tmp_path: Path) -> None:
    from plainml._wheel import checkout_root

    if checkout_root() is None:
        pytest.skip("plainml isn't running from a source checkout")
    site = _built_site(tmp_path / "built")
    out = export_static(tmp_path / "site", bundle_wheel=True, site_files=site)
    wheel = json.loads((out / "manifest.json").read_text())["wheel"]
    assert wheel.startswith("wheels/") and (out / wheel).is_file()
    with zipfile.ZipFile(out / wheel) as archive:
        names = archive.namelist()
    assert "plainml/datasets/churn.csv" in names  # the example datasets ship too
    assert not any(n.startswith("plainml/web/static/") for n in names)  # the site is already there
