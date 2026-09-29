"""The web app's API: uploads, jobs, runs, per-file downloads, and the optional token."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from plainml.web.server import create_app


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "runs"))


def _upload(client: TestClient, path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = client.post("/api/uploads", files={"file": (path.name, handle, "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()


def _finish(client: TestClient, job: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
    deadline = time.time() + timeout
    while job["status"] in ("queued", "running"):
        assert time.time() < deadline, f"job still {job['status']}: {job['stage']}"
        time.sleep(0.2)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


def test_page_and_info(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "app.js" in page.text
    assert client.get("/app.css").status_code == 200
    info = client.get("/api/info").json()
    assert info["needs_token"] is False and ".csv" in info["upload_types"]
    assert any(m["default"] for m in info["importance_methods"])


def test_upload_summary(client: TestClient, messy_csv: Path) -> None:
    summary = _upload(client, messy_csv)
    assert summary["rows"] > 0 and summary["preview"]["rows"]
    kinds = {c["name"]: c["kind"] for c in summary["columns"]}
    assert kinds["signup"] == "datetime"
    assert summary["suggested_target"] == "churned"
    assert client.get(f"/api/uploads/{summary['id']}").json()["rows"] == summary["rows"]
    bad = client.post(
        "/api/uploads", files={"file": ("notes.exe", b"MZ", "application/octet-stream")}
    )
    assert bad.status_code == 415


def test_train_job_run_and_files(client: TestClient, messy_csv: Path) -> None:
    upload = _upload(client, messy_csv)
    rejected = client.post(
        "/api/jobs", json={"task": "train", "upload": upload["id"], "options": {}}
    )
    assert (
        rejected.status_code == 422 and "column to predict" in rejected.json()["detail"]["message"]
    )
    job = client.post(
        "/api/jobs",
        json={
            "task": "train",
            "upload": upload["id"],
            "options": {"target": "churned", "speed": "quick"},
        },
    ).json()
    job = _finish(client, job)
    assert job["status"] == "done", job["error"]
    assert any("Training" in line for line in job["log"])  # console output was captured
    name = job["result"]["run"]
    run = client.get(f"/api/runs/{name}").json()
    assert run["predictable"] and run["report"] and run["tiles"][0]["value"]
    files = {f["name"]: f for f in run["files"]}
    assert {"report.html", "model.joblib", "leaderboard.csv"} <= set(files)
    assert run["files"][0]["name"] == "report.html"  # the report comes first
    download = client.get(files["model.joblib"]["download"])
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
    preview = client.get(files["leaderboard.csv"]["preview"]).json()
    assert preview["columns"] and preview["rows"]
    for sneaky in ("../../etc/passwd", "..%2F..%2Frun.json"):
        assert client.get(f"/api/runs/{name}/files/{sneaky}").status_code == 404
    assert client.get("/api/runs/..%2F/files/run.json").status_code == 404
    listed = client.get("/api/runs").json()
    assert listed[0]["run"] == name

    # predict with it, from the same upload
    predict = client.post(
        "/api/jobs",
        json={"task": "predict", "upload": upload["id"], "options": {"model": name, "proba": True}},
    ).json()
    predict = _finish(client, predict)
    assert predict["status"] == "done", predict["error"]
    result = predict["result"]
    assert result["kind"] == "output" and "predicted_churned" in result["preview"]["columns"]
    csv = client.get(result["files"][0]["download"])
    assert csv.status_code == 200 and b"predicted_churned" in csv.content


def test_output_tasks(client: TestClient, messy_csv: Path) -> None:
    upload = _upload(client, messy_csv)
    profile = _finish(
        client,
        client.post(
            "/api/jobs", json={"task": "profile", "upload": upload["id"], "options": {}}
        ).json(),
    )
    assert profile["status"] == "done" and profile["result"]["report"]
    assert client.get(profile["result"]["report"]).status_code == 200
    cleaned = _finish(
        client,
        client.post(
            "/api/jobs", json={"task": "clean", "upload": upload["id"], "options": {"impute": True}}
        ).json(),
    )
    assert cleaned["status"] == "done"
    data = client.get(cleaned["result"]["files"][0]["download"])
    frame = pd.read_csv(pd.io.common.BytesIO(data.content))
    assert len(frame) > 0


def test_failed_job_reports_the_error(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "flat.csv"
    pd.DataFrame({"x": range(30), "y": ["same"] * 30}).to_csv(path, index=False)
    upload = _upload(client, path)
    job = _finish(
        client,
        client.post(
            "/api/jobs", json={"task": "train", "upload": upload["id"], "options": {"target": "y"}}
        ).json(),
    )
    assert job["status"] == "failed" and "same value" in job["error"]["message"]
    # the worker survives a failure
    ok = _finish(
        client,
        client.post(
            "/api/jobs", json={"task": "profile", "upload": upload["id"], "options": {}}
        ).json(),
    )
    assert ok["status"] == "done"


def test_token(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "runs", token="s3cret"))
    assert client.get("/api/info").json()["signed_in"] is False
    assert client.get("/api/runs").status_code == 401
    assert client.post("/api/login", json={"token": "nope"}).status_code == 401
    assert client.post("/api/login", json={"token": "s3cret"}).status_code == 200
    assert client.get("/api/runs").status_code == 200  # the cookie is now set
