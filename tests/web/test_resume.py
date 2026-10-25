"""v5.1 step 1.3c: GET /api/v1/workspace answers what the session holds, in the same shape
its own routes answered, so the page can resume after a reload."""

import io
import threading

import pytest

from toolkit.services.web import training
from web_jobs import TRAIN_URL, train_and_wait, wait_for

URL = "/api/v1/workspace"
TARGET = "Label"
EMPTY = {"dataset": None, "run": None, "job": None, "export": None, "bundle": None,
         "predictions": None}


@pytest.fixture
def uploaded(upload, fixtures_dir):
    response = upload(fixtures_dir / "cic_ddos_sample.csv")
    assert response.status_code == 200
    return response.get_json()


def _body(columns):
    return {"features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing": "none", "models": ["decision_tree"]}


def _trained(client, uploaded):
    trained = train_and_wait(client, _body(uploaded["columns"]))
    assert trained.status_code == 200
    return trained.get_json()


def test_new_session_gets_nulls_and_creates_nothing(client, workspaces):
    response = client.get(URL)
    assert response.status_code == 200 and response.get_json() == EMPTY
    assert "Set-Cookie" not in response.headers and len(workspaces) == 0
    assert response.headers["Cache-Control"] == "no-store"


def test_dataset_is_the_upload_answer(client, uploaded):
    assert client.get(URL).get_json() == {**EMPTY, "dataset": uploaded}


def test_dataset_after_a_class_filter_counts_the_rows_left(client, uploaded):
    removed = client.post("/api/v1/dataset/filter", json={"target": TARGET, "remove": ["DDoS"]})
    assert removed.status_code == 200
    dataset = client.get(URL).get_json()["dataset"]
    assert dataset["dataset"]["rows"] == removed.get_json()["total_rows"]
    assert dataset["dataset"]["rows"] < uploaded["dataset"]["rows"]
    assert f"Total rows: {dataset['dataset']['rows']}." in dataset["message"]


def test_run_is_the_job_result(client, uploaded):
    result = _trained(client, uploaded)
    answer = client.get(URL).get_json()
    assert answer["run"] == result and answer["job"] is None
    assert answer["dataset"] == uploaded and answer["export"] is None


def test_running_job_is_listed_until_it_ends(client, uploaded, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_train = training.train_run

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_run", held)
    submitted = client.post(TRAIN_URL, json=_body(uploaded["columns"])).get_json()
    try:
        assert entered.wait(10)
        job = client.get(URL).get_json()["job"]
        assert job["id"] == submitted["id"] and job["url"] == submitted["url"]
        assert job["state"] == "running"
    finally:
        release.set()
    done = wait_for(client, submitted["url"])
    answer = client.get(URL).get_json()
    assert answer["job"] is None and answer["run"] == done["result"]


def test_export_is_the_export_answer(client, uploaded):
    _trained(client, uploaded)
    exported = client.post("/api/v1/run/export", json={"name": "resumed"})
    assert exported.status_code == 200
    assert client.get(URL).get_json()["export"] == exported.get_json()


def test_bundle_is_the_load_answer(client, uploaded, web_app):
    _trained(client, uploaded)
    assert client.post("/api/v1/run/export", json={"name": "resumed"}).status_code == 200
    data = client.get("/api/v1/run/bundle").data
    with web_app.test_client() as other:  # a new session that only loads the bundle
        loaded = other.post("/api/v1/bundle", data={"file": (io.BytesIO(data), "resumed.bundle")},
                            content_type="multipart/form-data")
        assert loaded.status_code == 200
        assert other.get(URL).get_json() == {**EMPTY, "bundle": loaded.get_json()}


def test_other_sessions_see_nothing(client, uploaded, web_app):
    _trained(client, uploaded)
    with web_app.test_client() as other:
        assert other.get(URL).get_json() == EMPTY


def test_cleared_workspace_answers_nulls(client, uploaded):
    assert client.delete(URL).status_code == 200
    assert client.get(URL).get_json() == EMPTY
