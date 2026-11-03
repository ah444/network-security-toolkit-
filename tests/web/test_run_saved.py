"""v5.3 step HR5a: every successful training is saved, exported or not (RD3).

Its record is published at the end of the training; Export writes the bundle only. When the
record can't be saved (the disk too full, HR4c), the run is "Completed, not saved", kept by
the workspace, and ``POST /api/v1/run/save`` saves it again under the same run id."""

import threading

import pytest

from toolkit.core.data import disk
from toolkit.core.data.ids.bundle_files import bundles_dir
from toolkit.core.data.ids.run_files import runs_dir
from toolkit.web.services import training
from web_jobs import TRAIN_URL, train_and_wait, wait_for

CIC = "cic_ddos_sample.csv"
TARGET = "Label"
MB = 2**20


@pytest.fixture
def columns(upload, fixtures_dir):
    return upload(fixtures_dir / CIC).get_json()["columns"]


@pytest.fixture
def free(monkeypatch):
    """The disk's free space, set by the test (in bytes)."""
    space = {"bytes": 100 * 1024 * MB}
    monkeypatch.setattr(disk, "free_bytes", lambda _path: space["bytes"])
    return space


def _body(columns):
    return {"features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing": "none", "test_size": 0.3, "models": ["decision_tree"]}


def _train(client, columns):
    trained = train_and_wait(client, _body(columns))
    assert trained.status_code == 200, trained.get_json()
    return trained.get_json()


def _record(run_id):
    return runs_dir() / f"{run_id}.json"


def test_a_training_is_saved_without_export(client, columns):
    first, second = _train(client, columns), _train(client, columns)
    for result in (first, second):
        assert (result["saved"], result["not_saved"]) == (True, None)
        assert _record(result["run_id"]).is_file()
    listed = [run["run_id"] for run in client.get("/api/v1/runs").get_json()["runs"]]
    assert sorted(listed) == sorted([first["run_id"], second["run_id"]])
    compared = client.get(f"/api/v1/runs/compare?a={first['run_id']}&b={second['run_id']}")
    assert compared.status_code == 200, compared.get_json()
    assert not list(bundles_dir().glob("*.bundle")), "no bundle without Export"


def test_export_writes_the_bundle_only(client, columns):
    run_id = _train(client, columns)["run_id"]
    record = _record(run_id).read_bytes()
    exported = client.post("/api/v1/run/export", json={"name": "first"})
    assert exported.status_code == 200, exported.get_json()
    assert (bundles_dir() / f"{run_id}.bundle").is_file()
    assert _record(run_id).read_bytes() == record, "the record is the training's, untouched"


def test_a_failed_save_says_so_and_save_again_saves_it(client, columns, free):
    free["bytes"] = 10 * MB  # under the 1 GB a new file must leave
    result = _train(client, columns)
    assert result["saved"] is False and result["not_saved"].startswith("Not saved: only 10 MB")
    assert not _record(result["run_id"]).exists()
    resumed = client.get("/api/v1/workspace").get_json()["run"]
    assert (resumed["run_id"], resumed["saved"]) == (result["run_id"], False)
    still = client.post("/api/v1/run/save")
    assert still.status_code == 507 and still.get_json()["error"]["code"] == "NO_SPACE"
    assert not _record(result["run_id"]).exists()
    free["bytes"] = 100 * 1024 * MB
    saved = client.post("/api/v1/run/save")
    assert saved.status_code == 200, saved.get_json()
    assert saved.get_json() == {"run_id": result["run_id"], "saved": True, "not_saved": None}
    assert _record(result["run_id"]).is_file()
    assert client.get("/api/v1/workspace").get_json()["run"]["saved"] is True
    again = client.post("/api/v1/run/save")
    assert again.status_code == 409 and again.get_json()["error"]["code"] == "ALREADY_SAVED"


def test_save_again_while_another_training_runs(client, columns, free, monkeypatch):
    free["bytes"] = 10 * MB
    unsaved = _train(client, columns)["run_id"]
    free["bytes"] = 100 * 1024 * MB
    entered, release = threading.Event(), threading.Event()
    real_train = training.train_run

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30), "the test never released the training"
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_run", held)
    submitted = client.post(TRAIN_URL, json=_body(columns))
    assert submitted.status_code == 202 and entered.wait(30)
    try:
        saved = client.post("/api/v1/run/save")
        assert saved.status_code == 200 and saved.get_json()["run_id"] == unsaved
        assert _record(unsaved).is_file()
    finally:
        release.set()
    job = wait_for(client, submitted.get_json()["url"])
    assert job["state"] == "done" and job["result"]["run_id"] != unsaved
    assert job["result"]["saved"] is True


def test_save_again_needs_a_training(client):
    answer = client.post("/api/v1/run/save")
    assert answer.status_code == 400 and answer.get_json()["error"]["code"] == "NOT_TRAINED"


def test_the_page_says_what_is_saved(client):
    page = client.get("/dynamic_ids").get_data(as_text=True)
    assert "not the rows of your file" in page
