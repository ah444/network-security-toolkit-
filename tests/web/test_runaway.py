"""One training at a time per workspace; a second request gets 409 (D8, step 0.8).

Since v5.1 step 1.2d a training is a job: the submit returns at once (202) and the job
runs in the app's worker thread, which these tests hold inside training with an Event.
"""

import threading

import pytest

from toolkit.services.web import training
from web_jobs import TRAIN_URL, train_and_wait, wait_for

TARGET = "Label"  # stripped by the v5 loader (B17)


@pytest.fixture
def columns(upload, fixtures_dir):
    return upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]


def _train_body(columns):
    return {"features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing": "none", "test_size": 0.3, "models": ["decision_tree"]}


@pytest.fixture
def held_training(monkeypatch):
    """Jobs stop inside training until released."""
    entered, release = threading.Event(), threading.Event()
    real_train = training.train_run

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30), "test never released the held training"
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_run", held)
    yield entered, release
    release.set()


def test_second_training_returns_409(client, columns, held_training):
    entered, release = held_training
    first = client.post(TRAIN_URL, json=_train_body(columns))
    assert first.status_code == 202
    assert entered.wait(30), "first training never started"
    second = client.post(TRAIN_URL, json=_train_body(columns))
    assert second.status_code == 409
    assert second.get_json()["error"]["code"] == "BUSY"
    assert "already" in second.get_json()["error"]["message"].lower()
    release.set()
    assert wait_for(client, first.get_json()["url"])["state"] == "done"
    assert train_and_wait(client, _train_body(columns)).status_code == 200, \
        "busy flag must be cleared after training"


def test_busy_flag_cleared_after_failure(client, columns, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    real_train = training.train_run
    monkeypatch.setattr(training, "train_run", broken)
    failed = train_and_wait(client, _train_body(columns))
    assert failed.job["state"] == "failed"  # an unexpected error: answered, never shown (S8)
    assert failed.get_json()["error"]["code"] == "INTERNAL"
    assert "boom" not in str(failed.job)
    monkeypatch.setattr(training, "train_run", real_train)
    assert train_and_wait(client, _train_body(columns)).status_code == 200


def test_other_workspace_waits_in_the_queue(web_app, client, columns, held_training,
                                            fixtures_dir):
    entered, release = held_training
    first = client.post(TRAIN_URL, json=_train_body(columns))
    assert entered.wait(30)
    other = web_app.test_client()
    path = fixtures_dir / "cic_ddos_sample.csv"
    with path.open("rb") as handle:
        other.post("/api/v1/dataset", data={"dataset": (handle, path.name)},
                   content_type="multipart/form-data")
    queued = other.post(TRAIN_URL, json=_train_body(columns))
    assert queued.status_code == 202, "another workspace is not refused, it waits"
    view = other.get(queued.get_json()["url"]).get_json()
    assert (view["state"], view["queue_position"]) == ("queued", 1)
    release.set()
    assert wait_for(client, first.get_json()["url"])["state"] == "done"
    assert wait_for(other, queued.get_json()["url"])["state"] == "done"
