"""v5.1 step 1.2d: training jobs - progress, the server-wide queue, cooperative cancel,
the time limit, and jobs that belong to one session (D8, D22).

The app's worker thread is held inside training with an Event where a test needs a job
to stay queued or running.
"""

import threading

import pytest

from toolkit.data.settings import WebConfig
from toolkit.services.web import jobs
from toolkit.services.web import training
from toolkit.controllers.web import training as web_training
from web_jobs import TRAIN_URL, train_and_wait, wait_for
from web_js import js_sources

TARGET = "Label"


@pytest.fixture
def columns(upload, fixtures_dir):
    return upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]


def _body(columns, models=("decision_tree",)):
    return {"features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing": "none", "models": list(models)}


def _upload(test_client, fixtures_dir):
    path = fixtures_dir / "cic_ddos_sample.csv"
    with path.open("rb") as handle:
        return test_client.post("/api/v1/dataset", data={"dataset": (handle, path.name)},
                                content_type="multipart/form-data").get_json()["columns"]


@pytest.fixture
def gate(monkeypatch):
    """Jobs wait inside training until ``release`` is set."""
    entered, release = threading.Event(), threading.Event()
    real_train = training.train_run

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30), "test never released the held training"
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_run", held)
    yield entered, release
    release.set()


def test_submit_then_done_with_full_progress(client, columns):
    submitted = client.post(TRAIN_URL, json=_body(columns, ("decision_tree", "naive_bayes")))
    assert submitted.status_code == 202
    view = submitted.get_json()
    assert view["url"] == f"/api/v1/jobs/{view['id']}" and view["state"] in ("queued", "running")
    job = wait_for(client, view["url"])
    assert job["state"] == "done" and job["phase"] == "Finished"
    assert job["done"] == job["total"] == 2 + 2 * (3 + 1) and job["percent"] == 100
    assert {row["model"] for row in job["result"]["leaderboard"]} == {"decision_tree",
                                                                      "naive_bayes"}
    assert "Trained 2 model(s)" in job["result"]["message"]


def test_bad_request_refused_at_submit(client, columns):
    response = client.post(TRAIN_URL, json={**_body(columns), "models": ["quantum_forest"]})
    assert response.status_code == 400 and "quantum_forest" in response.get_json()["error"][
        "message"]
    assert client.post(TRAIN_URL, json=_body(columns)).status_code == 202, "nothing stays locked"


def test_failure_in_the_job_keeps_nothing(client, columns, monkeypatch):
    def failing(*args, **kwargs):
        raise ValueError("every feature blocked")

    monkeypatch.setattr(training, "train_run", failing)
    failed = train_and_wait(client, _body(columns))
    assert failed.job["state"] == "failed"
    assert failed.get_json()["error"]["code"] == "TRAINING_FAILED"
    assert client.get("/api/v1/run/confusion").get_json()["error"]["code"] == "NOT_TRAINED"


def test_second_session_waits_in_the_queue(web_app, client, columns, gate, fixtures_dir):
    entered, release = gate
    first = client.post(TRAIN_URL, json=_body(columns)).get_json()
    assert entered.wait(30)
    running = client.get(first["url"]).get_json()
    assert running["state"] == "running" and running["phase"].startswith("Preparing")
    other = web_app.test_client()
    second = other.post(TRAIN_URL, json=_body(_upload(other, fixtures_dir))).get_json()
    queued = other.get(second["url"]).get_json()
    assert (queued["state"], queued["queue_position"]) == ("queued", 1)
    assert queued["phase"] == "Waiting for another training to finish"
    release.set()
    assert wait_for(client, first["url"])["state"] == "done"
    assert wait_for(other, second["url"])["state"] == "done"


def test_a_job_with_nothing_ahead_is_starting():
    """Step 1.3d: submitted, not yet picked up by the worker: "Starting", not "0 ahead"."""
    runner = jobs.JobRunner()
    job = jobs.Job(work=lambda job: {}, on_end=lambda: None, max_seconds=60, models=1, folds=3)
    runner._waiting.append(job)  # what submit() does before the worker takes it
    view = web_training.job_view(job, runner)
    assert (view["state"], view["queue_position"], view["phase"]) == ("queued", 0, "Starting")
    train_js = js_sources()["train.js"]
    assert '"Starting..."' in train_js and "ahead of this one" not in train_js
    assert '"Waiting for " + ahead + " other training"' in train_js


def test_cancel_a_queued_job(web_app, client, columns, gate, fixtures_dir):
    entered, release = gate
    first = client.post(TRAIN_URL, json=_body(columns)).get_json()
    assert entered.wait(30)
    other = web_app.test_client()
    other_columns = _upload(other, fixtures_dir)
    second = other.post(TRAIN_URL, json=_body(other_columns)).get_json()
    cancelled = other.delete(second["url"])
    assert cancelled.status_code == 202
    view = cancelled.get_json()
    assert view["state"] == "cancelled" and view["error"]["code"] == "CANCELLED"
    again = other.post(TRAIN_URL, json=_body(other_columns))
    assert again.status_code == 202, "the cancelled job released its workspace"
    release.set()
    assert wait_for(client, first["url"])["state"] == "done"
    assert wait_for(other, again.get_json()["url"])["state"] == "done"
    assert other.get(second["url"]).get_json()["state"] == "cancelled", "it never ran"


def test_cancel_a_running_job_keeps_the_previous_run(client, columns, gate, workspaces):
    entered, release = gate
    release.set()
    assert train_and_wait(client, _body(columns)).status_code == 200
    (ws,) = workspaces.values()
    previous = ws.run.run_id
    entered.clear()
    release.clear()
    job = client.post(TRAIN_URL, json=_body(columns, ("decision_tree", "naive_bayes"))).get_json()
    assert entered.wait(30)
    asked = client.delete(job["url"]).get_json()
    assert asked["state"] == "cancelling"
    release.set()
    ended = wait_for(client, job["url"])
    assert ended["state"] == "cancelled" and ended["error"]["code"] == "CANCELLED"
    assert ended["done"] < ended["total"], "it stopped at the next step"
    assert ws.run.run_id == previous, "a cancelled job stores nothing"
    assert not ws.training.locked() and not ws.leased
    assert train_and_wait(client, _body(columns)).status_code == 200, "the workspace stays usable"


def test_time_limit_stops_the_job(make_app, fixtures_dir):
    app = make_app(WebConfig(job_max_minutes=0))
    test_client = app.test_client()
    stopped = train_and_wait(test_client, _body(_upload(test_client, fixtures_dir)))
    assert stopped.job["state"] == "cancelled"
    assert stopped.get_json()["error"]["code"] == "TIME_LIMIT"


def test_jobs_belong_to_their_session(web_app, client, columns):
    job = client.post(TRAIN_URL, json=_body(columns)).get_json()
    stranger = web_app.test_client()
    for response in (stranger.get(job["url"]), stranger.delete(job["url"])):
        assert response.status_code == 404 and response.get_json()["error"]["code"] == "NOT_FOUND"
    assert client.get("/api/v1/jobs/no-such-job").status_code == 404
    assert wait_for(client, job["url"])["state"] == "done", "the stranger couldn't cancel it"


def test_only_the_last_five_jobs_are_kept(client, columns, workspaces):
    first = train_and_wait(client, _body(columns)).job
    for _ in range(5):
        train_and_wait(client, _body(columns))
    (ws,) = workspaces.values()
    assert len(ws.jobs) == 5
    assert client.get(first["url"]).status_code == 404


def test_queued_job_keeps_its_workspace(web_app, client, columns, gate, fixtures_dir,
                                        workspaces):
    entered, release = gate
    first = client.post(TRAIN_URL, json=_body(columns)).get_json()
    assert entered.wait(30)
    other = web_app.test_client()
    second = other.post(TRAIN_URL, json=_body(_upload(other, fixtures_dir))).get_json()
    idle = workspaces.idle_seconds
    workspaces.idle_seconds = -1
    try:
        assert workspaces.drop_idle() == 0, "running and queued jobs keep their workspaces"
    finally:
        workspaces.idle_seconds = idle
    release.set()
    assert wait_for(client, first["url"])["state"] == "done"
    assert wait_for(other, second["url"])["state"] == "done"
