"""v5.2 step 2.4b: jobs of more than one kind, each kind with its own queue.

Training's answers are pinned as they were before the jobs became generic (the 404 text
says "job" now, for any kind); the scan queue runs one scan at a time for the whole
server, next to a training, without a route yet (step 2.4d).
"""

import threading
import time

import pytest

from toolkit.core.data.settings import WebConfig
from toolkit.web.services import jobs, training
from web_jobs import TRAIN_URL, train_and_wait, wait_for

TARGET = "Label"
VIEW_KEYS = {"id", "url", "state", "done", "total", "percent", "phase", "elapsed_seconds"}


@pytest.fixture
def columns(upload, fixtures_dir):
    return upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]


def _body(columns):
    return {"features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing": "none", "models": ["decision_tree"]}


def _upload(test_client, fixtures_dir):
    path = fixtures_dir / "cic_ddos_sample.csv"
    with path.open("rb") as handle:
        return test_client.post("/api/v1/dataset", data={"dataset": (handle, path.name)},
                                content_type="multipart/form-data").get_json()["columns"]


@pytest.fixture
def gate(monkeypatch):
    """Training jobs wait inside training until ``release`` is set."""
    entered, release = threading.Event(), threading.Event()
    real_train = training.train_run

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30), "test never released the held training"
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_run", held)
    yield entered, release
    release.set()


# --- Training, pinned ----------------------------------------------------------------------

def test_a_training_answers_as_before(client, columns):
    done = train_and_wait(client, _body(columns)).job
    assert set(done) - {"kind"} == VIEW_KEYS | {"result"}
    assert (done["state"], done["phase"], done["percent"]) == ("done", "Finished", 100)
    assert done["done"] == done["total"] == 2 + 1 * (3 + 1)


def test_a_training_says_how_it_stopped(web_app, client, columns, gate, fixtures_dir):
    entered, release = gate
    first = client.post(TRAIN_URL, json=_body(columns)).get_json()
    assert entered.wait(30)
    other = web_app.test_client()
    second = other.post(TRAIN_URL, json=_body(_upload(other, fixtures_dir))).get_json()
    queued = other.get(second["url"]).get_json()
    assert queued["phase"] == "Waiting for another training to finish"
    assert other.delete(second["url"]).get_json()["error"] == {
        "code": "CANCELLED", "message": "Training cancelled before it started."}
    assert client.delete(first["url"]).get_json()["state"] == "cancelling"
    release.set()
    assert wait_for(client, first["url"])["error"] == {
        "code": "CANCELLED", "message": "Training cancelled. Nothing was kept from it."}


def test_a_training_past_its_time_limit(make_app, fixtures_dir):
    app = make_app(WebConfig(job_max_minutes=0))
    test_client = app.test_client()
    stopped = train_and_wait(test_client, _body(_upload(test_client, fixtures_dir)))
    assert stopped.get_json()["error"] == {
        "code": "TIME_LIMIT",
        "message": "Training stopped after 0 minutes (TOOLKIT_JOB_MAX_MIN). Nothing was kept."}


def test_the_training_worker_thread_keeps_its_name(client, columns):
    train_and_wait(client, _body(columns))
    assert "training-jobs" in {thread.name for thread in threading.enumerate()}


# --- The scan queue ------------------------------------------------------------------------

def _scan(work, max_seconds=60):
    return jobs.Job(kind=jobs.SCAN, work=work, on_end=lambda: None, max_seconds=max_seconds,
                    steps=lambda done, total: f"Port {done} of {total}")


def _held(entered, release):
    """A scan's work that runs until ``release``, a step at a time (where a cancel acts)."""
    def work(job):
        job.start(10)
        entered.set()
        while not release.wait(0.01):
            job.advance(0)
        return {"open": []}
    return work


def _submit(app, ws, job):
    ws.add_job(job)
    app.extensions["toolkit"].scan_jobs.submit(job)
    return f"/api/v1/jobs/{job.id}"


def test_a_scan_runs_next_to_a_training(web_app, client, columns, gate, workspaces):
    entered, release = gate
    trained = client.post(TRAIN_URL, json=_body(columns)).get_json()
    assert entered.wait(30), "the training is running"
    (ws,) = workspaces.values()
    scan_url = _submit(web_app, ws, _scan(lambda job: {"open": [22]}))
    scanned = wait_for(client, scan_url)
    assert (scanned["kind"], scanned["state"], scanned["result"]) == ("scan", "done",
                                                                      {"open": [22]})
    training_view = client.get(trained["url"]).get_json()
    assert (training_view["kind"], training_view["state"]) == ("training", "running"), \
        "the scan didn't wait for the training"
    release.set()
    assert wait_for(client, trained["url"])["state"] == "done"


def test_one_scan_at_a_time_for_the_whole_server(web_app, client, columns, workspaces):
    entered, release = threading.Event(), threading.Event()
    (ws,) = workspaces.values()
    first = _submit(web_app, ws, _scan(_held(entered, release)))
    assert entered.wait(30)
    second = _submit(web_app, ws, _scan(lambda job: {"open": []}))
    running = client.get(first).get_json()
    assert (running["state"], running["phase"]) == ("running", "Port 0 of 10")
    queued = client.get(second).get_json()
    assert (queued["state"], queued["queue_position"]) == ("queued", 1)
    assert queued["phase"] == "Waiting for another scan to finish"
    assert client.delete(second).get_json()["error"] == {
        "code": "CANCELLED", "message": "Scan cancelled before it started."}
    assert client.delete(first).get_json()["state"] == "cancelling"
    stopped = wait_for(client, first)
    assert stopped["error"] == {"code": "CANCELLED",
                                "message": "Scan cancelled. Nothing was kept from it."}
    assert "result" not in stopped
    assert "scan-jobs" in {thread.name for thread in threading.enumerate()}
    release.set()


def test_a_scan_past_its_time_limit(web_app, client, columns, workspaces):
    def slow(job):
        job.start(2)
        time.sleep(0.05)
        job.advance()
        return {"open": []}

    (ws,) = workspaces.values()
    stopped = wait_for(client, _submit(web_app, ws, _scan(slow, max_seconds=0)))
    assert stopped["error"] == {
        "code": "TIME_LIMIT",
        "message": "Scan stopped after 0 minutes (TOOLKIT_SCAN_MAX_MIN). Nothing was kept."}


def test_a_queue_takes_only_its_own_kind(web_app):
    state = web_app.extensions["toolkit"]
    training_job = jobs.Job(kind=jobs.TRAINING, work=lambda job: {}, on_end=lambda: None,
                            max_seconds=60, steps=training.training_steps(1, 3))
    with pytest.raises(ValueError, match="training job can't wait in the scan queue"):
        state.scan_jobs.submit(training_job)
    assert state.scan_jobs.position(training_job) == 0 and training_job.state == "queued"
    assert state.runner(jobs.SCAN) is state.scan_jobs
    assert state.runner(jobs.TRAINING) is state.training_jobs
    with pytest.raises(LookupError):
        state.runner(jobs.JobKind("other", "Other", "TOOLKIT_OTHER"))


def test_the_trainer_page_never_resumes_a_scan(web_app, client, columns, workspaces):
    entered, release = threading.Event(), threading.Event()
    (ws,) = workspaces.values()
    scan_url = _submit(web_app, ws, _scan(_held(entered, release)))
    assert entered.wait(30)
    assert client.get("/api/v1/workspace").get_json()["job"] is None
    release.set()
    assert wait_for(client, scan_url)["state"] == "done"


def test_each_kind_keeps_its_own_last_five(client, columns, workspaces):
    first = train_and_wait(client, _body(columns)).job
    (ws,) = workspaces.values()
    for _ in range(6):
        scan = _scan(lambda job: {})
        scan.state = jobs.DONE
        ws.add_job(scan)
    assert client.get(first["url"]).status_code == 200, "scans didn't push out the training"
    assert sum(job.kind == jobs.SCAN for job in ws.jobs.values()) == 5
    gone = client.get("/api/v1/jobs/no-such-job").get_json()["error"]
    assert gone == {"code": "NOT_FOUND", "message": "No such job in this session."}
