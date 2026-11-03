"""v5.3 step HR1 (RD2): the permutation importance is a job on the training's queue. One
worker for both kinds, so one heavy job runs at a time; an importance is bound when it is
asked for to its run and model (references, not copies), so a training after it changes
nothing it computes; one job per run and model; the scan keeps its own queue."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

from toolkit.core.services.ids.importance import permutation_importances
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train
from toolkit.core.services.ids.exports import write_run
from toolkit.core.services.ids.saved_importance import NO_RECORD, saved_importance
from toolkit.web.services import importance, jobs
from toolkit.web.services.access import Access, lock_out
from toolkit.web.services.errors import NOT_TRAINED, ServiceError
from toolkit.web.services.workspace import Workspace
from toolkit.web.ui import console


def _run(seed):
    rng = np.random.default_rng(seed)
    label = rng.choice(["BENIGN", "DoS", "PortScan"], size=300)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=300),
                          "b": (label == "PortScan") * 1.5 + rng.normal(size=300),
                          "noise": rng.normal(size=300), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified",
                             models=("decision_tree", "logistic_regression"), workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=300, columns=4))


@pytest.fixture(scope="module")
def run():
    return _run(0)


@pytest.fixture(scope="module")
def other_run():
    return _run(1)


@pytest.fixture
def ws(run):
    workspace = Workspace()
    workspace.run = run
    return workspace


@pytest.fixture
def runner():
    return jobs.JobRunner(jobs.TRAINING, also=(jobs.IMPORTANCE,))


@pytest.fixture
def hold(runner):
    """A training job that holds the worker until ``release`` is set; its time on it is kept."""
    release = threading.Event()
    held = _timed(jobs.TRAINING, lambda: release.wait(30))
    runner.submit(held)
    _until(lambda: held.state == jobs.RUNNING)
    yield held, release
    release.set()


TIMES: dict[str, tuple[float, float]] = {}


def _timed(kind, body=lambda: None):
    def work(job):
        begun = time.monotonic()
        body()
        TIMES[job.id] = (begun, time.monotonic())
        return {}

    return jobs.Job(kind=kind, work=work, on_end=lambda: None, max_seconds=60,
                    steps=lambda done, total: "")


def _until(condition, seconds=30):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _wait(job):
    _until(lambda: job.state in jobs.FINISHED)
    return job


def _expected(run, model):
    found = permutation_importances(run.evaluation.training.models[model], run.split.test,
                                    "Label", seed=run.evaluation.seed)
    return [{"feature": f.feature, "mean": f.mean, "std": f.std} for f in found.features]


def test_the_importance_of_the_selected_model_is_the_cores(ws, run, runner):
    job = _wait(importance.start_importance(ws, None, runner, 60))
    assert job.state == jobs.DONE, job.error
    result = job.result
    assert (result["run_id"], result["model"]) == (run.run_id, run.evaluation.selected)
    assert result["features"] == _expected(run, run.evaluation.selected)
    assert result["seed"] == run.evaluation.seed and result["rows"] == len(run.split.test)
    assert job.done == job.total == 3, "a step per feature"


def test_its_result_is_published_as_its_runs_file(ws, run, runner):
    """HR2: with the run's record there (saved by its training, HR5), the result is its own
    file."""
    write_run(run)
    job = _wait(importance.start_importance(ws, "decision_tree", runner, 60))
    assert (job.result["saved"], job.result["not_saved"]) == (True, None)
    stored = saved_importance(run.run_id, "decision_tree")
    assert stored["features"] == job.result["features"] == _expected(run, "decision_tree")


def test_without_a_record_the_job_says_it_was_not_saved(ws, run, runner):
    job = _wait(importance.start_importance(ws, None, runner, 60))
    assert job.state == jobs.DONE and job.result["saved"] is False
    assert job.result["not_saved"] == NO_RECORD
    assert saved_importance(run.run_id, run.evaluation.selected) is None


def test_it_computes_the_run_it_was_asked_for(ws, run, other_run, runner, hold):
    held, release = hold
    job = importance.start_importance(ws, "logistic_regression", runner, 60)
    ws.run = other_run  # a new training in the workspace while the importance waits
    release.set()
    result = _wait(job).result
    assert result["run_id"] == run.run_id
    assert result["features"] == _expected(run, "logistic_regression")


def test_one_job_per_run_and_model(ws, run, runner, hold):
    held, release = hold
    first = importance.start_importance(ws, "decision_tree", runner, 60)
    assert importance.start_importance(ws, "decision_tree", runner, 60) is first, "waiting"
    other_model = importance.start_importance(ws, "logistic_regression", runner, 60)
    assert other_model is not first
    runner.cancel(other_model)
    again = importance.start_importance(ws, "logistic_regression", runner, 60)
    assert again is not other_model, "a cancelled one can be asked for again"
    release.set()
    _wait(first)
    _wait(again)
    assert importance.start_importance(ws, "decision_tree", runner, 60) is first, "done"


def test_refused_before_the_queue(ws, runner):
    with pytest.raises(ServiceError) as unknown:
        importance.start_importance(ws, "knn", runner, 60)
    assert "has no model 'knn'" in unknown.value.message
    ws.run = None
    with pytest.raises(ServiceError) as untrained:
        importance.start_importance(ws, None, runner, 60)
    assert untrained.value.code == NOT_TRAINED
    assert runner.active() == [] and not ws.leased


def test_one_heavy_job_at_a_time_in_the_order_asked(ws, runner, hold, monkeypatch):
    held, release = hold
    real = importance.permutation_importances

    def timed(*args, **kwargs):
        begun = time.monotonic()
        found = real(*args, **kwargs)
        TIMES["importance"] = (begun, time.monotonic())
        return found

    monkeypatch.setattr(importance, "permutation_importances", timed)
    job = importance.start_importance(ws, None, runner, 60)
    after = _timed(jobs.TRAINING, lambda: time.sleep(0.05))
    runner.submit(after)
    assert runner.position(job) == 1 and runner.position(after) == 2
    assert job.phase() == after.phase() == "Waiting for another training or importance to finish"
    release.set()
    _wait(after)
    spans = sorted([TIMES[held.id], TIMES["importance"], TIMES[after.id]])
    assert spans == [TIMES[held.id], TIMES["importance"], TIMES[after.id]], "in order asked"
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)), "never two at once"


def test_it_holds_the_workspace_but_not_its_training(ws, runner, hold):
    held, release = hold
    job = importance.start_importance(ws, None, runner, 60)
    assert ws.leased and not ws.training.locked(), "a training can be asked for meanwhile"
    release.set()
    _wait(job)
    assert not ws.leased


def test_cancel_while_waiting_never_starts(ws, runner, hold, monkeypatch):
    held, release = hold
    called = []
    monkeypatch.setattr(importance, "permutation_importances",
                        lambda *args, **kwargs: called.append(1))
    job = importance.start_importance(ws, None, runner, 60)
    runner.cancel(job)
    release.set()
    _wait(held)
    time.sleep(0.05)
    assert job.state == jobs.STOPPED and called == []
    assert job.error["message"] == "Importance cancelled before it started."


def test_cancel_while_computing_stops_between_features(ws, runner, monkeypatch):
    inside, go_on = threading.Event(), threading.Event()

    def slow(*args, progress, **kwargs):
        progress.start(3)
        inside.set()
        go_on.wait(30)
        progress.advance()  # the cancel takes effect here
        raise AssertionError("not stopped")

    monkeypatch.setattr(importance, "permutation_importances", slow)
    job = importance.start_importance(ws, None, runner, 60)
    assert inside.wait(30)
    assert job.phase() == "Permutation importance, feature 1 of 3"
    runner.cancel(job)
    go_on.set()
    assert _wait(job).error == {"code": "CANCELLED",
                                "message": "Importance cancelled. Nothing was kept from it."}


def test_the_worker_goes_on_after_a_job_raises(ws, runner, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(importance, "permutation_importances", broken)
    failed = _wait(importance.start_importance(ws, None, runner, 60))
    assert failed.state == jobs.FAILED and failed.error["code"] == "INTERNAL"
    after = _timed(jobs.TRAINING)
    runner.submit(after)
    assert _wait(after).state == jobs.DONE
    assert not ws.leased


def test_a_scan_runs_beside_them(ws, runner, hold):
    held, release = hold
    job = importance.start_importance(ws, None, runner, 60)
    scans = jobs.JobRunner(jobs.SCAN)
    scan = _timed(jobs.SCAN)
    scans.submit(scan)
    assert _wait(scan).state == jobs.DONE and job.state == jobs.QUEUED
    with pytest.raises(ValueError):
        scans.submit(_timed(jobs.IMPORTANCE))


def test_the_server_gives_both_kinds_one_queue(web_app):
    state = web_app.extensions["toolkit"]
    assert state.runner(jobs.IMPORTANCE) is state.runner(jobs.TRAINING) is state.training_jobs
    assert state.runner(jobs.SCAN) is state.scan_jobs


def test_a_new_token_cancels_a_remote_waiting_importance(ws, runner, hold):
    access = Access()
    runner._refusal = access.refusal  # as create_app gives it
    held, release = hold
    job = importance.start_importance(ws, None, runner, 60,
                                      origin=jobs.Origin("192.168.56.5", access.run_id))
    assert lock_out(access, [runner])[1] == 1
    assert job.state == jobs.STOPPED and "the access token was changed" in job.error["message"]
    assert held.state == jobs.RUNNING, "a running job goes on"


def test_a_cut_off_importance_says_what_was_lost():
    assert console.interrupted_line("importance", "2026-11-03 14:02", "this computer", True) == (
        "INTERRUPTED: an importance asked for at 2026-11-03 14:02 from this computer was cut "
        "off while it ran, when the web server stopped. Nothing of it was kept: its run's "
        "models and test split were in memory only, so train again to calculate it.")
    assert console.interrupted_line("training", "x", "y", False).startswith("INTERRUPTED: a training")


def test_the_trainers_page_tells_a_cut_off_importance(web_app, client):
    web_app.extensions["toolkit"].interrupted = {
        "training": ["INTERRUPTED: a training x"], "importance": ["INTERRUPTED: an importance y"]}
    page = client.get("/dynamic_ids").get_data(as_text=True)
    assert "INTERRUPTED: a training x" in page and "INTERRUPTED: an importance y" in page
    assert "INTERRUPTED" not in client.get("/dynamic_ids").get_data(as_text=True)
