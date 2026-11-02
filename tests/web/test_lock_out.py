"""HS1 (RD1): a new access token, made on this computer only, locks every other computer out at
once without stopping the app. A job another computer asked for under the old unlock is
cancelled while it still waits and never starts; a running one goes on and can be stopped on
this computer; this computer's own jobs are not touched. The checks of every request hold for
other computers too. No test binds a network address: other computers are a REMOTE_ADDR.
"""

import logging
import threading
import time

import pytest

from toolkit.core.data import logs
from toolkit.core.data.settings import WebConfig
from toolkit.web.controllers import scan as scan_routes
from toolkit.web.services import access as access_service
from toolkit.web.services import jobs
from toolkit.web.services import training
from web_jobs import TRAIN_URL, wait_for
from web_scan import fake_network

OTHER, SECOND = "192.168.56.20", "192.168.56.21"  # two other computers
TOKEN_URL = "/api/v1/access/token"
JOBS_URL = "/api/v1/access/jobs"
TARGET = "Label"


@pytest.fixture
def open_app(make_app):
    return make_app(WebConfig(host="0.0.0.0"))


@pytest.fixture
def access(open_app):
    return open_app.extensions["toolkit"].access


def _other(app, address=OTHER):
    test_client = app.test_client()
    test_client.environ_base["REMOTE_ADDR"] = address
    return test_client


def _unlock(test_client, typed):
    test_client.get("/unlock")  # the unlock page gives the session its CSRF token
    return test_client.post("/api/v1/unlock", json={"token": typed})


def _unlocked(app, address=OTHER):
    test_client = _other(app, address)
    assert _unlock(test_client, app.extensions["toolkit"].access.token).status_code == 200
    return test_client


def _code(response):
    return response.get_json()["error"]["code"]


@pytest.fixture
def gate(monkeypatch):
    """Trainings wait inside training until ``release`` is set."""
    entered, release = threading.Event(), threading.Event()
    real_train = training.train_run

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30), "test never released the held training"
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_run", held)
    yield entered, release
    release.set()


def _train(test_client, fixtures_dir):
    """Upload the sample and queue a training of it; returns the job's view."""
    path = fixtures_dir / "cic_ddos_sample.csv"
    with path.open("rb") as handle:
        columns = test_client.post("/api/v1/dataset", data={"dataset": (handle, path.name)},
                                   content_type="multipart/form-data").get_json()["columns"]
    body = {"features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing": "none", "models": ["decision_tree"]}
    submitted = test_client.post(TRAIN_URL, json=body)
    assert submitted.status_code == 202
    return submitted.get_json()


def _job(app, job_id):
    """The job object with this id, in whichever session's workspace it is."""
    found = [ws.jobs[job_id] for ws in app.extensions["toolkit"].workspaces.values()
             if job_id in ws.jobs]
    assert len(found) == 1
    return found[0]


def _ended(job, timeout=120):
    deadline = time.monotonic() + timeout
    while job.state not in jobs.FINISHED:
        assert time.monotonic() < deadline, f"the job didn't end within {timeout} s"
        time.sleep(0.02)
    return job


# --- The new token ------------------------------------------------------------------------

def test_a_new_token_locks_every_other_computer_out(open_app, access):
    other = _unlocked(open_app)
    old = access.token
    made = open_app.test_client().post(TOKEN_URL)
    assert made.status_code == 200 and access.token != old
    answer = made.get_data(as_text=True)
    assert old not in answer and access.token not in answer, "the token is in the terminal only"
    assert made.get_json()["cancelled"] == 0
    for page in ("/", "/scan", "/dynamic_ids", "/runs"):
        assert other.get(page).headers["Location"] == "/unlock"
    for method, url in (("GET", "/api/v1/scan/options"), ("GET", "/api/v1/jobs/abc"),
                        ("DELETE", "/api/v1/jobs/abc"), ("POST", TRAIN_URL),
                        ("POST", "/api/v1/jobs/scan")):
        refused = other.open(url, method=method, json={})
        assert refused.status_code == 401 and _code(refused) == "LOCKED", url
    assert _code(_unlock(other, old)) == "WRONG_TOKEN", "the old token is gone"
    assert _unlock(other, access.token).status_code == 200, "the new one unlocks"
    assert other.get("/scan").status_code == 200


def test_the_new_token_is_printed_never_logged(open_app, access, caplog, capsys):
    caplog.set_level(logging.INFO)
    made = open_app.test_client().post(TOKEN_URL)
    assert made.status_code == 200 and "Location" not in made.headers
    assert access.token in capsys.readouterr().out
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "A new access token was made on this computer" in text and access.token not in text
    assert logs.redact(f"token {access.token}") == "token [REDACTED]", "a registered secret"


def test_the_control_is_on_this_computer_only(open_app, access):
    old = access.token
    other = _unlocked(open_app)
    for method, url in (("POST", TOKEN_URL), ("GET", JOBS_URL), ("DELETE", f"{JOBS_URL}/abc")):
        refused = other.open(url, method=method)
        assert refused.status_code == 403 and _code(refused) == "THIS_COMPUTER_ONLY", url
    local = open_app.test_client()
    cross_site = local.post(TOKEN_URL, headers={"Origin": "http://evil.example"})
    assert cross_site.status_code == 403 and _code(cross_site) == "CROSS_SITE"
    local.send_token = False
    no_token = local.post(TOKEN_URL)
    assert no_token.status_code == 403 and _code(no_token) == "CSRF_TOKEN"
    assert access.token == old, "nothing was refused after the token was changed"


def test_without_host_there_is_no_token_to_change(client):
    for method, url in (("POST", TOKEN_URL), ("GET", JOBS_URL), ("DELETE", f"{JOBS_URL}/abc")):
        refused = client.open(url, method=method)
        assert refused.status_code == 404 and _code(refused) == "NOT_FOUND", url


# --- What happens to each job -------------------------------------------------------------

def test_what_happens_to_each_job(open_app, access, gate, fixtures_dir):
    entered, release = gate
    running = _unlocked(open_app)
    first = _train(running, fixtures_dir)
    assert entered.wait(30)
    second = _train(_unlocked(open_app, SECOND), fixtures_dir)
    local = open_app.test_client()
    mine = _train(local, fixtures_dir)
    listed = local.get(JOBS_URL).get_json()["jobs"]
    assert [(job["source"], job["state"]) for job in listed] == [(OTHER, "running"),
                                                                 (SECOND, "queued")]
    assert listed[0]["kind"] == "training" and listed[0]["id"] == first["id"]

    made = local.post(TOKEN_URL).get_json()
    assert made["cancelled"] == 1
    waiting = _job(open_app, second["id"])
    assert waiting.state == "cancelled" and waiting.started is None
    assert "the access token was changed" in waiting.error["message"]
    assert [job["id"] for job in local.get(JOBS_URL).get_json()["jobs"]] == [first["id"]]
    refused = running.get(first["url"])
    assert refused.status_code == 401 and _code(refused) == "LOCKED", "nor follow it"

    release.set()
    assert _ended(_job(open_app, first["id"])).state == "done", "a running one goes on"
    assert wait_for(local, mine["url"])["state"] == "done", "this computer's job is untouched"
    assert waiting.started is None, "it never started"


def test_a_running_job_from_another_computer_can_be_stopped_here(open_app, gate, fixtures_dir):
    entered, release = gate
    first = _train(_unlocked(open_app), fixtures_dir)
    assert entered.wait(30)
    local = open_app.test_client()
    local.post(TOKEN_URL)
    stopped = local.delete(f"{JOBS_URL}/{first['id']}")
    assert stopped.status_code == 202 and stopped.get_json()["state"] == "cancelling"
    release.set()
    assert _ended(_job(open_app, first["id"])).state == "cancelled"
    gone = local.delete(f"{JOBS_URL}/{first['id']}")
    assert gone.status_code == 404 and _code(gone) == "NOT_FOUND", "it has ended"


def test_this_computers_jobs_are_not_listed(open_app, gate, fixtures_dir):
    entered, release = gate
    local = open_app.test_client()
    mine = _train(local, fixtures_dir)
    assert entered.wait(30)
    assert local.get(JOBS_URL).get_json()["jobs"] == []
    refused = local.delete(f"{JOBS_URL}/{mine['id']}")
    assert refused.status_code == 404 and _code(refused) == "NOT_FOUND"
    release.set()
    assert wait_for(local, mine["url"])["state"] == "done"


def test_a_remote_scan_records_where_it_came_from(open_app, access, monkeypatch):
    rec = fake_network(monkeypatch)
    monkeypatch.setattr(scan_routes, "load_api_keys", lambda: dict(rec.keys))
    other = _unlocked(open_app)
    submitted = other.post("/api/v1/jobs/scan",
                           json={"target": "10.0.0.5", "ports": "22", "authorized": True})
    assert submitted.status_code == 202
    job = _job(open_app, submitted.get_json()["id"])
    assert (job.origin.source, job.origin.unlock) == (OTHER, access.run_id)
    assert wait_for(other, submitted.get_json()["url"])["state"] == "done"


def test_a_job_asked_for_under_an_old_unlock_never_starts():
    """Even one the sweep missed: it was queued after the new token was made."""
    access = access_service.Access()
    runner = jobs.JobRunner(jobs.TRAINING, refusal=access.refusal)
    old = jobs.Origin(OTHER, access.run_id)
    access.renew()
    ran, ended = [], threading.Event()
    late = jobs.Job(kind=jobs.TRAINING, work=lambda job: ran.append(job) or {},
                    on_end=ended.set, max_seconds=60, steps=lambda done, total: "", origin=old)
    runner.submit(late)
    assert ended.wait(10) and late.state == "cancelled" and late.started is None and ran == []
    assert late.error["code"] == "CANCELLED"
    local = jobs.Job(kind=jobs.TRAINING, work=lambda job: {}, on_end=lambda: None,
                     max_seconds=60, steps=lambda done, total: "")
    runner.submit(local)
    assert _ended(local, 10).state == "done", "this computer's job has no unlock to check"


def test_cancel_waiting_never_stops_a_running_job():
    runner = jobs.JobRunner(jobs.TRAINING)
    entered, release = threading.Event(), threading.Event()

    def work(job):
        entered.set()
        assert release.wait(10)
        return {}

    job = jobs.Job(kind=jobs.TRAINING, work=work, on_end=lambda: None, max_seconds=60,
                   steps=lambda done, total: "")
    runner.submit(job)
    assert entered.wait(10)
    assert runner.cancel_waiting(job, "not now") is False
    assert not job.cancel_requested.is_set()
    release.set()
    assert _ended(job, 10).state == "done"


# --- The checks of every request, from another computer -----------------------------------

def test_the_checks_of_every_request_hold_for_other_computers(open_app):
    other = _unlocked(open_app)
    no_token = other.post(TRAIN_URL, json={}, headers={"X-CSRF-Token": "wrong"})
    assert no_token.status_code == 403 and _code(no_token) == "CSRF_TOKEN"
    cross_site = other.post(TRAIN_URL, json={}, headers={"Origin": "http://evil.example"})
    assert cross_site.status_code == 403 and _code(cross_site) == "CROSS_SITE"
    for test_client in (other, _other(open_app, SECOND)):  # unlocked, and not
        rebound = test_client.post(TRAIN_URL, json={}, headers={"Host": "evil.example:5000",
                                                                "X-CSRF-Token": "any"})
        assert rebound.status_code == 400 and _code(rebound) == "BAD_HOST"
