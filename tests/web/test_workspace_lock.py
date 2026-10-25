"""v5.1 step 1.2c: a lock per workspace, a lease while training, workspaces only on upload.

Concurrency is tested with blocking hooks (threading.Event): the Flask test client sends
requests one after another, so a request is held in a background thread while others run;
a training job is held inside training in the app's worker thread.
"""

import io
import threading

import pytest

from toolkit.web.services import training
from toolkit.web.services.workspace import Workspace
from web_jobs import TRAIN_URL, wait_for

TARGET = "Label"
CSV = b"a,b,Label\n1,2,X\n3,4,Y\n5,6,X\n7,8,Y\n"


def _upload(test_client, payload=CSV):
    return test_client.post("/api/v1/dataset", data={"dataset": (io.BytesIO(payload), "d.csv")},
                            content_type="multipart/form-data")


def _in_background(web_app, test_client, method, url, **kwargs):
    """A request from a background thread in test_client's session; returns (thread, result)."""
    background = web_app.test_client()
    name = web_app.config["SESSION_COOKIE_NAME"]
    cookie = test_client.get_cookie(name)
    if cookie is not None:
        background.set_cookie(name, cookie.value)
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault(
        "response", getattr(background, method)(url, **kwargs)))
    thread.start()
    return thread, result


@pytest.fixture
def held_training(monkeypatch):
    """Trainings started from background threads stop inside training until released."""
    entered, release = threading.Event(), threading.Event()
    real_train = training.train_run

    def held(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            entered.set()
            assert release.wait(30), "test never released the held training"
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_run", held)
    return entered, release


@pytest.fixture
def training_now(web_app, client, upload, fixtures_dir, held_training):
    """A training job running in client's session (held until the test ends)."""
    columns = upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    entered, release = held_training
    body = {"features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing": "none", "models": ["decision_tree"]}
    submitted = client.post(TRAIN_URL, json=body)
    assert submitted.status_code == 202, submitted.get_json()
    assert entered.wait(30), "the training never started"
    yield columns
    release.set()
    assert wait_for(client, submitted.get_json()["url"])["state"] == "done"


def test_reads_create_no_workspace(client, workspaces):
    for url in ("/api/v1/dataset/preview", "/api/v1/dataset/classes?target=x",
                "/api/v1/dataset/distribution?target=x", "/api/v1/run/confusion",
                "/api/v1/run/bundle"):
        response = client.get(url)
        assert response.status_code in (400, 404) and "Set-Cookie" not in response.headers, url
    for url, body in (("/api/v1/run/predict", {"values": {"a": 1}}),
                      ("/api/v1/bundle/predict", {"values": {"a": 1}}),
                      ("/api/v1/run/export", {"name": "x"}),
                      ("/api/v1/jobs/training", {"target": "x", "features": ["a"]}),
                      ("/api/v1/dataset/filter", {"target": "x", "remove": ["y"]})):
        response = client.post(url, json=body)
        assert response.status_code == 400 and "Set-Cookie" not in response.headers, url
    assert client.delete("/api/v1/workspace").status_code == 200
    assert len(workspaces) == 0


def test_upload_creates_the_workspace(client, workspaces):
    response = _upload(client)
    assert response.status_code == 200 and "Set-Cookie" in response.headers
    assert len(workspaces) == 1


def test_dataset_changes_refused_while_training(client, training_now):
    for response in (_upload(client),
                     client.post("/api/v1/dataset/filter",
                                 json={"target": TARGET, "remove": ["DDoS"]}),
                     client.delete("/api/v1/workspace")):
        assert response.status_code == 409
        assert response.get_json()["error"]["code"] == "BUSY"
    preview = client.get("/api/v1/dataset/preview", query_string={"rows": 3})
    assert preview.status_code == 200, "reading stays possible while training"


def test_dataset_changes_allowed_after_training(client, workspaces, training_now, held_training):
    (ws,) = workspaces.values()
    _, release = held_training
    release.set()
    assert ws.training.acquire(timeout=60), "the training never ended"  # waits for its end
    ws.training.release()
    assert _upload(client).status_code == 200


def test_leased_workspace_is_not_dropped(web_app, workspaces, training_now):
    idle = workspaces.idle_seconds
    workspaces.idle_seconds = -1  # restored before the job ends (its workspace is then idle)
    try:
        assert _upload(web_app.test_client()).status_code == 200  # drops idle ones first
        assert len(workspaces) == 2, "the training's workspace was dropped"
    finally:
        workspaces.idle_seconds = idle


def test_lease_counts():
    ws = Workspace()
    assert not ws.leased
    with ws.lease(), ws.lease():
        assert ws.leased
    assert not ws.leased


def test_requests_of_one_session_take_turns(web_app, client, workspaces):
    _upload(client)
    (ws,) = workspaces.values()
    with ws.lock:
        thread, result = _in_background(web_app, client, "get", "/api/v1/dataset/preview")
        thread.join(0.5)
        assert thread.is_alive(), "the request ran while the workspace was locked"
    thread.join(30)
    assert result["response"].status_code == 200
