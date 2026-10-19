"""Step 0.8: one training at a time per workspace; a second request gets 409 (D8)."""

import threading

import pytest

TARGET = " Label"


@pytest.fixture
def columns(upload, fixtures_dir):
    return upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]


def _train_body(columns):
    return {"input_features": [c for c in columns if c != TARGET], "target": TARGET,
            "balancing_method": "none", "test_size": 0.3, "selected_models": ["decision_tree"]}


@pytest.fixture
def held_training(app_module, monkeypatch):
    """Training requests from background threads stop inside training until released."""
    entered, release = threading.Event(), threading.Event()
    real_split = app_module.train_test_split

    def split(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            entered.set()
            assert release.wait(30), "test never released the held training"
        return real_split(*args, **kwargs)

    monkeypatch.setattr(app_module, "train_test_split", split)
    return entered, release


def _start(app_module, test_client, body):
    """POST /configure from a background thread, in the same session as test_client.

    The thread gets its own client (a client used in a `with` block can't be shared
    across threads) carrying a copy of the session cookie.
    """
    background = app_module.app.test_client()
    name = app_module.app.config["SESSION_COOKIE_NAME"]
    cookie = test_client.get_cookie(name)
    if cookie is not None:
        background.set_cookie(name, cookie.value)
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("response", background.post("/configure", json=body)))
    thread.start()
    return thread, result


def test_second_training_returns_409(app_module, client, columns, held_training):
    entered, release = held_training
    thread, first = _start(app_module, client, _train_body(columns))
    try:
        assert entered.wait(30), "first training never started"
        second = client.post("/configure", json=_train_body(columns))
        assert second.status_code == 409
        assert "already" in second.get_json()["error"].lower()
    finally:
        release.set()
        thread.join(60)
    assert first["response"].status_code == 200
    assert client.post("/configure", json=_train_body(columns)).status_code == 200, \
        "busy flag must be cleared after training"


def test_busy_flag_cleared_after_failure(app_module, client, columns, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    real_split = app_module.train_test_split
    monkeypatch.setattr(app_module, "train_test_split", broken)
    with pytest.raises(RuntimeError):
        client.post("/configure", json=_train_body(columns))
    monkeypatch.setattr(app_module, "train_test_split", real_split)
    assert client.post("/configure", json=_train_body(columns)).status_code == 200


def test_other_workspace_not_blocked(app_module, client, columns, held_training, fixtures_dir):
    entered, release = held_training
    thread, _ = _start(app_module, client, _train_body(columns))
    try:
        assert entered.wait(30)
        other = app_module.app.test_client()
        path = fixtures_dir / "cic_ddos_sample.csv"
        with path.open("rb") as handle:
            other.post("/upload", data={"dataset": (handle, path.name)},
                       content_type="multipart/form-data")
        assert other.post("/configure", json=_train_body(columns)).status_code == 200
    finally:
        release.set()
        thread.join(60)
