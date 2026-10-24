"""v5.1 step 1.2c: memory caps (least recently used first, never a leased workspace), the
idle sweep without requests, and leftover uploads removed at startup (R4, S11, D22)."""

import io
import os
import time

import pytest

from toolkit.data.settings import WebConfig
from toolkit.data.web.uploads import remove_stale_uploads
from toolkit.web.workspace import Workspace, WorkspaceStore, start_idle_sweep

CSV = b"a,b,Label\n" + b"".join(b"%d,%d,%s\n" % (i, i * 2, b"XY"[i % 2:i % 2 + 1])
                                 for i in range(60))


def _upload(test_client, payload=CSV):
    return test_client.post("/api/v1/dataset", data={"dataset": (io.BytesIO(payload), "d.csv")},
                            content_type="multipart/form-data")


def _has_data(test_client):
    return test_client.get("/api/v1/dataset/preview").status_code == 200


def test_upload_reports_memory(client, workspaces):
    dataset = _upload(client).get_json()["dataset"]
    (ws,) = workspaces.values()
    assert ws.data_bytes > 0 and dataset["memory_mb"] == round(ws.data_bytes / 2**20, 1)


def test_workspace_cap_drops_least_recently_used(make_app):
    app = make_app(WebConfig(max_workspaces=2))
    first, second, third = app.test_client(), app.test_client(), app.test_client()
    _upload(first)
    _upload(second)
    assert _has_data(first)  # first is now used more recently than second
    assert _upload(third).status_code == 200
    assert len(app.extensions["toolkit"].workspaces) == 2
    assert _has_data(first) and _has_data(third)
    assert not _has_data(second), "the least recently used workspace should have been dropped"


def test_workspace_cap_never_drops_a_leased_workspace(make_app):
    app = make_app(WebConfig(max_workspaces=1))
    first, second = app.test_client(), app.test_client()
    _upload(first)
    (ws,) = app.extensions["toolkit"].workspaces.values()
    with ws.lease():
        refused = _upload(second)
    assert refused.status_code == 503 and refused.get_json()["error"]["code"] == "BUSY"
    assert _has_data(first)


def test_data_cap_drops_others_first(make_app, monkeypatch):
    app = make_app()
    store = app.extensions["toolkit"].workspaces
    first, second = app.test_client(), app.test_client()
    _upload(first)
    (ws,) = store.values()
    monkeypatch.setattr(store, "max_data_bytes", int(ws.data_bytes * 1.5))  # room for one
    assert _upload(second).status_code == 200
    assert not _has_data(first) and _has_data(second)
    assert sum(w.data_bytes for w in store.values()) <= store.max_data_bytes


def test_data_cap_refuses_what_never_fits(client, workspaces, monkeypatch):
    monkeypatch.setattr(workspaces, "max_data_bytes", 100)
    response = _upload(client)
    error = response.get_json()["error"]
    assert response.status_code == 413 and error["code"] == "DATA_LIMIT"
    assert "TOOLKIT_MAX_DATA_MB" in error["message"]
    assert not _has_data(client), "a refused dataset must not be kept"


def test_data_cap_refuses_when_the_others_are_training(make_app, monkeypatch):
    app = make_app()
    store = app.extensions["toolkit"].workspaces
    first, second = app.test_client(), app.test_client()
    _upload(first)
    (ws,) = store.values()
    monkeypatch.setattr(store, "max_data_bytes", int(ws.data_bytes * 1.5))
    with ws.lease():
        response = _upload(second)
    assert response.status_code == 413 and "busy training" in response.get_json()["error"]["message"]
    assert _has_data(first)


def test_filter_lowers_the_counted_memory(client, workspaces):
    _upload(client)
    (ws,) = workspaces.values()
    before = ws.data_bytes
    assert client.post("/api/v1/dataset/filter",
                       json={"target": "Label", "remove": ["X"]}).status_code == 200
    assert 0 < ws.data_bytes < before


def test_drop_idle_skips_leased():
    store = WorkspaceStore(idle_seconds=-1)
    store._workspaces.update({"a": Workspace(), "b": Workspace()})
    with store._workspaces["a"].lease():
        assert store.drop_idle() == 1
    assert list(store._workspaces) == ["a"]


def test_idle_sweep_runs_without_requests():
    store = WorkspaceStore(idle_seconds=-1)
    store._workspaces["a"] = Workspace()
    stop = start_idle_sweep(store, interval=0.05)
    try:
        deadline = time.monotonic() + 5
        while len(store) and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        stop.set()
    assert len(store) == 0


def test_stale_uploads_removed_new_ones_kept(tmp_path):
    old, new = tmp_path / "old_upload.csv", tmp_path / "new_upload.csv"
    old.write_text("x", encoding="utf-8")
    new.write_text("x", encoding="utf-8")
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    assert remove_stale_uploads(tmp_path) == 1
    assert not old.exists() and new.exists()


@pytest.mark.parametrize("age, kept", [(7200, False), (60, True)])
def test_app_start_sweeps_leftover_uploads(make_app, tmp_path, monkeypatch, age, kept):
    uploads = tmp_path / "instance" / "uploads"
    uploads.mkdir(parents=True)
    leftover = uploads / "crash_leftover.csv"
    leftover.write_text("x", encoding="utf-8")
    os.utime(leftover, (time.time() - age, time.time() - age))
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path / "instance"))
    make_app()
    assert leftover.exists() is kept
