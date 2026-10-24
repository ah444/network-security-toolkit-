"""Step 0.8: one workspace per browser session instead of three shared globals (S7, R4)."""

import importlib

import pytest

import toolkit.web as web
from web_jobs import train_and_wait

TARGET = "Label"  # stripped by the v5 loader (B17)
WEB_MODULES = (*(f"toolkit.web.{name}" for name in (
    "api", "bundles", "datasets", "pages", "schemas", "security", "state", "training",
    "workspace")), *(f"toolkit.services.web.{name}" for name in (
    "datasets", "errors", "jobs", "prediction", "training", "workspace")))


def _upload(test_client, path):
    with path.open("rb") as handle:
        return test_client.post("/api/v1/dataset", data={"dataset": (handle, path.name)},
                                content_type="multipart/form-data")


def _train(test_client, columns):
    features = [c for c in columns if c != TARGET]
    return train_and_wait(test_client, {
        "features": features, "target": TARGET, "balancing": "none",
        "test_size": 0.3, "models": ["decision_tree"],
    })


@pytest.fixture
def two_clients(web_app):
    # Plain clients: two nested `with` clients clash over Flask's preserved contexts.
    return web_app.test_client(), web_app.test_client()


def test_no_shared_globals_left():
    for module in (web, *(importlib.import_module(name) for name in WEB_MODULES)):
        for name in ("data_store", "model_config", "loaded_model_config", "_workspaces"):
            assert not hasattr(module, name), f"module-level '{name}' is shared by every user"


def test_two_apps_share_nothing(make_app, fixtures_dir):
    first, second = make_app(), make_app()
    _upload(first.test_client(), fixtures_dir / "cic_ddos_sample.csv")
    assert len(first.extensions["toolkit"].workspaces) == 1
    assert len(second.extensions["toolkit"].workspaces) == 0


def test_two_sessions_isolated(two_clients, fixtures_dir):
    alice, bob = two_clients
    columns = _upload(alice, fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    assert alice.get("/api/v1/dataset/preview", query_string={"rows": 5}).status_code == 200
    assert bob.get("/api/v1/dataset/preview", query_string={"rows": 5}).status_code == 400, \
        "Bob sees Alice's dataset"

    assert _train(alice, columns).status_code == 200
    values = {c: "0" for c in columns if c != TARGET}
    bob_predict = bob.post("/api/v1/run/predict", json={"values": values, "model": "decision_tree"})
    assert bob_predict.status_code == 400, "Bob can use Alice's trained model"

    _upload(bob, fixtures_dir / "label_leak.csv")
    alice_preview = alice.get("/api/v1/dataset/preview", query_string={"rows": 5}).get_json()
    assert TARGET in alice_preview["columns"], "Bob's upload replaced Alice's dataset"


def test_clear_workspace_frees_and_reusable(workspaces, client, fixtures_dir):
    _upload(client, fixtures_dir / "cic_ddos_sample.csv")
    assert len(workspaces) == 1

    cleared = client.delete("/api/v1/workspace")
    assert cleared.status_code == 200
    assert len(workspaces) == 0, "the workspace's data must be released"
    assert client.get("/api/v1/dataset/preview", query_string={"rows": 5}).status_code == 400

    again = _upload(client, fixtures_dir / "cic_ddos_sample.csv")
    assert again.status_code == 200
    assert client.get("/api/v1/dataset/preview", query_string={"rows": 5}).status_code == 200


def test_clear_workspace_blocked_cross_site(client):
    response = client.delete("/api/v1/workspace", headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_idle_workspaces_are_dropped(workspaces, two_clients, fixtures_dir, monkeypatch):
    alice, bob = two_clients
    _upload(alice, fixtures_dir / "cic_ddos_sample.csv")
    (alice_ws,) = workspaces.values()
    monkeypatch.setattr(workspaces, "idle_seconds", -1)
    _upload(bob, fixtures_dir / "cic_ddos_sample.csv")
    assert len(workspaces) == 1, "Alice's idle workspace should have been dropped"
    assert workspaces.values()[0] is not alice_ws


def test_home_page_creates_no_workspace(workspaces, client):
    assert client.get("/").status_code == 200
    assert client.get("/dynamic_ids").status_code == 200
    assert len(workspaces) == 0
