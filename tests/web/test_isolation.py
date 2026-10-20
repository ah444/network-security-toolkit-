"""Step 0.8: one workspace per browser session instead of three shared globals (S7, R4)."""

import pytest

TARGET = "Label"  # stripped by the v5 loader (B17)


def _upload(test_client, path):
    with path.open("rb") as handle:
        return test_client.post("/upload", data={"dataset": (handle, path.name)},
                                content_type="multipart/form-data")


def _train(test_client, columns):
    features = [c for c in columns if c != TARGET]
    return test_client.post("/configure", json={
        "input_features": features, "target": TARGET, "balancing_method": "none",
        "test_size": 0.3, "selected_models": ["decision_tree"],
    })


@pytest.fixture
def two_clients(app_module):
    # Plain clients: two nested `with` clients clash over Flask's preserved contexts.
    return app_module.app.test_client(), app_module.app.test_client()


def test_no_shared_globals_left(app_module):
    for name in ("data_store", "model_config", "loaded_model_config"):
        assert not hasattr(app_module, name), f"module-level '{name}' is shared by every user"


def test_two_sessions_isolated(two_clients, fixtures_dir):
    alice, bob = two_clients
    columns = _upload(alice, fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    assert alice.post("/preview_data", json={"num_rows": 5}).status_code == 200
    assert bob.post("/preview_data", json={"num_rows": 5}).status_code == 400, \
        "Bob sees Alice's dataset"

    assert _train(alice, columns).status_code == 200
    values = {c: "0" for c in columns if c != TARGET}
    bob_predict = bob.post("/predict", json={"values": values, "algorithm": "decision_tree"})
    assert bob_predict.status_code == 400, "Bob can use Alice's trained model"

    _upload(bob, fixtures_dir / "label_leak.csv")
    alice_preview = alice.post("/preview_data", json={"num_rows": 5}).get_json()
    assert TARGET in alice_preview["columns"], "Bob's upload replaced Alice's dataset"


def test_clear_workspace_frees_and_reusable(app_module, client, fixtures_dir):
    _upload(client, fixtures_dir / "cic_ddos_sample.csv")
    assert len(app_module._workspaces) == 1

    cleared = client.delete("/workspace")
    assert cleared.status_code == 200
    assert len(app_module._workspaces) == 0, "the workspace's data must be released"
    assert client.post("/preview_data", json={"num_rows": 5}).status_code == 400

    again = _upload(client, fixtures_dir / "cic_ddos_sample.csv")
    assert again.status_code == 200
    assert client.post("/preview_data", json={"num_rows": 5}).status_code == 200


def test_clear_workspace_blocked_cross_site(client):
    response = client.delete("/workspace", headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_idle_workspaces_are_dropped(app_module, two_clients, fixtures_dir, monkeypatch):
    alice, bob = two_clients
    _upload(alice, fixtures_dir / "cic_ddos_sample.csv")
    monkeypatch.setattr(app_module, "WORKSPACE_IDLE_SECONDS", -1)
    bob.post("/preview_data", json={"num_rows": 5})
    assert len(app_module._workspaces) == 1, "Alice's idle workspace should have been dropped"


def test_home_page_creates_no_workspace(app_module, client):
    assert client.get("/").status_code == 200
    assert client.get("/dynamic_ids").status_code == 200
    assert len(app_module._workspaces) == 0
