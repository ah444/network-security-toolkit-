"""v5.3 step HR4b: the storage report and registering a bundle again, on the Saved Runs page.
``GET /api/v1/storage`` only reads; ``POST /api/v1/bundles/<run id>/register`` adds the
record a kept bundle carries, never replacing one."""

import pytest

from toolkit.core.data.ids.bundle_files import bundles_dir
from toolkit.core.data.ids.run_files import runs_dir
from web_jobs import train_and_wait

CIC = "cic_ddos_sample.csv"
TARGET = "Label"


@pytest.fixture
def kept(client, upload, fixtures_dir):
    """A run exported, then deleted with its bundle kept: its run id and record's bytes."""
    columns = upload(fixtures_dir / CIC).get_json()["columns"]
    trained = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET, "balancing": "none",
        "test_size": 0.3, "models": ["decision_tree"]})
    assert trained.status_code == 200, trained.get_json()
    run_id = client.post("/api/v1/run/export", json={"name": "first"}).get_json()["run_id"]
    record = (runs_dir() / f"{run_id}.json").read_bytes()
    assert client.post(f"/api/v1/runs/{run_id}/delete", json={}).status_code == 200
    return run_id, record


def _kinds(client):
    return {item["kind"]: item for item in client.get("/api/v1/storage").get_json()["kinds"]}


def test_the_report_names_the_kept_bundle(client, kept):
    answer = client.get("/api/v1/storage")
    assert answer.status_code == 200
    body = answer.get_json()
    assert set(body) == {"kinds", "total_bytes", "free_bytes", "warn_bytes", "floor_bytes",
                         "warnings"}
    kinds = _kinds(client)
    assert kinds["unregistered_bundles"]["files"] == [f"{kept[0]}.bundle"]
    assert kinds["records"]["count"] == 0
    assert all("\\" not in name and "/" not in name
               for item in kinds.values() for name in item["files"]), "names, never paths"


def test_register_brings_the_run_back(client, kept):
    run_id, record = kept
    answer = client.post(f"/api/v1/bundles/{run_id}/register")
    assert answer.status_code == 200, answer.get_json()
    assert answer.get_json() == {"run_id": run_id, "record": f"{run_id}.json"}
    assert (runs_dir() / f"{run_id}.json").read_bytes() == record
    assert [r["run_id"] for r in client.get("/api/v1/runs").get_json()["runs"]] == [run_id]
    assert _kinds(client)["unregistered_bundles"]["count"] == 0
    again = client.post(f"/api/v1/bundles/{run_id}/register")
    assert again.status_code == 409 and "already saved" in again.get_json()["error"]["message"]


def test_a_changed_bundle_is_refused(client, kept):
    path = bundles_dir() / f"{kept[0]}.bundle"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.unlink()
    path.write_bytes(bytes(data))
    answer = client.post(f"/api/v1/bundles/{kept[0]}/register")
    assert answer.status_code == 400 and answer.get_json()["error"]["code"] == "BAD_SIGNATURE"
    assert not (runs_dir() / f"{kept[0]}.json").exists()


def test_no_bundle_and_a_bad_id(client):
    missing = client.post("/api/v1/bundles/20261030T120000Z-0000000a/register")
    assert missing.status_code == 404 and missing.get_json()["error"]["code"] == "NOT_FOUND"
    bad = client.post("/api/v1/bundles/not-a-run/register")
    assert bad.status_code == 400 and bad.get_json()["error"]["message"] == "That is not a run id."
