"""v5.3 step HR4a (RD4: IA8): deleting a saved run from the Saved Runs page.
``GET /api/v1/runs/<run id>/files`` names every file the delete removes, in the order they go;
``POST /api/v1/runs/<run id>/delete`` removes them, the bundle only when asked. A file that
can't be removed is a 409 that names it, and the run stays listed. A workspace holding the
run's export forgets it once its bundle is gone, so Export saves it again."""

import pytest

from toolkit.core.data.ids.bundle_files import bundles_dir
from toolkit.core.data.ids.run_files import runs_dir
from toolkit.core.services.ids import run_removal
from web_jobs import train_and_wait

CIC = "cic_ddos_sample.csv"
TARGET = "Label"


@pytest.fixture
def exported(client, upload, fixtures_dir):
    """A run trained and exported through the routes, as the page does: its run id."""
    columns = upload(fixtures_dir / CIC).get_json()["columns"]
    trained = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET, "balancing": "none",
        "test_size": 0.3, "models": ["decision_tree"]})
    assert trained.status_code == 200, trained.get_json()
    return client.post("/api/v1/run/export", json={"name": "first"}).get_json()["run_id"]


def _files(run_id):
    return [runs_dir() / f"{run_id}.json", bundles_dir() / f"{run_id}.bundle"]


def _delete(client, run_id, **body):
    return client.post(f"/api/v1/runs/{run_id}/delete", json=body)


def test_the_files_route_names_every_file_in_order(client, exported):
    answer = client.get(f"/api/v1/runs/{exported}/files")
    assert answer.status_code == 200
    files = answer.get_json()["files"]
    assert [(f["kind"], f["name"]) for f in files] == [
        ("bundle", f"{exported}.bundle"), ("record", f"{exported}.json")]
    assert [f["size"] for f in files] == [path.stat().st_size for path in reversed(_files(exported))]


def test_the_bundle_is_kept_unless_asked(client, exported):
    answer = _delete(client, exported)
    assert answer.status_code == 200, answer.get_json()
    body = answer.get_json()
    assert [f["kind"] for f in body["removed"]] == ["record"]
    assert body["kept"]["name"] == f"{exported}.bundle"
    record, bundle = _files(exported)
    assert not record.exists() and bundle.exists()
    assert client.get("/api/v1/runs").get_json()["runs"] == []
    assert client.get("/api/v1/run/bundle").status_code == 200, "the workspace's bundle is there"


def test_with_its_bundle_and_the_workspace_exports_it_again(client, exported):
    answer = _delete(client, exported, with_bundle=True)
    assert answer.status_code == 200
    assert [f["kind"] for f in answer.get_json()["removed"]] == ["bundle", "record"]
    assert answer.get_json()["kept"] is None
    assert not any(path.exists() for path in _files(exported))
    assert client.get("/api/v1/run/bundle").status_code == 404, "the export is forgotten"
    again = client.post("/api/v1/run/export", json={"name": "again"})
    assert again.status_code == 200 and again.get_json()["run_id"] == exported
    assert all(path.exists() for path in _files(exported))


def test_a_file_that_cant_be_removed_is_a_409_and_the_run_stays(client, exported, monkeypatch):
    real = run_removal.remove_file

    def remove(path):
        if path.suffix == ".bundle":
            raise PermissionError(13, "used by another process", str(path))
        real(path)
    monkeypatch.setattr(run_removal, "remove_file", remove)
    answer = _delete(client, exported, with_bundle=True)
    assert answer.status_code == 409
    error = answer.get_json()["error"]
    assert error["code"] == "NOT_REMOVED" and f"{exported}.bundle" in error["message"]
    assert str(bundles_dir()) not in error["message"]
    assert [r["run_id"] for r in client.get("/api/v1/runs").get_json()["runs"]] == [exported]
    assert client.get("/api/v1/run/bundle").status_code == 200, "nothing forgotten"


def test_no_such_run_and_a_bad_id(client):
    missing = "20261030T120000Z-0000000a"
    for answer in (client.get(f"/api/v1/runs/{missing}/files"), _delete(client, missing)):
        assert answer.status_code == 404 and answer.get_json()["error"]["code"] == "NOT_FOUND"
    for answer in (client.get("/api/v1/runs/not-a-run/files"), _delete(client, "not-a-run")):
        assert answer.status_code == 400
        assert answer.get_json()["error"]["message"] == "That is not a run id."


def test_with_bundle_must_be_true_or_false(client, exported):
    answer = _delete(client, exported, with_bundle="yes")
    assert answer.status_code == 400
    assert answer.get_json()["error"]["message"] == "'with_bundle' must be true or false."
    assert all(path.exists() for path in _files(exported))
