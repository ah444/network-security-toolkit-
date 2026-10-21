"""S4: export names are restricted and exports never overwrite (active again from 0.4g)."""

import pytest

from toolkit import bundle_files
from toolkit.web.security import safe_model_basename
from web_jobs import train_and_wait

TARGET = "Label"


@pytest.mark.parametrize("bad", ["../../evil", "a/b", r"a\b", "", "   ", "x" * 81, "na:me", ".."])
def test_bad_export_name_refused(bad):
    with pytest.raises(ValueError):
        safe_model_basename(bad)


@pytest.mark.parametrize("good", ["my_model", "IDS-v5", "run42"])
def test_good_export_name_kept(good):
    assert safe_model_basename(good) == good


@pytest.fixture
def trained(client, upload, fixtures_dir):
    columns = upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    response = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET,
        "models": ["decision_tree"], "balancing": "none"})
    assert response.status_code == 200, response.get_json()


def test_export_never_overwrites(client, trained):
    first = client.post("/api/v1/run/export", json={"name": "first"}).get_json()
    path = bundle_files.bundles_dir() / first["file"]
    content, modified = path.read_bytes(), path.stat().st_mtime_ns
    second = client.post("/api/v1/run/export", json={"name": "second"}).get_json()
    assert second["file"] == first["file"] and second["sha256"] == first["sha256"]
    assert path.read_bytes() == content and path.stat().st_mtime_ns == modified
    assert [p.name for p in bundle_files.bundles_dir().iterdir()] == [first["file"]]
    download = client.get("/api/v1/run/bundle")
    assert "second.bundle" in download.headers["Content-Disposition"]
    assert download.data == content


@pytest.mark.parametrize("bad", ["../../evil", "a/b", ""])
def test_bad_export_name_refused_over_http(client, trained, bad):
    response = client.post("/api/v1/run/export", json={"name": bad})
    assert response.status_code == 400
    assert not bundle_files.bundles_dir().exists()


def test_export_needs_a_trained_model(client):
    response = client.post("/api/v1/run/export", json={"name": "x"})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "NOT_TRAINED"
    assert "train" in response.get_json()["error"]["message"].lower()
