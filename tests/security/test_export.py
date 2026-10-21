"""S4: export names are restricted and exports never overwrite (active again from 0.4g)."""

import pytest

from toolkit import bundle_files
from toolkit.web.security import safe_model_basename

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
    response = client.post("/configure", json={
        "input_features": [c for c in columns if c != TARGET], "target": TARGET,
        "selected_models": ["decision_tree"], "balancing_method": "none"})
    assert response.status_code == 200, response.get_json()


def test_export_never_overwrites(client, trained):
    first = client.post("/export_model", json={"model_name": "first"}).get_json()
    path = bundle_files.bundles_dir() / first["file"]
    content, modified = path.read_bytes(), path.stat().st_mtime_ns
    second = client.post("/export_model", json={"model_name": "second"}).get_json()
    assert second["file"] == first["file"] and second["sha256"] == first["sha256"]
    assert path.read_bytes() == content and path.stat().st_mtime_ns == modified
    assert [p.name for p in bundle_files.bundles_dir().iterdir()] == [first["file"]]
    download = client.get("/download_bundle")
    assert "second.bundle" in download.headers["Content-Disposition"]
    assert download.data == content


@pytest.mark.parametrize("bad", ["../../evil", "a/b", ""])
def test_bad_export_name_refused_over_http(client, trained, bad):
    response = client.post("/export_model", json={"model_name": bad})
    assert response.status_code == 400
    assert not bundle_files.bundles_dir().exists()


def test_export_needs_a_trained_model(client):
    response = client.post("/export_model", json={"model_name": "x"})
    assert response.status_code == 400
    assert "train" in response.get_json()["error"].lower()
