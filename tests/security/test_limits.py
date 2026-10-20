"""S9: upload size limit, JSON-only bodies, clamped inputs."""

import io

import pytest

JSON_ROUTES = ["/preview_data", "/filter_target_values", "/get_target_values",
               "/analyze_class_distribution", "/apply_smote", "/recommend_features",
               "/configure", "/get_confusion_matrix", "/predict", "/predict_loaded",
               "/export_model"]


def test_oversized_upload_413(app_module, client):
    app_module.app.config["MAX_CONTENT_LENGTH"] = 100
    payload = b"a,b,Label\n" + b"1,2,X\n" * 100
    response = client.post("/upload", data={"dataset": (io.BytesIO(payload), "big.csv")},
                           content_type="multipart/form-data")
    assert response.status_code == 413
    assert "too large" in response.get_json()["error"].lower()


def test_upload_limit_is_configured_by_default(app_module):
    assert app_module.app.config["MAX_CONTENT_LENGTH"] >= 1024 * 1024 * 1024  # big datasets ok


@pytest.mark.parametrize("route", JSON_ROUTES)
def test_non_json_400(client, route):
    response = client.post(route, data="target=x", content_type="text/plain")
    assert response.status_code == 400
    assert "json" in response.get_json()["error"].lower()


def test_preview_rows_clamped(client, upload, fixtures_dir):
    upload(fixtures_dir / "cic_ddos_sample.csv")
    big = client.post("/preview_data", json={"num_rows": 10**9}).get_json()
    assert len(big["preview"]) == 100
    small = client.post("/preview_data", json={"num_rows": -5}).get_json()
    assert len(small["preview"]) == 1
    bad = client.post("/preview_data", json={"num_rows": "lots"})
    assert bad.status_code == 400


def test_recommend_k_clamped(client, upload, fixtures_dir):
    upload(fixtures_dir / "cic_ddos_sample.csv")
    response = client.post("/recommend_features", json={"target": "Label", "k": 10**9})
    assert response.status_code == 200
    assert len(response.get_json()["recommended_features"]) <= 14
    bad = client.post("/recommend_features", json={"target": "Label", "k": "many"})
    assert bad.status_code == 400
