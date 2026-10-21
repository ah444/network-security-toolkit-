"""S9: upload size limit, JSON-only bodies, clamped inputs."""

import io

import pytest

# Every POST route with a JSON body (GET routes read the query string; uploads are multipart).
JSON_ROUTES = ["/api/v1/dataset/filter", "/api/v1/dataset/balancing-preview",
               "/api/v1/dataset/feature-scores", "/api/v1/run", "/api/v1/run/predict",
               "/api/v1/bundle/predict", "/api/v1/run/export"]


def test_oversized_upload_413(web_app, client):
    web_app.config["MAX_CONTENT_LENGTH"] = 100
    payload = b"a,b,Label\n" + b"1,2,X\n" * 100
    response = client.post("/api/v1/dataset", data={"dataset": (io.BytesIO(payload), "big.csv")},
                           content_type="multipart/form-data")
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "TOO_LARGE"
    assert "too large" in response.get_json()["error"]["message"].lower()


def test_upload_limit_is_configured_by_default(web_app):
    assert web_app.config["MAX_CONTENT_LENGTH"] >= 1024 * 1024 * 1024  # big datasets ok


@pytest.mark.parametrize("route", JSON_ROUTES)
def test_non_json_400(client, route):
    response = client.post(route, data="target=x", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "BAD_REQUEST"
    assert "json" in response.get_json()["error"]["message"].lower()


def test_preview_rows_clamped(client, upload, fixtures_dir):
    upload(fixtures_dir / "cic_ddos_sample.csv")
    big = client.get("/api/v1/dataset/preview", query_string={"rows": 10**9}).get_json()
    assert len(big["preview"]) == 100
    small = client.get("/api/v1/dataset/preview", query_string={"rows": -5}).get_json()
    assert len(small["preview"]) == 1
    bad = client.get("/api/v1/dataset/preview", query_string={"rows": "lots"})
    assert bad.status_code == 400


def test_recommend_k_clamped(client, upload, fixtures_dir):
    upload(fixtures_dir / "cic_ddos_sample.csv")
    response = client.post("/api/v1/dataset/feature-scores", json={"target": "Label", "k": 10**9})
    assert response.status_code == 200
    assert len(response.get_json()["recommended_features"]) <= 14
    bad = client.post("/api/v1/dataset/feature-scores", json={"target": "Label", "k": "many"})
    assert bad.status_code == 400
