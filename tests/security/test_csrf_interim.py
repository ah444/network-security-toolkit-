"""S10 interim (until the v5.1 CSRF token): JSON-only state changes + Origin check."""

import io


def test_form_post_rejected(client):
    response = client.post("/configure", data={"target": "x"})  # HTML-form style body
    assert response.status_code == 400


def test_foreign_origin_upload_rejected(client):
    response = client.post(
        "/upload", data={"dataset": (io.BytesIO(b"a,Label\n1,X\n"), "d.csv")},
        content_type="multipart/form-data", headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_foreign_origin_json_rejected(client):
    response = client.post("/preview_data", json={"num_rows": 5},
                           headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_same_origin_upload_allowed(client):
    response = client.post(
        "/upload", data={"dataset": (io.BytesIO(b"a,b,Label\n1,2,X\n3,4,Y\n"), "d.csv")},
        content_type="multipart/form-data", headers={"Origin": "http://localhost"})
    assert response.status_code == 200, response.get_json()
