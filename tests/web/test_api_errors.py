"""v5.1 step 1.2b: one error shape, ``{"error": {"code", "message", "details"?}}`` (S8)."""

import io

from toolkit.services.web import errors
from toolkit.controllers.web import api


def _error(response):
    body = response.get_json()
    assert set(body) == {"error"}, body
    assert {"code", "message"} <= set(body["error"]) <= {"code", "message", "details"}
    return body["error"]


def test_error_codes_are_distinct():
    codes = [value for name, value in vars(errors).items() if name.isupper() and isinstance(value, str)]
    assert len(codes) == len(set(codes)) >= 13
    assert all(code == code.upper() for code in codes)


def test_details_only_when_given(web_app):
    with web_app.test_request_context():
        plain, status = api.error_response("Nope.", 400, errors.BAD_REQUEST)
        detailed, _ = api.error_response("Nope.", 400, errors.TRAINING_FAILED, failures=[1])
    assert status == 400 and plain.get_json() == {"error": {"code": "BAD_REQUEST",
                                                             "message": "Nope."}}
    assert detailed.get_json()["error"]["details"] == {"failures": [1]}


def test_unknown_api_route_is_json(client):
    response = client.get("/api/v1/no_such_route")
    assert response.status_code == 404 and _error(response)["code"] == "NOT_FOUND"


def test_pages_keep_html_404(client):
    response = client.get("/no_such_page")
    assert response.status_code == 404 and response.mimetype == "text/html"


def test_cross_site_has_a_code(client):
    response = client.post("/api/v1/dataset/filter", json={"target": "x", "remove": ["y"]},
                           headers={"Origin": "http://evil.example"})
    assert response.status_code == 403 and _error(response)["code"] == "CROSS_SITE"


def test_errors_of_each_kind(client):
    assert _error(client.get("/api/v1/dataset/preview", query_string={}))["code"] == "NO_DATASET"
    assert _error(client.post("/api/v1/dataset/filter", data="x",
                              content_type="text/plain"))["code"] == "BAD_REQUEST"
    upload = client.post("/api/v1/dataset",
                         data={"dataset": (io.BytesIO(b"a,Label\n1,X\n2,Y\n"), "d.csv")},
                         content_type="multipart/form-data")
    assert upload.status_code == 200
    classes = client.get("/api/v1/dataset/classes", query_string={"target": "nope"})
    assert _error(classes)["code"] == "UNKNOWN_COLUMN"
    predict = client.post("/api/v1/run/predict", json={"values": {"a": 1}})
    assert _error(predict)["code"] == "NOT_TRAINED"
    assert _error(client.post("/api/v1/bundle/predict", json={"values": {"a": 1}}))["code"] == \
        "NO_MODEL_LOADED"
