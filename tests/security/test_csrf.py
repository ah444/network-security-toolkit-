"""S10, step 1.2e: CSRF token on every state change, plus the earlier layers (JSON-only
bodies, the Origin check)."""

import io
import re

import pytest

TOKEN_REFUSED = "CSRF_TOKEN"


def _small_csv():
    return io.BytesIO(b"a,b,Label\n1,2,X\n3,4,Y\n")


def _upload(test_client, **kwargs):
    return test_client.post("/api/v1/dataset", data={"dataset": (_small_csv(), "d.csv")},
                            content_type="multipart/form-data", **kwargs)


def _code(response):
    return response.get_json()["error"]["code"]


def _page_token(test_client):
    page = test_client.get("/dynamic_ids").get_data(as_text=True)
    return re.search(r'<meta name="csrf-token" content="([^"]+)">', page).group(1)


# ---- the token ----


def test_page_carries_the_token(client):
    token = _page_token(client)
    assert len(token) >= 40
    with client.session_transaction() as session:
        assert session["csrf"] == token


def test_token_is_kept_for_the_session(client):
    first = _page_token(client)
    assert _upload(client).status_code == 200
    assert client.delete("/api/v1/workspace").status_code == 200  # clearing keeps it
    assert _page_token(client) == first


def test_each_session_gets_its_own_token(web_app):
    first, second = web_app.test_client(), web_app.test_client()
    assert _page_token(first) != _page_token(second)


@pytest.mark.parametrize("method, url", [
    ("post", "/api/v1/dataset/filter"),
    ("post", "/api/v1/jobs/training"),
    ("post", "/api/v1/run/export"),
    ("post", "/api/v1/run/predict"),
    ("post", "/api/v1/bundle/predict"),
    ("post", "/api/v1/run/predict-file"),
    ("post", "/api/v1/bundle/predict-file"),
    ("delete", "/api/v1/workspace"),
    ("delete", "/api/v1/jobs/abc"),
])
def test_state_change_without_token_refused(client, method, url):
    _page_token(client)  # the session has a token; the request just doesn't send it
    client.send_token = False
    response = getattr(client, method)(url, json={})
    assert response.status_code == 403 and _code(response) == TOKEN_REFUSED
    assert "Reload the page" in response.get_json()["error"]["message"]


def test_upload_and_bundle_load_without_token_refused(client):
    _page_token(client)
    client.send_token = False
    assert _code(_upload(client)) == TOKEN_REFUSED
    response = client.post("/api/v1/bundle", data={"file": (io.BytesIO(b"x"), "m.bundle")},
                           content_type="multipart/form-data")
    assert response.status_code == 403 and _code(response) == TOKEN_REFUSED


@pytest.mark.parametrize("token", ["", "wrong", "é" * 43])
def test_wrong_token_refused(client, token):
    _page_token(client)
    response = _upload(client, headers={"X-CSRF-Token": token})
    assert response.status_code == 403 and _code(response) == TOKEN_REFUSED


def test_another_sessions_token_refused(web_app):
    victim, attacker = web_app.test_client(), web_app.test_client()
    stolen = _page_token(victim)
    _page_token(attacker)
    response = _upload(attacker, headers={"X-CSRF-Token": stolen})
    assert response.status_code == 403 and _code(response) == TOKEN_REFUSED


def test_session_without_token_refused(client):
    """No page loaded yet: the session has no token, so nothing can match."""
    response = _upload(client, headers={"X-CSRF-Token": "made-up"})
    assert response.status_code == 403 and _code(response) == TOKEN_REFUSED


def test_token_from_before_a_restart_refused(make_app):
    """A restart makes a new signing key: the old cookie is unreadable, the old page's token
    matches nothing, and the user is told to reload."""
    before, after = make_app(), make_app()
    old_client = before.test_client()
    old_token = _page_token(old_client)
    new_client = after.test_client()
    new_client.set_cookie("session", old_client.get_cookie("session").value)
    response = _upload(new_client, headers={"X-CSRF-Token": old_token})
    assert response.status_code == 403 and _code(response) == TOKEN_REFUSED
    assert _upload(new_client).status_code == 200  # after a reload (new token) it works


def test_reads_need_no_token(client):
    client.send_token = False
    assert client.get("/api/v1/models").status_code == 200
    assert client.get("/api/v1/dataset/preview").status_code == 400  # NO_DATASET, not 403


def test_same_origin_upload_with_token_allowed(client):
    response = _upload(client, headers={"Origin": "http://localhost"})
    assert response.status_code == 200, response.get_json()


# ---- the earlier layers stay ----


def test_form_post_rejected(client):
    response = client.post("/api/v1/jobs/training", data={"target": "x"})  # HTML-form style body
    assert response.status_code == 400


def test_foreign_origin_upload_rejected(client):
    response = _upload(client, headers={"Origin": "http://evil.example"})
    assert response.status_code == 403 and _code(response) == "CROSS_SITE"


def test_foreign_origin_json_rejected(client):
    response = client.post("/api/v1/dataset/filter", json={"target": "x", "remove": ["y"]},
                           headers={"Origin": "http://evil.example"})
    assert response.status_code == 403 and _code(response) == "CROSS_SITE"
