"""Step 0.8: the session cookie only carries the workspace ID, with safe flags (S10, D23)."""


def _set_cookie_after_upload(test_client, fixtures_dir):
    path = fixtures_dir / "cic_ddos_sample.csv"
    with path.open("rb") as handle:
        response = test_client.post("/upload", data={"dataset": (handle, path.name)},
                                    content_type="multipart/form-data")
    assert response.status_code == 200
    cookies = response.headers.getlist("Set-Cookie")
    assert len(cookies) == 1, cookies
    return cookies[0]


def test_session_cookie_flags(client, fixtures_dir, monkeypatch):
    cookie = _set_cookie_after_upload(client, fixtures_dir)
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Secure" not in cookie, "plain http://127.0.0.1 would drop a Secure cookie"
    with client.session_transaction() as session:
        assert set(session) == {"ws"}, "the cookie must hold the workspace ID only"


def test_secure_flag_with_toolkit_https(load_app_module, fixtures_dir, monkeypatch):
    monkeypatch.setenv("TOOLKIT_HTTPS", "1")
    module = load_app_module()
    with module.app.test_client() as test_client:
        cookie = _set_cookie_after_upload(test_client, fixtures_dir)
    assert "Secure" in cookie


def test_secret_key_not_hard_coded(load_app_module, monkeypatch):
    monkeypatch.delenv("TOOLKIT_SECRET_KEY", raising=False)
    first, second = load_app_module().app.secret_key, load_app_module().app.secret_key
    assert first and second and first != second
