"""S5 (second layer) and S10: security headers on every response."""

import pytest


@pytest.mark.parametrize("url", ["/", "/dynamic_ids", "/api/v1/models",
                                 "/static/js/main.js"])
def test_security_headers_present(client, url):
    headers = client.get(url).headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "Content-Security-Policy" in headers
    assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert "camera=()" in headers["Permissions-Policy"]


def test_permissions_policy_turns_off_unused_features(client):
    policy = client.get("/").headers["Permissions-Policy"]
    features = dict(part.strip().split("=", 1) for part in policy.split(","))
    for feature in ("camera", "microphone", "geolocation", "payment", "usb"):
        assert features[feature] == "()", feature


@pytest.mark.parametrize("url", ["/api/v1/models", "/api/v1/dataset/preview",
                                 "/api/v1/no-such-route", "/dynamic_ids"])
def test_api_answers_and_trainer_page_are_not_stored(client, url):
    assert client.get(url).headers["Cache-Control"] == "no-store"


def test_error_and_upload_answers_are_not_stored(client):
    response = client.post("/api/v1/dataset/filter", data="not json",
                           content_type="text/plain")
    assert response.status_code == 400
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("url", ["/", "/static/js/main.js"])
def test_home_page_and_static_files_stay_cacheable(client, url):
    assert "no-store" not in client.get(url).headers.get("Cache-Control", "")


def test_csp_present_and_scripts_strict(client):
    csp = client.get("/").headers["Content-Security-Policy"]
    directives = dict(part.strip().split(" ", 1) for part in csp.split(";") if part.strip())
    assert directives["script-src"] == "'self'"  # no inline scripts, no eval
    assert "'unsafe-inline'" not in directives["script-src"]
    assert directives["object-src"] == "'none'"
    assert directives["frame-ancestors"] == "'none'"
    # v5.1 step 1.3b: no inline styles left (tests/web/test_css.py), so none are allowed.
    assert directives["style-src"] == "'self'"


def test_templates_have_no_inline_scripts_or_handlers(repo_root):
    """Keeps the strict script-src from breaking the page."""
    import re

    templates = (repo_root / "toolkit/web/templates").glob("*.html")
    for template in templates:
        html = template.read_text(encoding="utf-8")
        assert not re.search(r"<script(?![^>]*\bsrc=)", html), template.name
        assert not re.search(r"\son[a-z]+\s*=", html), template.name
