"""S5 (second layer) and S10: security headers on every response."""

import pytest


@pytest.mark.parametrize("url", ["/", "/dynamic_ids", "/get_available_models",
                                 "/static/js/main.js"])
def test_security_headers_present(client, url):
    headers = client.get(url).headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "Content-Security-Policy" in headers


def test_csp_present_and_scripts_strict(client):
    csp = client.get("/").headers["Content-Security-Policy"]
    directives = dict(part.strip().split(" ", 1) for part in csp.split(";") if part.strip())
    assert directives["script-src"] == "'self'"  # no inline scripts, no eval
    assert "'unsafe-inline'" not in directives["script-src"]
    assert directives["object-src"] == "'none'"
    assert directives["frame-ancestors"] == "'none'"
    # Inline style attributes are still used by the v4 page (tightened in v5.1).
    assert directives["style-src"] == "'self' 'unsafe-inline'"


def test_templates_have_no_inline_scripts_or_handlers(repo_root):
    """Keeps the strict script-src from breaking the page."""
    import re

    templates = (repo_root / "toolkit/web/templates").glob("*.html")
    for template in templates:
        html = template.read_text(encoding="utf-8")
        assert not re.search(r"<script(?![^>]*\bsrc=)", html), template.name
        assert not re.search(r"\son[a-z]+\s*=", html), template.name
