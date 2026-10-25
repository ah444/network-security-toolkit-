"""v5.1 step 1.2b: the /api/v1 routes are exactly the documented ones (docs/web_api.md)."""

import re

import pytest

from web_js import all_js, js_sources

OLD_ROUTES = ["/upload", "/preview_data", "/filter_target_values", "/get_target_values",
              "/analyze_class_distribution", "/apply_smote", "/recommend_features",
              "/get_available_models", "/configure", "/get_confusion_matrix", "/predict",
              "/predict_loaded", "/export_model", "/download_bundle", "/load_model", "/workspace",
              "/api/v1/run"]  # /api/v1/run: replaced by the training jobs in step 1.2d


def _documented(repo_root):
    text = (repo_root / "docs" / "web_api.md").read_text(encoding="utf-8")
    return {(method, route) for method, route in
            re.findall(r"^\| (GET|POST|DELETE) \| `(/api/v1[^`]*)` \|", text, re.MULTILINE)}


def _served(web_app):
    return {(method, rule.rule) for rule in web_app.url_map.iter_rules()
            if rule.rule.startswith("/api/") for method in rule.methods - {"HEAD", "OPTIONS"}}


def test_routes_match_the_documentation(repo_root, web_app):
    assert len(_documented(repo_root)) == 22
    assert _served(web_app) == _documented(repo_root)


@pytest.mark.parametrize("route", OLD_ROUTES)
def test_old_routes_are_gone(client, route):
    assert client.post(route, json={}).status_code == 404
    assert client.get(route).status_code == 404


def test_wrong_method_is_json_405(client):
    response = client.post("/api/v1/models", json={})
    assert response.status_code == 405
    assert response.get_json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_page_uses_only_api_routes():
    # the two predict routes are passed to predictWith(url, ...), which calls send(url, ...),
    # the two predict-file routes are the url of predict.js's BATCH parts (step 1.4f);
    # /api/v1/workspace is used twice: GET to resume after a reload, DELETE to clear
    urls = re.findall(r'(?:getJSON|send|predictWith)\("(/[^"?]*)|\burl: "(/[^"?]*)', all_js())
    urls = [call or part for call, part in urls]
    assert len(urls) == 17 and all(url.startswith("/api/v1/") for url in urls), urls


def test_page_sends_every_state_change_with_the_token():
    """Step 1.2e: a POST/DELETE without the token is refused (403 CSRF_TOKEN). Since step
    1.3a every request goes through api.js, and send() always adds the token."""
    code = all_js()
    # 13 POST/DELETE routes in 11 send() calls: both predict routes share predictWith's,
    # both predict-file routes predictFile's (step 1.4f)
    sends = re.findall(r'\bsend\([^,]+,\s*"([A-Z]+)"', code)
    assert len(sends) == 11 and set(sends) == {"POST", "DELETE"}, sends
    api = js_sources()["api.js"]
    assert api.count("fetch(") == 2  # getJSON and send
    assert '"X-CSRF-Token": csrfToken()' in api
