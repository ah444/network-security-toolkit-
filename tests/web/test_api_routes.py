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
    # + the scan's 4 (2.4d), the unlock (2.4f), the saved runs' 3 (v5.3 F2c), the access 3 (HS1),
    # a run's files and its delete (HR4a), the storage report and a bundle registered (HR4b),
    # a run saved again (HR5a), the importance job and a saved run's results (F3d)
    assert len(_documented(repo_root)) == 40
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
    # /api/v1/workspace is used twice: GET to resume after a reload, DELETE to clear;
    # the scan page (v5.2 step 2.4e): its options, a scan started, and the session's scan,
    # asked after a reload and after a scan that was cancelled; the unlock page's (2.4f);
    # the saved runs' page (v5.3 F2e): the list, one run and the comparison; the home page's
    # other computers (HS1): their jobs and a new token (a job's Stop sends to its own url);
    # a run's files and its delete (HR4a); the storage report and a bundle registered (HR4b);
    # a run saved again (HR5a); the importance calculated and a saved run's results (F3e)
    urls = re.findall(r'(?:getJSON|send|predictWith)\("(/[^"?]*)|\burl: "(/[^"?]*)', all_js())
    urls = [call or part for call, part in urls]
    assert len(urls) == 34 and all(url.startswith("/api/v1/") for url in urls), urls


def test_page_sends_every_state_change_with_the_token():
    """Step 1.2e: a POST/DELETE without the token is refused (403 CSRF_TOKEN). Since step
    1.3a every request goes through api.js, and send() always adds the token."""
    code = all_js()
    # 22 POST/DELETE routes in 21 send() calls: both predict routes share predictWith's,
    # both predict-file routes predictFile's (step 1.4f); a scan started and cancelled (2.4e),
    # the unlock (2.4f); a new token and another computer's job stopped (HS1); a run's
    # delete (HR4a); a bundle registered again (HR4b); a run saved again (HR5a); the
    # importance calculated and cancelled (F3e: a job's DELETE, as a training's)
    sends = re.findall(r'\bsend\([^,]+,\s*"([A-Z]+)"', code)
    assert len(sends) == 21 and set(sends) == {"POST", "DELETE"}, sends
    api = js_sources()["api.js"]
    assert api.count("fetch(") == 2  # getJSON and send
    assert '"X-CSRF-Token": csrfToken()' in api
