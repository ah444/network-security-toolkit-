"""Characterization smoke tests: the existing apps still start (they must keep passing)."""

import importlib.util
import sys

import pytest

WEB_APP = "tools/IDS/IDS_Web_CIC_Dynamic/app.py"


@pytest.fixture(scope="module")
def web_client(request):
    root = request.config.rootpath
    spec = importlib.util.spec_from_file_location("ids_web_app", root / WEB_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ids_web_app"] = module
    spec.loader.exec_module(module)
    module.app.config["TESTING"] = True
    with module.app.test_client() as client:
        yield client


@pytest.mark.parametrize("url", ["/", "/dynamic_ids"])
def test_web_pages_render(web_client, url):
    response = web_client.get(url)
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_available_models_endpoint(web_client):
    response = web_client.get("/get_available_models")
    assert response.status_code == 200
    ids = {model["id"] for model in response.get_json()["models"]}
    assert {"random_forest", "logistic_regression"} <= ids
