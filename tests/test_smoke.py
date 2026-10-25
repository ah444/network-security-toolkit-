"""Characterization smoke tests: the existing apps still start (they must keep passing)."""

import pytest


@pytest.mark.parametrize("url", ["/", "/dynamic_ids"])
def test_web_pages_render(client, url):
    response = client.get(url)
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_available_models_endpoint(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    ids = {model["id"] for model in response.get_json()["models"]}
    assert {"random_forest", "logistic_regression"} <= ids
