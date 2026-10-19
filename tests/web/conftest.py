"""Fixtures for tests of the web IDS app (tools/IDS/IDS_Web_CIC_Dynamic/app.py)."""

import importlib.util
import sys

import pytest

APP_FILE = "tools/IDS/IDS_Web_CIC_Dynamic/app.py"


@pytest.fixture
def app_module(request):
    """A freshly imported app module, so global state never leaks between tests."""
    path = request.config.rootpath / APP_FILE
    spec = importlib.util.spec_from_file_location("ids_web_app_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ids_web_app_under_test"] = module
    spec.loader.exec_module(module)
    module.app.config["TESTING"] = True
    yield module
    for leftover in (path.parent / "uploads").glob("*"):
        if leftover.name != ".gitkeep":
            leftover.unlink()


@pytest.fixture
def client(app_module):
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def upload(client):
    """Upload a file through the real /upload route; returns the JSON response."""

    def _upload(path, name=None):
        with path.open("rb") as handle:
            response = client.post("/upload", data={"dataset": (handle, name or path.name)},
                                   content_type="multipart/form-data")
        return response

    return _upload
