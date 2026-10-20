"""Shared pytest fixtures for the whole test suite."""

import importlib.util
import platform
import sys
from pathlib import Path

import pytest

from platforms import PLATFORMS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _scratch_instance_dir(tmp_path_factory, monkeypatch):
    """Runs, bundles and the signing key go to a temporary instance/ in every test."""
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path_factory.mktemp("instance")))


@pytest.fixture
def fake_platform(monkeypatch):
    """Make the code under test believe it runs on another OS.

    Usage: ``fake_platform("linux")``. Patches ``platform.system()`` and
    ``sys.platform`` only; OS-specific code must decide through one of these
    (not ``os.name``, which can't be faked safely because the standard
    library depends on it).
    """

    def _set(name: str) -> None:
        system, sys_platform = PLATFORMS[name]
        monkeypatch.setattr(platform, "system", lambda: system)
        monkeypatch.setattr(sys, "platform", sys_platform)

    return _set


# ---- Web IDS app fixtures (tools/IDS/IDS_Web_CIC_Dynamic/app.py) ----

WEB_APP_FILE = "tools/IDS/IDS_Web_CIC_Dynamic/app.py"


@pytest.fixture
def load_app_module(request):
    """Import a fresh copy of the app module (call it after setting env variables)."""
    path = request.config.rootpath / WEB_APP_FILE

    def _load():
        spec = importlib.util.spec_from_file_location("ids_web_app_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ids_web_app_under_test"] = module
        spec.loader.exec_module(module)
        module.app.config["TESTING"] = True
        return module

    yield _load
    for leftover in (path.parent / "uploads").glob("*"):
        if leftover.name != ".gitkeep":
            leftover.unlink()


@pytest.fixture
def app_module(request, load_app_module):
    """A freshly imported app module, so global state never leaks between tests."""
    path = request.config.rootpath / WEB_APP_FILE
    yield load_app_module()
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
