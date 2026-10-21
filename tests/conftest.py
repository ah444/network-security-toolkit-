"""Shared pytest fixtures for the whole test suite."""

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


# ---- Web IDS app fixtures (toolkit.web, v5.1 step 1.2a) ----


@pytest.fixture
def make_app():
    """Build a fresh web app (call it after setting env variables): its own workspaces,
    model store and upload folder, so state never leaks between apps or tests."""
    from toolkit.web import create_app

    def _make(config=None):
        app = create_app(config)
        app.config["TESTING"] = True
        return app

    return _make


@pytest.fixture
def web_app(make_app):
    return make_app()


@pytest.fixture
def client(web_app):
    with web_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def workspaces(web_app):
    """The app's WorkspaceStore: ``len()``, ``values()``, ``idle_seconds``."""
    return web_app.extensions["toolkit"].workspaces


@pytest.fixture
def upload(client):
    """Upload a file through the real /upload route; returns the JSON response."""

    def _upload(path, name=None):
        with path.open("rb") as handle:
            response = client.post("/upload", data={"dataset": (handle, name or path.name)},
                                   content_type="multipart/form-data")
        return response

    return _upload
