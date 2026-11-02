"""Shared pytest fixtures for the whole test suite."""

import platform
import re
import sys
from pathlib import Path

import pytest

from levels import LEVELS
from toolkit.core.data import protect
from platforms import PLATFORMS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def pytest_collection_modifyitems(items):
    """Every test gets its file's level from tests/levels.py: ``pytest -m unit`` runs one."""
    tests_dir = REPO_ROOT / "tests"
    for item in items:
        level = LEVELS.get(item.path.relative_to(tests_dir).as_posix())
        if level:
            item.add_marker(getattr(pytest.mark, level))

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
    monkeypatch.setenv("TOOLKIT_NO_BROWSER", "1")  # no test ever opens a real browser
    # no test finds this computer's nmap: TOOLKIT_NMAP_PATH, when set, is the only place looked at
    monkeypatch.setenv("TOOLKIT_NMAP_PATH", str(tmp_path_factory.mktemp("no_nmap") / "nmap"))
    # no test changes the permissions of a real file (this checkout's .env): HS5's own tests
    # put the real one back (tests/test_protect.py)
    monkeypatch.setattr(protect, "_apply", lambda path: None)


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

STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}
APP_ADDRESS = "127.0.0.1"  # where the test clients send their requests (the Host check)


def _token_client_class():
    from flask.testing import FlaskClient
    from werkzeug.datastructures import Headers

    class TokenClient(FlaskClient):
        """Sends the session's CSRF token on POST/PUT/PATCH/DELETE like the page does
        (step 1.2e): the token is read from the real page's <meta name="csrf-token"> the
        first time the session needs one. A test that passes its own X-CSRF-Token header
        keeps it; ``send_token = False`` sends none."""

        send_token = True

        def csrf_token(self):
            with self.session_transaction() as session:
                token = session.get("csrf")
            if token is None:
                page = super().open("/dynamic_ids").get_data(as_text=True)
                token = re.search(r'<meta name="csrf-token" content="([^"]+)">', page).group(1)
            return token

        # the cookies are the app's address's (``SERVER_NAME``, see ``make_app``)
        def get_cookie(self, key, domain=APP_ADDRESS, path="/"):
            return super().get_cookie(key, domain, path)

        def set_cookie(self, key, value="", *, domain=APP_ADDRESS, **kwargs):
            super().set_cookie(key, value, domain=domain, **kwargs)

        def open(self, *args, **kwargs):
            if self.send_token and kwargs.get("method", "GET").upper() in STATE_CHANGING:
                headers = Headers(kwargs.get("headers"))
                if "X-CSRF-Token" not in headers:
                    headers["X-CSRF-Token"] = self.csrf_token()
                kwargs["headers"] = headers
            return super().open(*args, **kwargs)

    return TokenClient


@pytest.fixture
def make_app():
    """Build a fresh web app (call it after setting env variables): its own workspaces,
    model store and upload folder, so state never leaks between apps or tests. Its test
    clients send the CSRF token (see ``_token_client_class``)."""
    from toolkit.web.controllers import create_app

    def _make(config=None):
        app = create_app(config)
        app.config["TESTING"] = True
        # The test clients' address, as a browser on the app sends it (the Host check, v5.2
        # step 2.4a); since Flask 3.1 it doesn't limit which hosts the routes answer.
        app.config["SERVER_NAME"] = f"{APP_ADDRESS}:{app.extensions['toolkit'].config.port}"
        app.test_client_class = _token_client_class()
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
            response = client.post("/api/v1/dataset", data={"dataset": (handle, name or path.name)},
                                   content_type="multipart/form-data")
        return response

    return _upload
