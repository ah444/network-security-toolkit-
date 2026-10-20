"""Fixtures for the port scanner tests (step 0.6). No test reaches the internet."""

import importlib.util
import socket
import sys
import threading

import pytest

from toolkit.recon.intel import nvd
from toolkit.recon.intel import keys as intel_keys

SCANNER_FILE = "tools/Port_scan/Port_Scan.py"


@pytest.fixture(scope="module")
def scanner(request):
    """Import the scanner script as a module."""
    path = request.config.rootpath / SCANNER_FILE
    spec = importlib.util.spec_from_file_location("port_scan_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["port_scan_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_real_keys(scanner, monkeypatch, tmp_path):
    """The user's own API keys (environment, .env, api_keys.txt) are never used in tests."""
    for variable in (*intel_keys.KEY_VARIABLES.values(), "OPENROUTER_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(intel_keys, "API_KEYS_PATH", tmp_path / "no_api_keys.txt")
    monkeypatch.setattr(intel_keys, "ENV_PATH", tmp_path / "no.env")
    monkeypatch.setattr(intel_keys.load_api_keys, "__defaults__",
                        (None, tmp_path / "no_api_keys.txt", tmp_path / "no.env"))


@pytest.fixture
def tcp_server():
    """A local TCP server that sends ``reply`` to each client (after reading its request
    when ``read_first``, like an HTTP server); returns its port."""
    servers = []

    def _start(reply=b"", host="127.0.0.1", family=socket.AF_INET, read_first=False):
        server = socket.socket(family, socket.SOCK_STREAM)
        server.bind((host, 0))
        server.listen()
        server.settimeout(5)
        servers.append(server)

        def serve():
            while True:
                try:
                    client, _ = server.accept()
                except OSError:
                    return
                with client:
                    client.settimeout(5)
                    if read_first:
                        try:
                            client.recv(1024)
                        except OSError:
                            pass
                    if reply:
                        try:
                            client.sendall(reply)
                        except OSError:
                            pass

        threading.Thread(target=serve, daemon=True).start()
        return server.getsockname()[1]

    yield _start
    for server in servers:
        server.close()


@pytest.fixture
def free_port():
    """A local port with nothing listening (TCP and UDP)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeSession:
    """Records every request; answers from ``get``/``post`` (a value or an exception)."""

    def __init__(self, get=None, post=None):
        self.answers = {"get": get, "post": post}
        self.calls = []

    def _answer(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        answer = self.answers[method]
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, FakeResponse) else FakeResponse(answer)

    def get(self, url, **kwargs):
        return self._answer("get", url, **kwargs)

    def post(self, url, **kwargs):
        return self._answer("post", url, **kwargs)


@pytest.fixture
def fake_session():
    return FakeSession


@pytest.fixture
def fake_response():
    return FakeResponse


@pytest.fixture
def no_nvd_wait(scanner, monkeypatch):
    monkeypatch.setattr(nvd, "NVD_INTERVAL", {False: 0.0, True: 0.0})
