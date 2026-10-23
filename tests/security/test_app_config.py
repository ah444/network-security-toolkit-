"""S6: no debug server by default; the app is served by waitress on 127.0.0.1 only."""

import ast
import socket

import flask
import pytest

from toolkit import web


@pytest.fixture
def calls(monkeypatch):
    recorded = {}
    import waitress

    monkeypatch.setattr(waitress, "serve", lambda app, **kw: recorded.setdefault(
        "serve", {**kw, "address": kw["sockets"][0].getsockname()}))
    monkeypatch.setattr(flask.Flask, "run",
                        lambda self, **kw: recorded.setdefault("run", kw))
    return recorded


def test_debug_off_by_default(calls, monkeypatch):
    monkeypatch.delenv("TOOLKIT_DEBUG", raising=False)
    web.run_server()
    assert "run" not in calls, "the Flask debug/dev server must not be used by default"
    assert calls["serve"]["address"][0] == "127.0.0.1"
    assert "host" not in calls["serve"], "waitress gets the socket bound here (step 1.4d)"


def test_debug_only_when_explicitly_enabled(calls, monkeypatch):
    monkeypatch.setenv("TOOLKIT_DEBUG", "1")
    web.run_server()
    assert calls["run"]["debug"] is True
    assert calls["run"]["host"] == "127.0.0.1"
    assert calls["run"]["use_reloader"] is False, "one process, no reloader (B16)"


def test_port_from_environment(calls, monkeypatch):
    with socket.socket() as probe:  # a port free right now
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("TOOLKIT_PORT", str(port))
    web.run_server()
    assert calls["serve"]["address"] == ("127.0.0.1", port)


def test_main_block_uses_run_server(repo_root):
    package = (repo_root / "toolkit/web/__init__.py").read_text(encoding="utf-8")
    assert "debug=True" not in package.split("def run_server", 1)[0], "debug hard-coded"
    source = (repo_root / "toolkit/web/__main__.py").read_text(encoding="utf-8")
    assert "debug=" not in source and ".run(" not in source, "python -m only calls run"
    tree = ast.parse(source)
    calls = [ast.unparse(n) for n in tree.body
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert calls == ["run()"]  # step 1.4b: its --help, then run_server (tests/cli/test_main_menu.py)
