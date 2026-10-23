"""S6: no debug server by default; the app is served by waitress on 127.0.0.1 only."""

import ast

import flask
import pytest

from toolkit import web


@pytest.fixture
def calls(monkeypatch):
    recorded = {}
    import waitress

    monkeypatch.setattr(waitress, "serve", lambda app, **kw: recorded.setdefault("serve", kw))
    monkeypatch.setattr(flask.Flask, "run",
                        lambda self, **kw: recorded.setdefault("run", kw))
    return recorded


def test_debug_off_by_default(calls, monkeypatch):
    monkeypatch.delenv("TOOLKIT_DEBUG", raising=False)
    web.run_server()
    assert "run" not in calls, "the Flask debug/dev server must not be used by default"
    assert calls["serve"]["host"] == "127.0.0.1"


def test_debug_only_when_explicitly_enabled(calls, monkeypatch):
    monkeypatch.setenv("TOOLKIT_DEBUG", "1")
    web.run_server()
    assert calls["run"]["debug"] is True
    assert calls["run"]["host"] == "127.0.0.1"


def test_port_from_environment(calls, monkeypatch):
    monkeypatch.setenv("TOOLKIT_PORT", "5055")
    web.run_server()
    assert calls["serve"]["port"] == 5055


def test_main_block_uses_run_server(repo_root):
    package = (repo_root / "toolkit/web/__init__.py").read_text(encoding="utf-8")
    assert "debug=True" not in package.split("def run_server", 1)[0], "debug hard-coded"
    source = (repo_root / "toolkit/web/__main__.py").read_text(encoding="utf-8")
    assert "debug=" not in source and ".run(" not in source, "python -m only calls run_server"
    tree = ast.parse(source)
    calls = [ast.unparse(n) for n in tree.body
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert calls == ["run_server()"]
