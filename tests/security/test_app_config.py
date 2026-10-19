"""S6: no debug server by default; the app is served by waitress on 127.0.0.1 only."""

import ast

import pytest


@pytest.fixture
def calls(app_module, monkeypatch):
    recorded = {}
    import waitress

    monkeypatch.setattr(waitress, "serve", lambda app, **kw: recorded.setdefault("serve", kw))
    monkeypatch.setattr(app_module.app, "run", lambda **kw: recorded.setdefault("run", kw))
    return recorded


def test_debug_off_by_default(app_module, calls, monkeypatch):
    monkeypatch.delenv("TOOLKIT_DEBUG", raising=False)
    app_module.run_server()
    assert "run" not in calls, "the Flask debug/dev server must not be used by default"
    assert calls["serve"]["host"] == "127.0.0.1"


def test_debug_only_when_explicitly_enabled(app_module, calls, monkeypatch):
    monkeypatch.setenv("TOOLKIT_DEBUG", "1")
    app_module.run_server()
    assert calls["run"]["debug"] is True
    assert calls["run"]["host"] == "127.0.0.1"


def test_port_from_environment(app_module, calls, monkeypatch):
    monkeypatch.setenv("TOOLKIT_PORT", "5055")
    app_module.run_server()
    assert calls["serve"]["port"] == 5055


def test_main_block_uses_run_server(repo_root):
    source = (repo_root / "tools/IDS/IDS_Web_CIC_Dynamic/app.py").read_text(encoding="utf-8")
    assert "debug=True" not in source.split("def run_server", 1)[0], "debug hard-coded"
    tree = ast.parse(source)
    main_blocks = [n for n in tree.body if isinstance(n, ast.If)
                   and "__main__" in ast.unparse(n.test)]
    assert main_blocks and "run_server()" in ast.unparse(main_blocks[-1])
