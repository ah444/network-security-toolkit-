"""Architecture rules for the ML core (plan §2): no web framework, no print, no network."""

import ast

import pytest

FORBIDDEN_IMPORTS = {"flask", "werkzeug", "requests", "socket", "urllib", "http"}


def _core_files(repo_root):
    files = sorted((repo_root / "toolkit" / "ids_core").rglob("*.py"))
    assert files, "toolkit/ids_core is missing"
    return files


@pytest.fixture
def core_trees(repo_root):
    return {path.name: ast.parse(path.read_text(encoding="utf-8")) for path in _core_files(repo_root)}


def test_core_has_no_print(core_trees):
    for name, tree in core_trees.items():
        prints = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "print"]
        assert prints == [], f"{name}: print() at lines {prints}; use logging"


def test_core_imports_no_web_or_network(core_trees):
    for name, tree in core_trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert not roots & FORBIDDEN_IMPORTS, f"{name} imports {roots & FORBIDDEN_IMPORTS}"
