"""mypy --strict on the whole toolkit package (v5.1 steps 1.1e and 1.5a, plan §6.1).

mypy itself runs in the pre-commit hook; these tests keep its scope from shrinking
unnoticed: strict mode, the whole package, the hook step and the pinned version.
"""

import tomllib


def _mypy_config(repo_root):
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["mypy"]


def test_strict_mypy_covers_the_whole_package(repo_root):
    config = _mypy_config(repo_root)
    assert config["strict"] is True
    assert config["files"] == ["toolkit"]


def test_no_toolkit_module_is_left_out_of_the_check(repo_root):
    """Our own modules are all checked: an override may only name third-party libraries
    without type information, and may only ignore their missing types."""
    for override in _mypy_config(repo_root).get("overrides", []):
        assert set(override) <= {"module", "ignore_missing_imports"}
        assert not [m for m in override["module"] if m.startswith("toolkit")]


def test_hook_runs_mypy_and_mypy_is_pinned(repo_root):
    hook = (repo_root / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    assert '"$PY" -m mypy || exit 1' in hook
    dev = (repo_root / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("mypy==") for line in dev)
