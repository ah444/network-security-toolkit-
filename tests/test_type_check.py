"""mypy --strict on the new toolkit modules (v5.1 step 1.1e, plan §6.1).

mypy itself runs in the pre-commit hook; these tests keep its scope from shrinking
unnoticed: strict mode, the modules of step 1.1e, the hook step and the pinned version.
"""

import tomllib

STRICT_MODULES = {"toolkit/recon", "toolkit/ports.py", "toolkit/logging_setup.py",
                  "toolkit/config.py"}


def _mypy_config(repo_root):
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["mypy"]


def test_strict_mypy_covers_the_new_modules(repo_root):
    config = _mypy_config(repo_root)
    assert config["strict"] is True
    assert STRICT_MODULES <= set(config["files"])
    assert all((repo_root / path).exists() for path in config["files"])


def test_untyped_modules_are_only_read_silently(repo_root):
    """Modules imported for types only may be read silently, never skipped or ignored."""
    for override in _mypy_config(repo_root).get("overrides", []):
        assert set(override) <= {"module", "follow_imports"}
        assert override.get("follow_imports", "silent") == "silent"


def test_hook_runs_mypy_and_mypy_is_pinned(repo_root):
    hook = (repo_root / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    assert '"$PY" -m mypy || exit 1' in hook
    dev = (repo_root / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("mypy==") for line in dev)
