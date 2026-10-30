"""v5.2 step 2.3a: the code and the configuration break exactly the rules listed in
``tests/exceptions_inventory.py``, each with a reason; nothing more, nothing stale.

Comments are read with ``tokenize``, so a ``# noqa`` written inside a string (a test's sample
code) is not a suppression. A suppression that no longer suppresses anything is also caught by
ruff itself (RUF100) and, for ``type: ignore``, by mypy's strict mode (unused ignores).
"""

import io
import re
import tokenize
import tomllib
from collections import Counter

import pytest

import exceptions_inventory as inventory

FOLDERS = ("toolkit", "scripts", "tests")
PATTERNS = {
    "noqa": re.compile(r"#\s*noqa\b(?::\s*(?P<codes>[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*))?",
                       re.IGNORECASE),
    "type: ignore": re.compile(r"#\s*type:\s*ignore\b(?:\[(?P<codes>[^\]]*)\])?"),
    "pragma: no cover": re.compile(r"#\s*pragma:\s*no\s+cover\b"),
    "ruff: noqa": re.compile(r"#\s*ruff\s*:\s*noqa\b", re.IGNORECASE),
}
JS_PATTERNS = re.compile(r"eslint-disable|@ts-ignore|@ts-nocheck|jshint\s+ignore")


def _python_files(repo_root):
    for folder in FOLDERS:
        for path in sorted((repo_root / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def _found(repo_root):
    """(file, rule) -> count, where rule is e.g. ``noqa[E402]`` or ``pragma: no cover``."""
    counts = Counter()
    for path in _python_files(repo_root):
        name = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.COMMENT:
                continue
            for kind, pattern in PATTERNS.items():
                for match in pattern.finditer(token.string):
                    codes = (match.groupdict().get("codes") or "").replace(" ", "")
                    counts[name, f"{kind}[{codes}]" if codes else kind] += 1
    return counts


def _pyproject(repo_root):
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_suppressions_are_the_listed_ones(repo_root):
    listed = Counter({(path, rule): count for path, rule, count, _ in inventory.SUPPRESSIONS})
    found = _found(repo_root)
    new = {key: n for key, n in found.items() if n > listed.get(key, 0)}
    stale = {key: n for key, n in listed.items() if found.get(key, 0) < n}
    assert not new, f"not in tests/exceptions_inventory.py (add it with a reason): {new}"
    assert not stale, f"listed but no longer in the code (remove or lower the count): {stale}"


def test_every_suppression_names_its_rule(repo_root):
    """A bare ``# noqa`` or ``# type: ignore`` would switch off every rule on the line."""
    bare = {key for key in _found(repo_root)
            if key[1] in ("noqa", "type: ignore", "ruff: noqa")}
    assert not bare, bare


def test_every_entry_has_a_reason():
    for path, rule, count, reason in inventory.SUPPRESSIONS:
        assert count >= 1 and len(reason.split()) >= 5, (path, rule)
    for table, key, _, reason in inventory.CONFIGURATION:
        assert len(reason.split()) >= 5, (table, key)


def test_per_file_ignores_are_the_listed_ones(repo_root):
    configured = _pyproject(repo_root)["tool"]["ruff"]["lint"].get("per-file-ignores", {})
    assert {k: tuple(v) for k, v in configured.items()} == inventory.PER_FILE_IGNORES


def test_mypy_overrides_are_the_listed_ones(repo_root):
    overrides = _pyproject(repo_root)["tool"]["mypy"].get("overrides", [])
    found = [("tool.mypy.overrides", key, tuple(override["module"]))
             for override in overrides for key, value in override.items()
             if key != "module" and value is True]
    listed = [(table, key, value) for table, key, value, _ in inventory.CONFIGURATION
              if table == "tool.mypy.overrides"]
    assert found == listed


def test_no_suppressions_in_the_javascript(repo_root):
    for path in sorted((repo_root / "toolkit").rglob("*.js")):
        assert not JS_PATTERNS.search(path.read_text(encoding="utf-8")), path.name


@pytest.mark.parametrize("text, expected", [
    ("x = 1  # noqa: E402", {"noqa[E402]": 1}),
    ("x = 1  # noqa", {"noqa": 1}),
    ("x = 1  # NOQA:E402,F401", {"noqa[E402,F401]": 1}),
    ("x = 1  # type: ignore[misc]", {"type: ignore[misc]": 1}),
    ("x = 1  # type: ignore", {"type: ignore": 1}),
    ("raise AssertionError  # pragma: no cover", {"pragma: no cover": 1}),
    ("# ruff: noqa", {"ruff: noqa": 1}),
    ("x = '# noqa: E402'", {}),
], ids=lambda value: value if isinstance(value, str) else "")
def test_the_reader_finds_comments_not_strings(tmp_path, text, expected):
    (tmp_path / "toolkit").mkdir()
    (tmp_path / "toolkit" / "sample.py").write_text(text + "\n", encoding="utf-8")
    found = {rule: n for (_, rule), n in _found(tmp_path).items()}
    assert found == expected
