r"""Portability guard: keeps the code runnable on Windows, Linux and macOS.

Scans the project's Python source for patterns that break on other OSes:
- ``open()`` / ``read_text()`` / ``write_text()`` in text mode without ``encoding=``
- ``shell=True`` in subprocess calls
- hard-coded Windows paths (``C:\...`` or ``dir\sub\`` string literals)

Existing findings in the original files are frozen in ``BASELINE`` (a ratchet):
a file may never get *more* findings, and when a fix lowers a count, the
baseline must be lowered in the same commit.
"""

import ast
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ["main.py", "tools", "toolkit"]

BASELINE = {
    "tools/Port_scan/Port_Scan.py": {"open_without_encoding": 4},
}

_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]|\\[A-Za-z][A-Za-z0-9_ -]+\\")
_TEXT_IO_METHODS = {"read_text", "write_text"}


def _source_files():
    for entry in SCAN_ROOTS:
        path = REPO_ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def _mode_of(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        return str(call.args[1].value)
    return "r"


def _findings(tree: ast.AST) -> Counter:
    found: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            keywords = {kw.arg for kw in node.keywords}
            func = node.func
            is_open = isinstance(func, ast.Name) and func.id == "open"
            is_text_method = isinstance(func, ast.Attribute) and func.attr in _TEXT_IO_METHODS
            if is_open and "b" not in _mode_of(node) and "encoding" not in keywords:
                found["open_without_encoding"] += 1
            if is_text_method and "encoding" not in keywords:
                found["open_without_encoding"] += 1
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value:
                    found["shell_true"] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _WINDOWS_PATH.search(node.value):
                found["hardcoded_windows_path"] += 1
    return found


def _scan() -> dict[str, Counter]:
    results = {}
    for path in _source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if ".venv" in path.parts:
            continue
        found = _findings(ast.parse(path.read_text(encoding="utf-8"), filename=rel))
        if found:
            results[rel] = found
    return results


def test_no_backslash_paths_shell_true_or_unencoded_open():
    results = _scan()
    problems = []
    for rel, found in results.items():
        allowed = BASELINE.get(rel, {})
        for rule, count in found.items():
            if count > allowed.get(rule, 0):
                problems.append(f"{rel}: {rule} = {count} (allowed {allowed.get(rule, 0)})")
    assert problems == [], "new portability problems:\n" + "\n".join(problems)


def test_baseline_is_not_stale():
    """When a fix removes a finding, lower BASELINE in the same commit."""
    results = _scan()
    stale = []
    for rel, rules in BASELINE.items():
        for rule, allowed in rules.items():
            actual = results.get(rel, Counter()).get(rule, 0)
            if actual < allowed:
                stale.append(f"{rel}: {rule} is now {actual}, lower BASELINE from {allowed}")
    assert stale == [], "\n".join(stale)


def test_guard_detects_each_bad_pattern():
    # Raw strings: the samples are Python source code, parsed by ast below.
    bad = r"""
open('a.txt')
open('a.txt', 'w', newline='')
Path('a').write_text('x')
subprocess.run('ls', shell=True)
p = r'C:\Users\me'
q = r'tools\IDS\x'
"""
    good = r"""
open('a.bin', 'rb')
open('a.txt', encoding='utf-8')
Path('a').read_text(encoding='utf-8')
subprocess.run(['ls'])
pattern = r'\d+\.\d+'
eol = '\r\n'
"""
    assert _findings(ast.parse(bad)) == Counter(
        {"open_without_encoding": 3, "shell_true": 1, "hardcoded_windows_path": 2}
    )
    assert _findings(ast.parse(good)) == Counter()
