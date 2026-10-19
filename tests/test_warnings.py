"""B19: no module may silence warnings globally (it hid sklearn's version-mismatch warning)."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN = [REPO_ROOT / "main.py", *sorted((REPO_ROOT / "tools").rglob("*.py"))]


def _global_ignores(tree: ast.Module) -> list[int]:
    """Line numbers of module-level warnings.filterwarnings/simplefilter('ignore', ...)."""
    lines = []
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        func = stmt.value.func
        if isinstance(func, ast.Attribute) and func.attr in {"filterwarnings", "simplefilter"}:
            args = stmt.value.args
            if args and isinstance(args[0], ast.Constant) and args[0].value == "ignore":
                lines.append(stmt.lineno)
    return lines


def test_no_global_warning_suppression():
    offenders = {}
    for path in SCAN:
        found = _global_ignores(ast.parse(path.read_text(encoding="utf-8")))
        if found:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = found
    assert offenders == {}, f"global warning suppression found: {offenders}"
