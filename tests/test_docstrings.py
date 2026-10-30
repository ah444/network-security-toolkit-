"""v5.2 step 2.3p: every module, class, function and method of toolkit/ has a docstring.

Exempt: dunder methods (``__init__``, ``__call__``, ...: the class's docstring covers them)
and functions defined inside a function (details of a documented function, which says what
is not obvious about them). Whether a docstring is right is reviewed, not tested; this test
only keeps any from being left out."""

import ast

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _undocumented(tree, path):
    """``path:line kind name`` of each class, function and method of ``tree`` without a
    docstring, under the rules above."""
    missing = []

    def walk(node, parents):
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
                walk(child, parents)
                continue
            is_class = isinstance(child, ast.ClassDef)
            nested = any(isinstance(parent, _FUNCTIONS) for parent in parents)
            dunder = child.name.startswith("__") and child.name.endswith("__")
            if not nested and (is_class or not dunder) and ast.get_docstring(child) is None:
                kind = "class" if is_class else (
                    "method" if parents and isinstance(parents[-1], ast.ClassDef) else "function")
                name = ".".join([parent.name for parent in parents] + [child.name])
                missing.append(f"{path}:{child.lineno} {kind} {name}")
            walk(child, [*parents, child])

    if ast.get_docstring(tree) is None:
        missing.append(f"{path}:1 module")
    walk(tree, [])
    return missing


def test_everything_has_a_docstring(repo_root):
    missing = []
    for path in sorted((repo_root / "toolkit").rglob("*.py")):
        if "__pycache__" not in path.parts:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            missing += _undocumented(tree, path.relative_to(repo_root).as_posix())
    assert missing == []


def test_the_check_sees_each_case():
    source = '''
class Plain:
    def method(self): pass
    def __init__(self): pass
    class Inner:
        """Documented."""
        def deep(self): pass

def function():
    def nested(): pass
    class Local: pass

async def waits():
    """Documented."""

def _private():
    """Documented."""
'''
    assert _undocumented(ast.parse(source), "m.py") == [
        "m.py:1 module", "m.py:2 class Plain", "m.py:3 method Plain.method",
        "m.py:7 method Plain.Inner.deep", "m.py:9 function function"]
