"""Module boundaries of the toolkit package (plan §2, v5.1 step 1.1a).

The package is a modular monolith: the ML core (``ids_core``) and the scanner (``recon``)
never import each other, nothing in ``toolkit`` imports the old ``tools/`` scripts (gone
since v5.1 step 1.4a) or a presentation module, and only presentation modules may print or
use the web framework. Presentation is ``web``, ``ids_cli`` and any ``cli`` or ``__main__``
module (``toolkit/recon/cli.py`` is the scanner's user side).
The ML-core-only rules (no network) are in ``tests/ml/test_core_rules.py``.
"""

import ast
from pathlib import Path

PRESENTATION = {"web", "ids_cli", "cli", "__main__"}  # first name under toolkit/
USER_SIDE = {"cli", "__main__"}  # a module with this last name talks to the user
WEB_FRAMEWORK = {"flask", "werkzeug", "jinja2", "waitress"}
INDEPENDENT = ("ids_core", "recon")


def _module_name(path: Path, root: Path) -> tuple[str, bool]:
    """Dotted module name of a file under ``root`` (the folder holding ``toolkit/``)."""
    parts = path.relative_to(root).with_suffix("").parts
    is_package = parts[-1] == "__init__"
    return ".".join(parts[:-1] if is_package else parts), is_package


def _imported(tree: ast.AST, module: str, is_package: bool) -> list[str]:
    """Absolute names of everything a module imports (relative imports resolved)."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if not node.level:
                names.append(node.module or "")
                continue
            base = module.split(".") if is_package else module.split(".")[:-1]
            base = base[:len(base) - (node.level - 1)]
            if node.module:
                names.append(".".join([*base, node.module]))
            else:  # from . import x
                names += [".".join([*base, alias.name]) for alias in node.names]
    return names


def _presentation(module: str) -> bool:
    return _area(module) in PRESENTATION or module.split(".")[-1] in USER_SIDE


def _area(module: str) -> str:
    """``toolkit.recon.scanner`` -> ``recon``; top-level modules give their own name."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def boundary_violations(root: Path) -> list[str]:
    """Every broken boundary rule in ``root/toolkit``, as readable messages."""
    found = []
    for path in sorted((root / "toolkit").rglob("*.py")):
        module, is_package = _module_name(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        area = _area(module)
        presentation = _presentation(module)
        for name in _imported(tree, module, is_package):
            top = name.split(".")[0]
            if top == "tools":
                found.append(f"{module} imports the script package {name}")
            if top == "toolkit" and area in INDEPENDENT:
                other = _area(name)
                if other in INDEPENDENT and other != area:
                    found.append(f"{module} imports {name} ({area} and {other} are independent)")
            if not presentation and top == "toolkit" and _presentation(name):
                found.append(f"{module} imports the presentation module {name}")
            if not presentation and top in WEB_FRAMEWORK:
                found.append(f"{module} imports the web framework ({name})")
        if not presentation:
            prints = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "print"]
            if prints:
                found.append(f"{module} calls print() at lines {prints}; use logging")
    return found


def test_package_boundaries_hold(repo_root):
    for area in INDEPENDENT:  # the rules must really see both packages
        assert (repo_root / "toolkit" / area).is_dir(), f"toolkit/{area} is missing"
    assert boundary_violations(repo_root) == []


def test_each_boundary_rule_is_detected(tmp_path):
    files = {
        "toolkit/__init__.py": "",
        "toolkit/config.py": "from .web import app\n",
        "toolkit/run_files.py": "import tools.Port_scan\n",
        "toolkit/ids_core/__init__.py": "",
        "toolkit/ids_core/split.py": "from ..recon import scanner\n",
        "toolkit/recon/__init__.py": "",
        "toolkit/recon/scanner.py": "import flask\nprint('scanning')\n",
        "toolkit/recon/cli.py": "from .scanner import scan\nprint('ok')\n",
        "toolkit/recon/export.py": "from .cli import main\n",
        "toolkit/recon/nvd.py": "from toolkit.ids_core.runs import train\n",
        "toolkit/web/__init__.py": "import flask\nfrom toolkit.recon import scanner\nprint('ok')\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    violations = boundary_violations(tmp_path)
    found = "\n".join(violations)

    assert "toolkit.config imports the presentation module toolkit.web" in found
    assert "toolkit.run_files imports the script package tools.Port_scan" in found
    assert "toolkit.ids_core.split imports toolkit.recon " in found
    assert "toolkit.recon.nvd imports toolkit.ids_core.runs " in found
    assert "toolkit.recon.scanner imports the web framework (flask)" in found
    assert "toolkit.recon.scanner calls print() at lines [2]" in found
    assert "toolkit.recon.export imports the presentation module toolkit.recon.cli" in found
    assert not [v for v in violations if v.startswith(("toolkit.web", "toolkit.recon.cli"))]
    assert len(violations) == 7


def test_no_tools_folder_in_git(repo_root):
    """Step 1.4a: the scripts moved into toolkit/; git tracks nothing under tools/."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "tools"], cwd=repo_root, capture_output=True,
                             text=True, encoding="utf-8", check=True).stdout.split()
    assert tracked == []
