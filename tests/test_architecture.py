"""The layers of the toolkit package (plan §2; layered since v5.1 step 1.5d).

    ui/           what the user sees: the web page, the CLI's prompts and printing
    controllers/  requests and flags in, a service called, answers and exit codes out
    services/     what the tools do: no user interface, no files, no network
    data/         files, the network, the environment (no database)

Each layer imports only the layers below it: controllers use ui, services and, from data,
only the settings, logs and the plumbing that starts things (the server's socket, child
processes); ui may import services' types to show them, never data or controllers;
services use data; data imports nothing above it. Inside services and data the tools stay
apart (``ids`` and ``recon`` never import each other; the web trainer is the IDS's).
Only ui prints or asks; only controllers use Flask or argparse; only data opens files,
sockets, HTTP or child processes. The ML-core-only rules are in ``tests/ml/test_core_rules.py``.
"""

import ast
from pathlib import Path

LAYERS = ("ui", "controllers", "services", "data")
# Until step 1.5d's controller and UI commit, these are the presentation code.
TRANSITIONAL = ("toolkit.web", "toolkit.ids_cli", "toolkit.cli", "toolkit.__main__",
                "toolkit.recon")
CONTROLLER_DATA = ("toolkit.data.settings", "toolkit.data.logs", "toolkit.data.web.net",
                   "toolkit.data.processes")
WEB_FRAMEWORK = {"flask", "werkzeug", "jinja2", "waitress"}
OUTSIDE = {"socket", "ssl", "requests", "urllib", "http", "subprocess", "webbrowser"}
FILE_METHODS = {"read_text", "read_bytes", "write_text", "write_bytes", "unlink", "mkdir",
                "rmdir", "iterdir", "glob", "rglob", "touch"}
OLD_NAMES = ("toolkit.ids_core", "toolkit.config", "toolkit.logging_setup", "toolkit.ports",
             "toolkit.bundle_files", "toolkit.run_files", "toolkit.recon.intel", "tools")
TOOLS = ("ids", "recon")


def _module_name(path: Path, root: Path) -> tuple[str, bool]:
    """Dotted module name of a file under ``root`` (the folder holding ``toolkit/``)."""
    parts = path.relative_to(root).with_suffix("").parts
    is_package = parts[-1] == "__init__"
    return ".".join(parts[:-1] if is_package else parts), is_package


def _imported(tree: ast.AST, module: str, is_package: bool) -> list[tuple[str, int]]:
    """Absolute names of everything a module imports (relative imports resolved); for
    ``from x import y`` that is ``x.y`` (a module or a name in ``x``, both are inside ``x``)."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".") if is_package else module.split(".")[:-1]
                base = base[:len(base) - (node.level - 1)]
                source = ".".join([*base, node.module] if node.module else base)
            else:
                source = node.module or ""
            names += [(f"{source}.{alias.name}", node.lineno) for alias in node.names]
    return names


def _within(name: str, prefixes) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def layer(module: str) -> str:
    """``toolkit.services.ids.split`` -> ``services``; transitional presentation code ->
    ``presentation``; the package itself and ``toolkit.__main__`` -> ``top``."""
    parts = module.split(".")
    if len(parts) > 1 and parts[1] in LAYERS:
        return parts[1]
    if _within(module, TRANSITIONAL) and module != "toolkit.__main__":
        return "presentation"
    return "top"


def tool(module: str) -> str:
    """``toolkit.data.recon.nvd`` -> ``recon``; ``toolkit.services.web.jobs`` -> ``web``."""
    parts = module.split(".")
    return parts[2] if len(parts) > 2 and parts[1] in LAYERS else ""


ALLOWED = {  # layer -> the layers it may import
    "data": {"data"},
    "services": {"services", "data"},
    "ui": {"ui", "services"},
    "controllers": {"controllers", "ui", "services", "data"},
    "presentation": {"presentation", "controllers", "ui", "services", "data", "top"},
    "top": {"presentation", "controllers", "top"},
}


def layer_violations(root: Path) -> list[str]:
    """Every broken layer rule in ``root/toolkit``, as readable messages."""
    found = []
    for path in sorted((root / "toolkit").rglob("*.py")):
        module, is_package = _module_name(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        here, here_tool = layer(module), tool(module)
        for name, line in _imported(tree, module, is_package):
            top = name.split(".")[0]
            if _within(name, OLD_NAMES):
                found.append(f"{module}:{line} imports the old name {name}")
                continue
            if top == "toolkit":
                there = layer(name)
                if name != "toolkit" and there not in ALLOWED[here]:
                    found.append(f"{module}:{line} ({here}) imports {name} ({there})")
                if here == "controllers" and there == "data" and \
                        not _within(name, CONTROLLER_DATA):
                    found.append(f"{module}:{line} (controllers) imports {name}: use a service")
                there_tool = tool(name)
                if here in ("services", "data") and {here_tool, there_tool} == {"ids", "recon"}:
                    found.append(f"{module}:{line} imports {name} ({here_tool} and "
                                 f"{there_tool} are independent)")
                if here in ("services", "data") and here_tool == "recon" and there_tool == "web":
                    found.append(f"{module}:{line} imports {name} (the scanner doesn't use the "
                                 "web trainer)")
            if top in WEB_FRAMEWORK and here not in ("controllers", "presentation"):
                found.append(f"{module}:{line} imports the web framework ({name})")
            if top == "argparse" and here not in ("controllers", "presentation"):
                found.append(f"{module}:{line} imports argparse")
            if top in OUTSIDE and here not in ("data", "presentation"):
                found.append(f"{module}:{line} imports {name}: only the data layer goes outside")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("print", "input") and \
                    here not in ("ui", "presentation"):
                found.append(f"{module}:{node.lineno} calls {func.id}(): only the UI talks to "
                             "the user")
            if here not in ("data", "presentation") and (
                    (isinstance(func, ast.Name) and func.id == "open")
                    or (isinstance(func, ast.Attribute) and func.attr in FILE_METHODS)):
                what = func.id if isinstance(func, ast.Name) else func.attr
                found.append(f"{module}:{node.lineno} calls {what}(): only the data layer "
                             "reads or writes files")
    return found


def test_layers_hold(repo_root):
    for folder in ("services/ids", "services/recon", "services/web", "data/ids", "data/recon",
                   "data/web"):  # the rules must really see every layer
        assert (repo_root / "toolkit" / folder / "__init__.py").is_file(), folder
    assert layer_violations(repo_root) == []


def test_each_layer_rule_is_detected(tmp_path):
    files = {
        "toolkit/__init__.py": "",
        "toolkit/data/__init__.py": "",
        "toolkit/data/settings.py": "from ..services import ids\n",
        "toolkit/data/recon/__init__.py": "",
        "toolkit/data/recon/nvd.py": "from ..ids import bundle_files\n",
        "toolkit/data/ids/__init__.py": "",
        "toolkit/data/ids/bundle_files.py": "import socket\nprint('x')\n",
        "toolkit/services/__init__.py": "",
        "toolkit/services/ids/__init__.py": "",
        "toolkit/services/ids/split.py": "import flask\nimport requests\nfrom ...ids_core import x\n",
        "toolkit/services/ids/runs.py": "from pathlib import Path\nPath('a').write_text('b')\n"
                                         "open('c')\ninput('?')\n",
        "toolkit/services/recon/__init__.py": "",
        "toolkit/services/recon/scan.py": "from ..ids import runs\nfrom ...controllers import x\n",
        "toolkit/ui/__init__.py": "",
        "toolkit/ui/cli.py": "from ..data import settings\nprint('fine')\n",
        "toolkit/controllers/__init__.py": "",
        "toolkit/controllers/scan.py": "import argparse\nfrom ..data.recon import nvd\n"
                                        "from ..data import settings\nfrom ..ui import cli\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    found = "\n".join(layer_violations(tmp_path))

    expected = [
        "toolkit.data.settings:1 (data) imports toolkit.services.ids (services)",
        "toolkit.data.recon.nvd:1 imports toolkit.data.ids.bundle_files (recon and ids are",
        "toolkit.data.ids.bundle_files:2 calls print()",
        "toolkit.services.ids.split:1 imports the web framework (flask)",
        "toolkit.services.ids.split:2 imports requests: only the data layer goes outside",
        "toolkit.services.ids.split:3 imports the old name toolkit.ids_core",
        "toolkit.services.ids.runs:2 calls write_text()",
        "toolkit.services.ids.runs:3 calls open()",
        "toolkit.services.ids.runs:4 calls input()",
        "toolkit.services.recon.scan:1 imports toolkit.services.ids.runs (recon and ids are",
        "toolkit.services.recon.scan:2 (services) imports toolkit.controllers.x (controllers)",
        "toolkit.ui.cli:1 (ui) imports toolkit.data.settings (data)",
        "toolkit.controllers.scan:2 (controllers) imports toolkit.data.recon.nvd: use a service",
    ]
    for line in expected:
        assert line in found, line
    assert len(found.splitlines()) == len(expected), found
    assert "toolkit.data.ids.bundle_files:1 imports socket" not in found  # data may
    assert "argparse" not in found and "toolkit.controllers.scan:3" not in found


def test_no_tools_folder_in_git(repo_root):
    """Step 1.4a: the scripts moved into toolkit/; git tracks nothing under tools/."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "tools"], cwd=repo_root, capture_output=True,
                             text=True, encoding="utf-8", check=True).stdout.split()
    assert tracked == []
