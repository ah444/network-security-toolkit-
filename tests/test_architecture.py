"""The toolkit's interfaces and layers (plan §2; layered since v5.1 step 1.5d, arranged by
interface since v5.2 step 2.3).

    core/   what both tools share: services/ (ids, recon, ports) and data/ (files, network)
    cli/    the command line: ui/, controllers/, data/ (the menu's child processes)
    web/    the web trainer: ui/, controllers/, services/, data/

``cli`` and ``web`` import ``core`` and never each other; ``core`` imports neither; the package
itself (``toolkit.start``, ``__main__``) picks the interface. Inside every interface the layers
are the same:

    ui/           what the user sees: the web page, the CLI's prompts and printing
    controllers/  requests and flags in, a service called, answers and exit codes out
    services/     what the tools do: no user interface, no files, no network
    data/         files, the network, the environment (no database)

Each layer imports only the layers below it: controllers use ui, services and, from data,
only the settings, logs and the plumbing that starts things (the server's socket and its
instance lock, child processes); ui may import services' types to show them, never data or controllers;
services use data; data imports nothing above it. Inside core the tools stay apart (``ids``
and ``recon`` never import each other). Only ui prints or asks; only the web's controllers
use Flask, only the CLI's controllers and the ``web`` command use argparse; only data opens
files, sockets, HTTP or child processes. The ML-core-only rules are in
``tests/ml/test_core_rules.py``.

The folders of the v5.1 layout (``toolkit/ui``, ``controllers``, ``services``, ``data``) are
gone since v5.2 step 2.3d and must stay empty; their module names are old names.
"""

import ast
from pathlib import Path

INTERFACES = ("core", "cli", "web")
LAYERS = ("ui", "controllers", "services", "data")
V51_FOLDERS = ("ui", "controllers", "services", "data")  # toolkit/<folder>, gone since 2.3d
CONTROLLER_DATA = ("toolkit.core.data.settings", "toolkit.core.data.keys",
                   "toolkit.core.data.logs", "toolkit.web.data.net",
                   "toolkit.web.data.instance_lock", "toolkit.cli.data.processes")
ARGPARSE_OUTSIDE_CLI = ("toolkit.web.controllers.command",)  # the web command's flags
# Jinja2 outside the web: only the scan report's HTML, a file (F5). No other module.
JINJA2_OUTSIDE_WEB = ("toolkit.core.data.recon.report_render",)
WEB_FRAMEWORK = {"flask", "werkzeug", "jinja2", "waitress"}
OUTSIDE = {"socket", "ssl", "requests", "urllib", "http", "subprocess", "webbrowser"}
FILE_METHODS = {"read_text", "read_bytes", "write_text", "write_bytes", "unlink", "mkdir",
                "rmdir", "iterdir", "glob", "rglob", "touch"}
# (toolkit.web and toolkit.cli, the v5.0 modules, are names again: the interfaces of step 2.3)
OLD_NAMES = ("toolkit.ids_core", "toolkit.config", "toolkit.logging_setup", "toolkit.ports",
             "toolkit.bundle_files", "toolkit.run_files", "toolkit.recon", "toolkit.ids_cli",
             "tools",
             # the v5.1 layers, arranged by interface in v5.2 step 2.3
             *(f"toolkit.{folder}" for folder in V51_FOLDERS))
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


def place(module: str) -> tuple[str, str, str]:
    """(interface, layer, tool): ``toolkit.core.services.ids.split`` -> ``("core",
    "services", "ids")``; ``toolkit.web.controllers.api`` -> ``("web", "controllers", "")``;
    the package itself, ``toolkit.start`` and ``__main__`` -> ``("top", "top", "")``; a
    folder of an interface (``toolkit.core``, ``toolkit.core.services``) -> its own layer
    or ``"top"``."""
    parts = module.split(".")
    if len(parts) > 1 and parts[1] in INTERFACES:
        layer = parts[2] if len(parts) > 2 and parts[2] in LAYERS else "top"
        tool = parts[3] if layer != "top" and len(parts) > 3 and parts[3] in TOOLS else ""
        return parts[1], layer, tool
    return "top", "top", ""


ALLOWED_LAYERS = {  # layer -> the layers it may import
    "data": {"data"},
    "services": {"services", "data"},
    "ui": {"ui", "services"},
    "controllers": {"controllers", "ui", "services", "data"},
    "top": {"controllers", "top"},
}
ALLOWED_INTERFACES = {  # interface -> the interfaces it may import
    "core": {"core"},
    "cli": {"cli", "core"},
    "web": {"web", "core"},
    "top": {"cli", "web", "top"},
}


def _import_problems(module, here, name):
    """What is wrong with ``module`` (at ``here``) importing the toolkit name ``name``."""
    (interface, layer, tool), (there_interface, there_layer, there_tool) = here, place(name)
    problems = []
    if name == "toolkit":
        return problems
    if there_interface not in ALLOWED_INTERFACES[interface]:
        problems.append(f"({interface}) imports {name} ({there_interface}): cli and web share "
                        "only core")
    if there_layer not in ALLOWED_LAYERS[layer]:
        problems.append(f"({layer}) imports {name} ({there_layer})")
    if layer == "controllers" and there_layer == "data" and not _within(name, CONTROLLER_DATA):
        problems.append(f"(controllers) imports {name}: use a service")
    if {tool, there_tool} == {"ids", "recon"}:
        problems.append(f"imports {name} ({tool} and {there_tool} are independent)")
    return problems


def _import_violations(module, is_package, tree):
    """The import rules ``module`` breaks: old names, interfaces and layers, the web
    framework, argparse, and the outside world outside the data layer."""
    here = place(module)
    interface, layer, _ = here
    found = []
    for name, line in _imported(tree, module, is_package):
        top = name.split(".")[0]
        if _within(name, OLD_NAMES):
            found.append(f"{module}:{line} imports the old name {name}")
            continue
        if top == "toolkit":
            found += [f"{module}:{line} {problem}"
                      for problem in _import_problems(module, here, name)]
        if top in WEB_FRAMEWORK and (interface, layer) != ("web", "controllers") and not (
                top == "jinja2" and module in JINJA2_OUTSIDE_WEB):
            found.append(f"{module}:{line} imports the web framework ({name})")
        if top == "argparse" and (interface, layer) != ("cli", "controllers") and \
                module not in ARGPARSE_OUTSIDE_CLI:
            found.append(f"{module}:{line} imports argparse")
        if top in OUTSIDE and layer != "data":
            found.append(f"{module}:{line} imports {name}: only the data layer goes outside")
    return found


def _call_violations(module, tree):
    """The call rules ``module`` breaks: print() and input() outside the UI, files opened
    outside the data layer."""
    _, layer, _ = place(module)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("print", "input") and layer != "ui":
            found.append(f"{module}:{node.lineno} calls {func.id}(): only the UI talks to "
                         "the user")
        if layer != "data" and (
                (isinstance(func, ast.Name) and func.id == "open")
                or (isinstance(func, ast.Attribute) and func.attr in FILE_METHODS)):
            what = func.id if isinstance(func, ast.Name) else func.attr
            found.append(f"{module}:{node.lineno} calls {what}(): only the data layer "
                         "reads or writes files")
    return found


def layer_violations(root: Path) -> list[str]:
    """Every broken interface or layer rule in ``root/toolkit``, as readable messages (per
    file: its imports, then its calls)."""
    found = []
    for path in sorted((root / "toolkit").rglob("*.py")):
        module, is_package = _module_name(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += _import_violations(module, is_package, tree)
        found += _call_violations(module, tree)
    return found


def test_layers_hold(repo_root):
    for folder in ("core/services/ids", "core/services/recon", "core/data/ids",
                   "core/data/recon", "cli/ui", "cli/controllers", "cli/data", "web/ui",
                   "web/controllers", "web/services", "web/data"):
        # the rules must really see every interface, layer and tool
        assert (repo_root / "toolkit" / folder / "__init__.py").is_file(), folder
    assert layer_violations(repo_root) == []


def test_the_v51_folders_hold_no_file(repo_root):
    """v5.2 step 2.3d: nothing is left in, or added to, toolkit/ui, controllers, services or
    data. A ``__pycache__`` that git doesn't track may stay on a computer after the move."""
    left = [path.relative_to(repo_root).as_posix()
            for folder in V51_FOLDERS for path in (repo_root / "toolkit" / folder).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts]
    assert left == []


def test_the_folder_check_sees_a_file(tmp_path):
    (tmp_path / "toolkit" / "ui" / "__pycache__").mkdir(parents=True)
    (tmp_path / "toolkit" / "ui" / "__pycache__" / "x.pyc").write_bytes(b"")
    test_the_v51_folders_hold_no_file(tmp_path)  # a cache only: fine
    (tmp_path / "toolkit" / "data" / "web").mkdir(parents=True)
    (tmp_path / "toolkit" / "data" / "web" / "net.py").write_text("", encoding="utf-8")
    try:
        test_the_v51_folders_hold_no_file(tmp_path)
    except AssertionError:
        return
    raise AssertionError("a file left in toolkit/data was not found")


def test_each_rule_is_detected(tmp_path):
    files = {
        "toolkit/__init__.py": "",
        "toolkit/start.py": "from .cli.controllers import main\nfrom .web.controllers import api\n"
                            "from .core.data import settings\n",
        "toolkit/core/__init__.py": "",
        "toolkit/core/data/__init__.py": "",
        "toolkit/core/data/settings.py": "from ..services import ids\n",
        "toolkit/core/data/recon/__init__.py": "",
        "toolkit/core/data/recon/nvd.py": "from ..ids import bundle_files\n",
        "toolkit/core/data/recon/report_render.py": "import jinja2\nimport flask\n",
        "toolkit/core/data/recon/scan_files.py": "from jinja2 import Template\n",
        "toolkit/core/data/ids/__init__.py": "",
        "toolkit/core/data/ids/bundle_files.py": "import socket\nprint('x')\n",
        "toolkit/core/services/__init__.py": "",
        "toolkit/core/services/ids/__init__.py": "",
        "toolkit/core/services/ids/split.py": "import flask\nimport requests\n"
                                              "from ....ids_core import x\n",
        "toolkit/core/services/ids/runs.py": "from pathlib import Path\nPath('a').write_text('b')\n"
                                             "open('c')\ninput('?')\n",
        "toolkit/core/services/recon/__init__.py": "",
        "toolkit/core/services/recon/scan.py": "from ..ids import runs\n"
                                               "from ....cli.controllers import x\n",
        "toolkit/cli/__init__.py": "",
        "toolkit/cli/ui/__init__.py": "",
        "toolkit/cli/ui/menu.py": "from ...core.data import settings\nprint('fine')\n",
        "toolkit/cli/controllers/__init__.py": "",
        "toolkit/cli/controllers/scan.py": "import argparse\nfrom ...core.data.recon import nvd\n"
                                           "from ...core.data import settings\nfrom ..ui import menu\n"
                                           "import flask\nfrom ...web.controllers import api\n",
        "toolkit/web/__init__.py": "",
        "toolkit/web/controllers/__init__.py": "import flask\nimport argparse\n",
        "toolkit/web/controllers/api.py": "from ...cli.ui import menu\n",
        "toolkit/web/controllers/command.py": "import argparse\nfrom ...core.data import settings\n",
        "toolkit/web/services/__init__.py": "",
        "toolkit/web/services/jobs.py": "from ...core.services.ids import runs\nimport flask\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    found = "\n".join(layer_violations(tmp_path))

    expected = [
        "toolkit.start:3 (top) imports toolkit.core.data.settings (core): cli and web share only",
        "toolkit.start:3 (top) imports toolkit.core.data.settings (data)",
        "toolkit.core.data.settings:1 (data) imports toolkit.core.services.ids (services)",
        "toolkit.core.data.recon.nvd:1 imports toolkit.core.data.ids.bundle_files (recon and "
        "ids are",
        "toolkit.core.data.ids.bundle_files:2 calls print()",
        "toolkit.core.data.recon.report_render:2 imports the web framework (flask)",
        "toolkit.core.data.recon.scan_files:1 imports the web framework (jinja2.Template)",
        "toolkit.core.services.ids.split:1 imports the web framework (flask)",
        "toolkit.core.services.ids.split:2 imports requests: only the data layer goes outside",
        "toolkit.core.services.ids.split:3 imports the old name toolkit.ids_core",
        "toolkit.core.services.ids.runs:2 calls write_text()",
        "toolkit.core.services.ids.runs:3 calls open()",
        "toolkit.core.services.ids.runs:4 calls input()",
        "toolkit.core.services.recon.scan:1 imports toolkit.core.services.ids.runs (recon and "
        "ids are",
        "toolkit.core.services.recon.scan:2 (core) imports toolkit.cli.controllers.x (cli)",
        "toolkit.core.services.recon.scan:2 (services) imports toolkit.cli.controllers.x "
        "(controllers)",
        "toolkit.cli.ui.menu:1 (ui) imports toolkit.core.data.settings (data)",
        "toolkit.cli.controllers.scan:2 (controllers) imports toolkit.core.data.recon.nvd: use "
        "a service",
        "toolkit.cli.controllers.scan:5 imports the web framework (flask)",
        "toolkit.cli.controllers.scan:6 (cli) imports toolkit.web.controllers.api (web)",
        "toolkit.web.controllers:2 imports argparse",
        "toolkit.web.controllers.api:1 (web) imports toolkit.cli.ui.menu (cli)",
        "toolkit.web.services.jobs:2 imports the web framework (flask)",
    ]
    for line in expected:
        assert line in found, line
    assert len(found.splitlines()) == len(expected), found
    assert "toolkit.core.data.ids.bundle_files:1 imports socket" not in found  # data may
    assert "toolkit.cli.controllers.scan:1" not in found  # the command line uses argparse
    assert "toolkit.cli.controllers.scan:3" not in found  # settings: allowed
    assert "toolkit.web.controllers:1" not in found  # the web routes use Flask
    assert "toolkit.web.controllers.command" not in found  # the web command's flags
    assert "toolkit.core.data.recon.report_render:1" not in found  # the report's Jinja2 only
    assert "toolkit.start:1" not in found and "toolkit.start:2" not in found  # it picks one
    assert "toolkit.web.services.jobs:1" not in found  # web uses core


def test_no_tools_folder_in_git(repo_root):
    """Step 1.4a: the scripts moved into toolkit/; git tracks nothing under tools/."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "tools"], cwd=repo_root, capture_output=True,
                             text=True, encoding="utf-8", check=True).stdout.split()
    assert tracked == []
