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
only the settings, logs and the plumbing that starts things (the server's socket, child
processes); ui may import services' types to show them, never data or controllers;
services use data; data imports nothing above it. Inside core the tools stay apart (``ids``
and ``recon`` never import each other). Only ui prints or asks; only the web's controllers
use Flask, only the CLI's controllers and the ``web`` command use argparse; only data opens
files, sockets, HTTP or child processes. The ML-core-only rules are in
``tests/ml/test_core_rules.py``.

Until v5.2 step 2.3d moves them, the packages of the v5.1 layout that are not moved yet are
placed by ``NOT_MOVED_YET`` (and the CLI's ``web`` command by ``CROSSING_UNTIL_STEP_23C``);
both lists name only what is still there, and are removed with the last move.
"""

import ast
from pathlib import Path

INTERFACES = ("core", "cli", "web")
LAYERS = ("ui", "controllers", "services", "data")
# Temporary (v5.2 steps 2.3b-2.3d): v5.1-layout package -> (interface, layer); the longest
# prefix wins. Only packages that have not moved yet.
NOT_MOVED_YET = {
    "toolkit.ui.cli": ("cli", "ui"),
    "toolkit.ui.cli.web": ("web", "ui"),  # the web command's console messages
    "toolkit.ui.web": ("web", "ui"),
    "toolkit.controllers.cli": ("cli", "controllers"),
    "toolkit.controllers.cli.web": ("web", "controllers"),  # the web command
    "toolkit.controllers.web": ("web", "controllers"),
    "toolkit.services.web": ("web", "services"),
    "toolkit.data.web": ("web", "data"),
    "toolkit.data.processes": ("cli", "data"),
}
# Temporary until step 2.3c: the CLI's dispatcher still imports the web command (then
# toolkit.start picks the interface).
CROSSING_UNTIL_STEP_23C = {("toolkit.controllers.cli.main", "toolkit.controllers.cli.web")}
CONTROLLER_DATA = ("toolkit.core.data.settings", "toolkit.core.data.keys",
                   "toolkit.core.data.logs", "toolkit.data.web.net", "toolkit.data.processes")
ARGPARSE_OUTSIDE_CLI = ("toolkit.web.controllers.command", "toolkit.controllers.cli.web")
WEB_FRAMEWORK = {"flask", "werkzeug", "jinja2", "waitress"}
OUTSIDE = {"socket", "ssl", "requests", "urllib", "http", "subprocess", "webbrowser"}
FILE_METHODS = {"read_text", "read_bytes", "write_text", "write_bytes", "unlink", "mkdir",
                "rmdir", "iterdir", "glob", "rglob", "touch"}
# (toolkit.web and toolkit.cli, the v5.0 modules, are names again: the interfaces of step 2.3)
OLD_NAMES = ("toolkit.ids_core", "toolkit.config", "toolkit.logging_setup", "toolkit.ports",
             "toolkit.bundle_files", "toolkit.run_files", "toolkit.recon", "toolkit.ids_cli",
             "tools",
             # moved into toolkit.core in v5.2 step 2.3b
             "toolkit.services.ids", "toolkit.services.recon", "toolkit.services.ports",
             "toolkit.data.settings", "toolkit.data.keys", "toolkit.data.logs",
             "toolkit.data.files", "toolkit.data.ids", "toolkit.data.recon")
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
    old = [p for p in NOT_MOVED_YET if module == p or module.startswith(p + ".")]
    if old:
        interface, layer = NOT_MOVED_YET[max(old, key=len)]
        return interface, layer, ""
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
    if name == "toolkit" or (module, name.rsplit(".", 1)[0]) in CROSSING_UNTIL_STEP_23C \
            or (module, name) in CROSSING_UNTIL_STEP_23C:
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


def layer_violations(root: Path) -> list[str]:
    """Every broken interface or layer rule in ``root/toolkit``, as readable messages."""
    found = []
    for path in sorted((root / "toolkit").rglob("*.py")):
        module, is_package = _module_name(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        here = place(module)
        interface, layer, _ = here
        for name, line in _imported(tree, module, is_package):
            top = name.split(".")[0]
            if _within(name, OLD_NAMES):
                found.append(f"{module}:{line} imports the old name {name}")
                continue
            if top == "toolkit":
                found += [f"{module}:{line} {problem}"
                          for problem in _import_problems(module, here, name)]
            if top in WEB_FRAMEWORK and (interface, layer) != ("web", "controllers"):
                found.append(f"{module}:{line} imports the web framework ({name})")
            if top == "argparse" and (interface, layer) != ("cli", "controllers") and \
                    module not in ARGPARSE_OUTSIDE_CLI:
                found.append(f"{module}:{line} imports argparse")
            if top in OUTSIDE and layer != "data":
                found.append(f"{module}:{line} imports {name}: only the data layer goes outside")
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


def test_layers_hold(repo_root):
    for folder in ("core/services/ids", "core/services/recon", "core/data/ids",
                   "core/data/recon", "ui/web", "ui/cli", "controllers/web", "controllers/cli",
                   "services/web", "data/web"):
        # the rules must really see every interface, layer and tool
        assert (repo_root / "toolkit" / folder / "__init__.py").is_file(), folder
    assert layer_violations(repo_root) == []


def test_the_temporary_rules_name_only_what_is_still_there(repo_root):
    """NOT_MOVED_YET and CROSSING_UNTIL_STEP_23C shrink with each move; a stale entry fails."""
    for prefix in NOT_MOVED_YET:
        path = repo_root / Path(*prefix.split("."))
        assert path.is_dir() or path.with_suffix(".py").is_file(), prefix
    for module, name in CROSSING_UNTIL_STEP_23C:
        for dotted in (module, name):
            assert (repo_root / Path(*dotted.split("."))).with_suffix(".py").is_file(), dotted


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
    assert "toolkit.start:1" not in found and "toolkit.start:2" not in found  # it picks one
    assert "toolkit.web.services.jobs:1" not in found  # web uses core


def test_no_tools_folder_in_git(repo_root):
    """Step 1.4a: the scripts moved into toolkit/; git tracks nothing under tools/."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "tools"], cwd=repo_root, capture_output=True,
                             text=True, encoding="utf-8", check=True).stdout.split()
    assert tracked == []
