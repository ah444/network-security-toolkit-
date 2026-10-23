"""Step 0.7: launcher menu bugs B15 ("n" exits the whole app) and R8 (menu recursion)."""

import builtins
import importlib.util
import inspect
import sys

import pytest


@pytest.fixture
def launcher(request, monkeypatch):
    """main.py with screen/animation/subprocess calls replaced by recorders."""
    spec = importlib.util.spec_from_file_location("launcher_under_test",
                                                  request.config.rootpath / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["launcher_under_test"] = module
    spec.loader.exec_module(module)

    calls = {"tools": [], "depths": []}

    def fake_run_tool(path):
        calls["tools"].append(path)
        calls["depths"].append(len(inspect.stack(0)))

    monkeypatch.setattr(module, "run_tool", fake_run_tool)
    monkeypatch.setattr(module, "welcome_message", lambda: None)
    monkeypatch.setattr(module, "clear_screen", lambda: None)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module, calls


def _feed(monkeypatch, answers):
    remaining = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(remaining))
    return remaining


def test_no_exits_app_on_n(launcher, monkeypatch):
    """Answering 'n' to 'Proceed to Intrusion Detection?' returns to the main menu."""
    module, calls = launcher
    remaining = _feed(monkeypatch, ["1", "n", "4"])  # recon -> no -> exit via menu
    with pytest.raises(SystemExit) as stop:
        module.main_menu()
    assert stop.value.code == 0
    assert list(remaining) == [], "menu must be shown again after 'n' (the '4' was never read)"
    assert calls["tools"] == ["toolkit.recon.cli"]


def test_no_recursion_depth_growth(launcher, monkeypatch):
    """Repeating detections must not grow the call stack (it used to recurse forever)."""
    module, calls = launcher
    cycles = 50
    # main menu 2 (CLI) -> repeat? y -> intrusion menu 1 (CLI) -> repeat? y -> ... -> n -> 4
    answers = ["2"] + ["y", "1"] * cycles + ["n", "4"]
    _feed(monkeypatch, answers)
    with pytest.raises(SystemExit):
        module.main_menu()
    assert len(calls["tools"]) == cycles + 1
    assert max(calls["depths"]) - min(calls["depths"]) <= 2, "call stack grows per repetition"


def test_return_from_intrusion_menu_goes_back_to_main(launcher, monkeypatch):
    module, calls = launcher
    remaining = _feed(monkeypatch, ["1", "y", "3", "4"])  # recon -> yes -> back -> exit
    with pytest.raises(SystemExit):
        module.main_menu()
    assert list(remaining) == []
    assert calls["tools"] == ["toolkit.recon.cli"]


def test_tools_run_as_child_processes(request, monkeypatch):
    """Step 1.4a: each tool is ``python -m <module>`` from the project folder, so its
    memory is freed and Ctrl+C or a crash in it brings the menu back."""
    spec = importlib.util.spec_from_file_location("launcher_child_test",
                                                  request.config.rootpath / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = []
    monkeypatch.setattr(module.subprocess, "run",
                        lambda args, cwd=None: started.append((args, cwd)))
    for name in (module.SCANNER, module.CLI_IDS, module.WEB_IDS):
        module.run_tool(name)
    root = str(request.config.rootpath)
    assert started == [([sys.executable, "-m", name], root) for name in
                       ("toolkit.recon.cli", "toolkit.ids_cli", "toolkit.web")]
    for name in ("toolkit.recon.cli", "toolkit.ids_cli", "toolkit.web"):
        assert importlib.util.find_spec(name) is not None, f"{name} can't be started"
