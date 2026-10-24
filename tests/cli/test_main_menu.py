"""The entry point ``python -m toolkit`` (v5.1 step 1.4b, plan D4, R9): flat menu and
subcommands. Step 0.7's launcher bugs stay covered: B15 ("n" exited the whole app) and R8
(menu recursion)."""

import ast
import builtins
import importlib.util
import inspect
import subprocess
import sys
import time

import pytest

from toolkit import cli


@pytest.fixture
def menu(monkeypatch):
    """toolkit.cli with the child processes replaced by a recorder; sleeping fails."""
    calls = {"tools": [], "depths": [], "prompts": []}

    def fake_run_child(command):
        calls["tools"].append(command)
        calls["depths"].append(len(inspect.stack(0)))
        return 0

    def no_sleep(seconds):
        raise AssertionError("the menu must not pause")

    monkeypatch.setattr(cli, "run_child", fake_run_child)
    monkeypatch.setattr(time, "sleep", no_sleep)
    return calls


def _feed(monkeypatch, calls, answers):
    remaining = iter(answers)

    def fake_input(prompt=""):
        calls["prompts"].append(prompt)
        answer = next(remaining)
        if isinstance(answer, type) and issubclass(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", fake_input)
    return remaining


def test_banner_once_and_menu_after_every_tool(menu, monkeypatch, capsys):
    remaining = _feed(monkeypatch, menu, ["1", "2", "3", "4"])
    assert cli.menu() == 0
    out = capsys.readouterr().out
    assert list(remaining) == []
    assert menu["tools"] == ["scan", "ids", "web"]
    assert out.count("Network Recon and Detection Tool") == 1, "banner printed once"
    assert out.count("1. Network Recon") == 4, "the choices again after every tool"


def test_a_tool_goes_back_to_the_menu_without_questions(menu, monkeypatch):
    """No "Proceed to Intrusion Detection?" or "another detection?" prompts (B15)."""
    remaining = _feed(monkeypatch, menu, ["1", "2", "4"])
    cli.menu()
    assert list(remaining) == []
    assert menu["tools"] == ["scan", "ids"]
    assert set(menu["prompts"]) == {"Enter your choice [1-4]: "}


def test_no_recursion_depth_growth(menu, monkeypatch):
    """Running tools many times must not grow the call stack (it used to recurse, R8)."""
    cycles = 50
    _feed(monkeypatch, menu, ["2", "3"] * cycles + ["4"])
    cli.menu()
    assert len(menu["tools"]) == 2 * cycles
    assert max(menu["depths"]) - min(menu["depths"]) == 0, "call stack grows per run"


@pytest.mark.parametrize("answer", ["4", "q", "Q", " quit ", "exit"])
def test_exit_choices(menu, monkeypatch, capsys, answer):
    _feed(monkeypatch, menu, [answer])
    assert cli.menu() == 0
    assert menu["tools"] == []
    assert "Goodbye!" in capsys.readouterr().out


def test_wrong_choice_says_so_and_asks_again(menu, monkeypatch, capsys):
    remaining = _feed(monkeypatch, menu, ["9", "", "4"])
    cli.menu()
    assert list(remaining) == []
    assert menu["tools"] == []
    assert capsys.readouterr().out.count("is not a choice: enter 1, 2, 3 or 4") == 2


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_ctrl_c_or_end_of_input_at_the_menu_exits(menu, monkeypatch, capsys, stop):
    _feed(monkeypatch, menu, ["1", stop])
    assert cli.menu() == 0
    assert menu["tools"] == ["scan"]
    assert "Goodbye!" in capsys.readouterr().out


def test_menu_tools_run_as_child_processes(request, monkeypatch):
    """Each choice is ``python -m toolkit <command>`` from the project folder, so its
    memory is freed and Ctrl+C or a crash in it brings the menu back (step 1.4a)."""
    started = []

    def fake_run(args, cwd=None, check=None):
        started.append((args, cwd))
        return subprocess.CompletedProcess(args, 3)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    codes = [cli.run_child(name) for name in cli.MENU_COMMANDS.values()]
    assert codes == [3, 3, 3]
    assert started == [([sys.executable, "-m", "toolkit", name], request.config.rootpath)
                       for name in ("scan", "ids", "web")]
    assert importlib.util.find_spec("toolkit.__main__") is not None

    # an installed copy (no project folder, step 1.5b) starts them in the current folder
    started.clear()
    monkeypatch.setattr(cli, "is_checkout", lambda: False)
    cli.run_child("scan")
    assert started == [([sys.executable, "-m", "toolkit", "scan"], None)]


def test_ctrl_c_in_a_tool_comes_back_to_the_menu(monkeypatch, capsys):
    def interrupted(args, cwd=None, check=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.subprocess, "run", interrupted)
    assert cli.run_child("web") == 130
    assert "Stopped." in capsys.readouterr().out


@pytest.mark.parametrize("name, module", [("scan", "toolkit.recon.cli"),
                                          ("ids", "toolkit.ids_cli.cli"),
                                          ("web", "toolkit.web")])
def test_subcommand_arguments_and_exit_code_reach_the_tool(monkeypatch, name, module):
    received = []

    def fake_run(argv, prog):
        received.append((argv, prog))
        sys.exit(7)

    monkeypatch.setattr(f"{module}.run", fake_run)
    with pytest.raises(SystemExit) as stop:
        cli.main([name, "--list", "a b", "-x"])
    assert stop.value.code == 7
    assert received == [(["--list", "a b", "-x"], f"python -m toolkit {name}")]


def test_help_lists_the_commands(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "usage: python -m toolkit" in out
    for name in ("scan", "ids", "web"):
        assert f"    {name} " in out
    # where the exports, runs, signing key and logs are, to find and back them up (step 1.5b)
    assert f"Data (exports, runs, signing key, logs): {cli.instance_dir()}" in out


@pytest.mark.parametrize("argv", [["bogus"], ["--bogus"], ["Scan"]])
def test_unknown_command_exits_2(capsys, argv):
    with pytest.raises(SystemExit) as stop:
        cli.main(argv)
    assert stop.value.code == 2
    assert "usage: python -m toolkit" in capsys.readouterr().err


@pytest.mark.parametrize("name, module", [("scan", "toolkit.recon.cli"),
                                          ("ids", "toolkit.ids_cli.cli"), ("web", None)])
def test_tool_help_names_the_command(monkeypatch, capsys, name, module):
    if module:  # the tool's log would go to the test instance; not needed for --help
        monkeypatch.setattr(f"{module}.setup_logging", lambda *args, **kwargs: None)

    def no_server():
        raise AssertionError("--help must not start the server")

    monkeypatch.setattr("toolkit.web.run_server", no_server)
    with pytest.raises(SystemExit) as stop:
        cli.main([name, "--help"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: python -m toolkit {name} ")


def test_web_command_starts_the_server(monkeypatch):
    started = []
    monkeypatch.setattr("toolkit.web.run_server", lambda config=None: started.append(True))
    assert cli.main(["web"]) == 0
    assert started == [True]
    with pytest.raises(SystemExit) as stop:
        cli.main(["web", "--no-such-flag"])
    assert stop.value.code == 2
    assert started == [True], "a wrong flag must not start the server"


def test_main_py_is_a_shim_for_the_entry_point(repo_root):
    source = (repo_root / "main.py").read_text(encoding="utf-8")
    assert len(source.splitlines()) == 3
    body = ast.parse(source).body
    assert ast.unparse(body[1]) == "from toolkit.cli import run"
    assert ast.unparse(body[2]) == "run()"


@pytest.mark.parametrize("entry", [["-m", "toolkit"], ["main.py"]])
def test_entry_points_from_the_project_folder(repo_root, entry):
    """The real commands: help exits 0, an unknown command exits 2."""
    shown = subprocess.run([sys.executable, *entry, "--help"], cwd=repo_root,
                           capture_output=True, text=True, timeout=60,
                           stdin=subprocess.DEVNULL)
    assert shown.returncode == 0, shown.stderr
    assert "scan" in shown.stdout and "web" in shown.stdout
    wrong = subprocess.run([sys.executable, *entry, "bogus"], cwd=repo_root,
                           capture_output=True, text=True, timeout=60,
                           stdin=subprocess.DEVNULL)
    assert wrong.returncode == 2
    assert "invalid choice: 'bogus'" in wrong.stderr


def test_walkthrough_script_waits_only_for_prompts_the_tools_print(repo_root):
    """scripts/menu_walkthrough.py (the release gate) follows the flat menu: every prompt
    it waits for is in the toolkit's code, and none of the removed menus is."""
    script = (repo_root / "scripts/menu_walkthrough.py").read_text(encoding="utf-8")
    tree = ast.parse(script)
    constants = {node.targets[0].id: node.value.value for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and isinstance(node.value, ast.Constant)}
    waited = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("answer",
                                                                             "expect"):
            for arg in node.args[:1] if node.func.id == "answer" else node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    waited.add(arg.value)
                elif isinstance(arg, ast.Name) and arg.id in constants:
                    waited.add(constants[arg.id])
    assert len(waited) > 15
    code = "".join(path.read_text(encoding="utf-8")
                   for path in (repo_root / "toolkit").rglob("*.py"))
    assert sorted(p for p in waited if p not in code) == []
    assert 'start("-m", "toolkit")' in script and 'start("main.py")' in script
