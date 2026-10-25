"""v5.2 step 2.3c: ``toolkit.start``, the one entry point, picks the interface from the first
word: ``web`` the web trainer's command, anything else the command line. In a real process,
running one never loads the other."""

import json
import subprocess
import sys

import pytest

from toolkit import start


def test_web_goes_to_the_web_command(monkeypatch):
    received = []
    monkeypatch.setattr("toolkit.web.controllers.command.main",
                        lambda argv, prog: received.append((argv, prog)) or 5)
    assert start.main(["web", "--port", "5001"]) == 5
    assert received == [(["--port", "5001"], "python -m toolkit web")]


@pytest.mark.parametrize("argv", [[], ["scan", "-h"], ["ids", "--list"], ["--help"], ["bogus"]])
def test_everything_else_goes_to_the_command_line(monkeypatch, argv):
    received = []
    monkeypatch.setattr("toolkit.cli.controllers.main.main",
                        lambda args: received.append(args) or 3)
    assert start.main(argv) == 3
    assert received == [argv]


def _loaded(repo_root, argv):
    """The toolkit modules (and whether Flask) a fresh process has loaded after
    ``start.main(argv)`` with every command's real work replaced."""
    code = (
        "import json, sys\n"
        "import toolkit.core.data.settings\n"
        "from toolkit import start\n"
        "try:\n"
        f"    start.main({argv!r})\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(json.dumps({'modules': sorted(m for m in sys.modules if m.startswith('toolkit')),\n"
        "                  'flask': 'flask' in sys.modules}))\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True,
                          text=True, encoding="utf-8", check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_command_line_loads_nothing_of_the_web_trainer(repo_root):
    loaded = _loaded(repo_root, ["scan", "--help"])
    assert "toolkit.cli.controllers.scan" in loaded["modules"]
    web = [m for m in loaded["modules"] if ".web" in m or m.startswith("toolkit.controllers")]
    assert web == [] and loaded["flask"] is False


def test_the_web_command_loads_nothing_of_the_command_line(repo_root):
    loaded = _loaded(repo_root, ["web", "--help"])
    assert "toolkit.web.controllers.command" in loaded["modules"]
    assert [m for m in loaded["modules"] if m.startswith("toolkit.cli")] == []
