"""v5.1 step 1.5e: a bad TOOLKIT_* setting is one line naming it and exit code 2, before
the command starts; never a traceback, a silent default, or a limit of 0."""

import re
import subprocess
import sys

import pytest

from toolkit import start
from toolkit.core.data import settings
from toolkit.core.data.settings import (COMMAND_SETTINGS, check_settings, max_workers,
                                   scan_limits, switch, web_config, whole_number)

WEB_NUMBERS = ("TOOLKIT_MAX_UPLOAD_MB", "TOOLKIT_MAX_PREDICT_MB", "TOOLKIT_WORKSPACE_IDLE_MIN",
               "TOOLKIT_MAX_WORKSPACES", "TOOLKIT_MAX_DATA_MB", "TOOLKIT_JOB_MAX_MIN")
NOT_WHOLE = ("abc", "0", "-5", "1.5", "2 MB")


@pytest.mark.parametrize("variable", WEB_NUMBERS)
@pytest.mark.parametrize("bad", NOT_WHOLE)
def test_web_numbers_must_be_whole_numbers_from_1(variable, bad):
    with pytest.raises(ValueError, match=f"^{variable} must be a whole number >= 1, not "
                                         f"{re.escape(repr(bad))}$"):
        web_config({variable: bad})


@pytest.mark.parametrize("bad", NOT_WHOLE)
def test_the_other_numbers_use_the_same_message(bad):
    for variable, read in (("TOOLKIT_MAX_WORKERS", max_workers),
                           ("TOOLKIT_SCAN_MAX_HOSTS", scan_limits),
                           ("TOOLKIT_SCAN_MAX_PROBES", scan_limits)):
        with pytest.raises(ValueError, match=f"^{variable} must be a whole number >= 1"):
            read({variable: bad})


def test_whole_number():
    assert whole_number({}, "TOOLKIT_X", 7) == 7
    assert whole_number({"TOOLKIT_X": " "}, "TOOLKIT_X", 7) == 7
    assert whole_number({"TOOLKIT_X": " 12 "}, "TOOLKIT_X", 7) == 12
    assert web_config({"TOOLKIT_MAX_WORKSPACES": "1"}).max_workspaces == 1


@pytest.mark.parametrize("variable", settings.SWITCHES)
def test_switches_are_0_or_1(variable):
    assert not switch({}, variable) and not switch({variable: "0"}, variable)
    assert not switch({variable: ""}, variable)
    assert switch({variable: "1"}, variable) and switch({variable: " 1 "}, variable)
    for bad in ("true", "yes", "on", "2", "no"):
        with pytest.raises(ValueError, match=f"^{variable} must be 1 \\(on\\) or 0 \\(off\\), "
                                             f"not '{bad}'$"):
            switch({variable: bad}, variable)


def test_nothing_set_is_fine_for_every_command():
    for command in COMMAND_SETTINGS:
        assert check_settings(command, {}) == []


def test_each_command_checks_only_what_it_reads():
    typo = {"TOOLKIT_MAX_UPLOAD_MB": "abc"}  # a web setting
    assert check_settings("web", typo) == ["TOOLKIT_MAX_UPLOAD_MB must be a whole number >= 1, "
                                           "not 'abc'"]
    assert check_settings("scan", typo) == [] and check_settings("ids", typo) == []
    assert check_settings("scan", {"TOOLKIT_SCAN_MAX_HOSTS": "0"}) == [
        "TOOLKIT_SCAN_MAX_HOSTS must be a whole number >= 1, not '0'"]
    assert check_settings("web", {"TOOLKIT_SCAN_MAX_HOSTS": "0"}) == []
    for command in COMMAND_SETTINGS:  # the log level is read by every tool
        assert check_settings(command, {"TOOLKIT_LOG_LEVEL": "verbose"}) == [
            "TOOLKIT_LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, not 'verbose'"]


def test_every_bad_setting_is_listed_in_a_fixed_order():
    bad = {"TOOLKIT_NO_BROWSER": "true", "TOOLKIT_LOG_LEVEL": "loud", "TOOLKIT_PORT": "abc",
           "TOOLKIT_MAX_WORKERS": "0", "TOOLKIT_JOB_MAX_MIN": "-5"}
    assert [problem.split()[0] for problem in check_settings("web", bad)] == [
        "TOOLKIT_LOG_LEVEL", "TOOLKIT_JOB_MAX_MIN", "TOOLKIT_MAX_WORKERS", "TOOLKIT_PORT",
        "TOOLKIT_NO_BROWSER"]


def test_every_setting_with_rules_is_checked(repo_root):
    """Each TOOLKIT_* name in the code is checked by the commands that read it, except the
    two that take any value."""
    named = set()
    for path in (repo_root / "toolkit").rglob("*.py"):
        named |= set(re.findall(r"\bTOOLKIT_[A-Z_]+[A-Z]\b", path.read_text(encoding="utf-8")))
    checked = {variable for variables in COMMAND_SETTINGS.values() for variable in variables}
    assert named - checked == {"TOOLKIT_INSTANCE_DIR", "TOOLKIT_SECRET_KEY"}
    assert checked <= named


@pytest.mark.parametrize("command, module", [("scan", "toolkit.cli.controllers.scan"),
                                             ("ids", "toolkit.cli.controllers.ids"),
                                             ("web", "toolkit.web.controllers.command")])
def test_a_bad_setting_stops_the_command_with_exit_code_2(command, module, monkeypatch, capsys):
    started = []
    monkeypatch.setattr(f"{module}.run",
                        lambda argv, prog: started.append(argv))
    monkeypatch.setenv("TOOLKIT_LOG_LEVEL", "verbose")
    assert start.main([command]) == 2
    assert started == [], "the command must not start"
    assert capsys.readouterr().out == ("TOOLKIT_LOG_LEVEL must be one of DEBUG, INFO, WARNING, "
                                       "ERROR, not 'verbose'\n")


def test_help_works_with_a_bad_setting(monkeypatch, capsys):
    monkeypatch.setenv("TOOLKIT_MAX_WORKSPACES", "0")
    monkeypatch.setattr("toolkit.web.controllers.command.run_server",
                        lambda config=None: pytest.fail("--help must not start the server"))
    with pytest.raises(SystemExit) as stop:
        start.main(["web", "--help"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith("usage: python -m toolkit web ")


def test_another_tools_typo_does_not_stop_a_command(monkeypatch):
    started = []
    monkeypatch.setattr("toolkit.cli.controllers.ids.run", lambda argv, prog: started.append(argv))
    monkeypatch.setenv("TOOLKIT_MAX_UPLOAD_MB", "abc")
    assert start.main(["ids", "--list"]) == 0
    assert started == [["--list"]]


def test_the_real_command_says_it_without_a_traceback(repo_root, monkeypatch):
    """The crash found in step 1.5e: a bad log level was a traceback and exit code 1."""
    monkeypatch.setenv("TOOLKIT_LOG_LEVEL", "verbose")
    monkeypatch.setenv("TOOLKIT_MAX_WORKSPACES", "0")
    done = subprocess.run([sys.executable, "-m", "toolkit", "web"], cwd=repo_root,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 2
    assert done.stdout.splitlines() == [
        "TOOLKIT_LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, not 'verbose'",
        "TOOLKIT_MAX_WORKSPACES must be a whole number >= 1, not '0'"]
    assert "Traceback" not in done.stderr
