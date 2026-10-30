"""v5.1 step 1.5e: docs/reference.md names every flag and every setting the code has, and
nothing the code doesn't have (plan D20)."""

import re

import pytest

from toolkit.cli.controllers import ids, scan
from toolkit.web.controllers import command as web
from toolkit.core.data.keys import DEFAULT_AI_MODEL, KEY_VARIABLES
from toolkit.core.data.settings import COMMAND_SETTINGS, SHARED_SETTINGS, ScanConfig, WebConfig

PARSERS = {"scan": scan.build_parser, "ids": ids.build_parser, "web": web.build_parser}
DEFAULTS = {  # the settings table's default column, from the code
    "TOOLKIT_SCAN_MAX_HOSTS": ScanConfig.max_hosts,
    "TOOLKIT_SCAN_MAX_PROBES": ScanConfig.max_probes,
    "TOOLKIT_SCAN_MAX_MIN": ScanConfig.max_minutes,
    "TOOLKIT_PORT": WebConfig.port,
    "TOOLKIT_MAX_UPLOAD_MB": WebConfig.max_upload_mb,
    "TOOLKIT_MAX_PREDICT_MB": WebConfig.max_predict_mb,
    "TOOLKIT_MAX_WORKSPACES": WebConfig.max_workspaces,
    "TOOLKIT_MAX_DATA_MB": WebConfig.max_data_mb,
    "TOOLKIT_WORKSPACE_IDLE_MIN": WebConfig.workspace_idle_minutes,
    "TOOLKIT_JOB_MAX_MIN": WebConfig.job_max_minutes,
}


@pytest.fixture
def reference(repo_root):
    return (repo_root / "docs" / "reference.md").read_text(encoding="utf-8")


def _section(text, heading):
    """From ``heading`` (a whole heading line's start) to the next heading of its level or
    above."""
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = re.compile(rf"^#{{1,{level}}} ", re.MULTILINE).search(text, start + len(heading))
    return text[start:end.start()] if end else text[start:]


def _rows(section):
    """The cells of each table row, the header and the --- line left out."""
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines() if line.startswith("| ")]
    return rows[1:] if rows else rows


def _named_in_code(repo_root):
    names = set()
    for path in (repo_root / "toolkit").rglob("*.py"):
        names |= set(re.findall(r"\bTOOLKIT_[A-Z_]+[A-Z]\b", path.read_text(encoding="utf-8")))
    return names


@pytest.mark.parametrize("command", PARSERS)
def test_every_flag_of_each_command_is_documented(reference, command):
    parser = PARSERS[command](f"python -m toolkit {command}")
    flags = {option for action in parser._actions for option in action.option_strings
             if option not in ("-h", "--help")}
    heading = re.search(rf"^## \d+\. `{command}`: ", reference, re.MULTILINE).group()
    section = _section(reference, heading)
    documented = {flag for row in _rows(section)
                  for flag in re.findall(r"`(--?[a-z][a-z-]*)", row[0])}
    assert documented == flags
    for action in parser._actions:
        if not action.option_strings:  # positional: scan's TARGET
            assert f"| `{action.metavar}` |" in section


def test_every_setting_is_in_the_table_and_no_other(reference, repo_root):
    rows = _rows(_section(reference, "### 7.1 `TOOLKIT_*` settings"))
    documented = [re.fullmatch(r"`(TOOLKIT_[A-Z_]+)`", row[0]).group(1) for row in rows]
    assert len(documented) == len(set(documented))
    assert set(documented) == _named_in_code(repo_root)


def test_the_table_says_which_command_reads_each_setting(reference):
    for row in _rows(_section(reference, "### 7.1 `TOOLKIT_*` settings")):
        variable, read_by = row[0].strip("`"), row[1]
        if read_by == "all":
            assert variable in SHARED_SETTINGS or variable == "TOOLKIT_INSTANCE_DIR"
        else:
            command = read_by.strip("`")
            assert variable in COMMAND_SETTINGS[command] or variable == "TOOLKIT_SECRET_KEY"
            assert variable not in SHARED_SETTINGS


def test_the_defaults_are_the_codes(reference):
    rows = {row[0].strip("`"): row for row in
            _rows(_section(reference, "### 7.1 `TOOLKIT_*` settings"))}
    for variable, default in DEFAULTS.items():
        assert rows[variable][3] == f"`{default}`", variable
    assert rows["TOOLKIT_LOG_LEVEL"][3] == "`INFO`"


def test_every_key_variable_is_documented(reference):
    rows = {row[0].strip("`"): row for row in
            _rows(_section(reference, "### 7.2 API keys and the AI model"))}
    assert set(rows) == {*KEY_VARIABLES.values(), "OPENROUTER_MODEL"}
    assert rows["OPENROUTER_MODEL"][1] == f"`{DEFAULT_AI_MODEL}`"


def _anchor(heading):
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to dashes."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_links_and_anchors_exist(reference, repo_root):
    links = re.findall(r"\]\(([^)#]+)(?:#([^)]+))?\)", reference)
    assert links
    for target, anchor in links:
        path = repo_root / "docs" / target
        assert path.is_file(), target
        if anchor:
            headings = re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
            assert anchor in {_anchor(heading) for heading in headings}, (target, anchor)
