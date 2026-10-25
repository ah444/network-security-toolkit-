"""v5.1 step 1.5f: docs/user_guide.md shows only pictures that exist, every picture in
docs/user-guide/ is used, its commands use only flags the commands have, and its links
resolve."""

import re

import pytest

from toolkit.cli.controllers import ids, scan
from toolkit.web.controllers import command as web
from toolkit.cli.controllers.main import COMMANDS as CLI_COMMANDS
from toolkit.cli.controllers.main import OTHER_COMMANDS

COMMANDS = {*CLI_COMMANDS, *OTHER_COMMANDS}  # what toolkit.start runs

PARSERS = {"scan": scan.build_parser, "ids": ids.build_parser, "web": web.build_parser}
MAX_PICTURE_BYTES = 200_000


@pytest.fixture
def guide(repo_root):
    return (repo_root / "docs" / "user_guide.md").read_text(encoding="utf-8")


def _anchor(heading):
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to dashes."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_every_picture_shown_exists_and_every_picture_is_shown(guide, repo_root):
    shown = re.findall(r"!\[[^\]]+\]\((user-guide/[^)]+)\)", guide)
    assert len(shown) == len(set(shown)) >= 10
    folder = repo_root / "docs" / "user-guide"
    assert {f"user-guide/{path.name}" for path in folder.iterdir()} == set(shown)


def test_pictures_are_small_jpegs(repo_root):
    for path in (repo_root / "docs" / "user-guide").iterdir():
        data = path.read_bytes()
        assert path.suffix == ".jpg" and data[:3] == b"\xff\xd8\xff", path.name
        assert len(data) <= MAX_PICTURE_BYTES, path.name


def test_commands_use_only_flags_the_commands_have(guide):
    commands = re.findall(r"python -m toolkit((?: [^\s`|]+)*)", guide)
    assert len(commands) >= 15
    seen = set()
    for words in commands:
        words = words.split()
        if not words or words[0].startswith("-"):  # the menu, or the toolkit's own --help
            assert all(word in ("--help", "-h") for word in words), words
            continue
        command, rest = words[0], words[1:]
        assert command in COMMANDS, command
        seen.add(command)
        options = {option for action in PARSERS[command](command)._actions
                   for option in action.option_strings}
        for word in rest:
            if word.startswith("-"):
                assert word in options, f"{command} has no {word}"
    assert seen == set(COMMANDS)


def test_links_and_anchors_resolve(guide, repo_root):
    docs = repo_root / "docs"
    links = re.findall(r"(?<!!)\[[^\]]+\]\(([^)#]*)(?:#([^)]+))?\)", guide)
    assert len(links) >= 10
    for target, anchor in links:
        path = (docs / target).resolve() if target else docs / "user_guide.md"
        assert path.is_file(), target
        if anchor:
            headings = re.findall(r"^#+ (.+)$", path.read_text(encoding="utf-8"),
                                  re.MULTILINE)
            assert anchor in {_anchor(heading) for heading in headings}, (target, anchor)


def test_readme_shows_pictures_of_the_guide(repo_root):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## 📸 Screenshots"):readme.index("## 🛠️ Troubleshooting")]
    assert "docs/user_guide.md" in section
    pictures = re.findall(r"!\[[^\]]+\]\((docs/user-guide/[^)]+)\)", section)
    assert pictures and all((repo_root / picture).is_file() for picture in pictures)
