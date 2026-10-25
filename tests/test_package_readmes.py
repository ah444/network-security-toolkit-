"""v5.2 step 2.3e: the README of each folder of the package stays true to it. It names every
module and folder of its interface down to the layer folders (a new one fails this test
until the README says what it is), every file it names exists, and its links resolve,
anchors included."""

import re
from pathlib import Path

import pytest

READMES = ("toolkit/README.md", "toolkit/core/README.md", "toolkit/cli/README.md",
           "toolkit/web/README.md")


def _children(folder: Path):
    """The names a README must mention: the folder's modules and subfolders, and those of
    its layer folders (``ui/``, ``controllers/``, ...), without caches and ``__init__.py``."""
    names = set()
    for child in folder.iterdir():
        if child.name in ("__pycache__", "README.md", "__init__.py"):
            continue
        names.add(child.name + ("/" if child.is_dir() else ""))
        if child.is_dir() and folder.name in ("core", "cli", "web"):
            names |= {grand.name + ("/" if grand.is_dir() else "") for grand in child.iterdir()
                      if grand.name not in ("__pycache__", "__init__.py")}
    return names


def _slug(heading: str) -> str:
    """The anchor a Markdown viewer makes of a heading (GitHub's rule, enough for these)."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


@pytest.mark.parametrize("readme", READMES)
def test_each_folder_has_its_readme(repo_root, readme):
    assert (repo_root / readme).is_file(), readme


@pytest.mark.parametrize("readme", READMES)
def test_the_readme_names_everything_in_its_folder(repo_root, readme):
    path = repo_root / readme
    text = path.read_text(encoding="utf-8")
    words = set(re.findall(r"[\w.]+/?", text))
    missing = sorted(name for name in _children(path.parent)
                     if name not in words and name.rstrip("/") not in words)
    assert not missing, f"{readme} doesn't say what these are: {missing}"


@pytest.mark.parametrize("readme", READMES)
def test_the_readme_links_resolve(repo_root, readme):
    path = repo_root / readme
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        file, _, anchor = target.partition("#")
        linked = (path.parent / file).resolve()
        assert linked.exists(), (readme, target)
        if anchor:
            headings = re.findall(r"^#+ (.+)$", linked.read_text(encoding="utf-8"), flags=re.M)
            assert anchor in {_slug(h) for h in headings}, (readme, target)


@pytest.mark.parametrize("readme", READMES)
def test_the_tests_and_files_it_names_exist(repo_root, readme):
    text = (repo_root / readme).read_text(encoding="utf-8")
    for named in re.findall(r"`((?:tests|toolkit|docs|scripts)/[\w./-]+\.\w+)`", text):
        assert (repo_root / named).is_file(), (readme, named)
