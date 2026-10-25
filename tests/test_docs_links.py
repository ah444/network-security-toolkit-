"""v5.1 step 1.5g: every link of README.md and CHANGELOG.md resolves (files and anchors),
the README points to every document, and the CHANGELOG's newest section is 5.1."""

import re

import pytest

PAGES = ("README.md", "CHANGELOG.md")
DOCUMENTS = ("docs/user_guide.md", "docs/reference.md", "docs/web_api.md",
             "docs/architecture.md", "docs/model_card.md", "docs/datasets.md",
             "CHANGELOG.md", "LICENSE")


def _without_code_blocks(text):
    """The text outside ``` fences: a '# comment' in a shell example is not a heading."""
    return re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)


def _anchor(heading):
    """GitHub's anchor for a heading: lower case, punctuation and emoji dropped, spaces to
    dashes (``## 📚 Documentation`` -> ``-documentation``)."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _anchors(path):
    text = _without_code_blocks(path.read_text(encoding="utf-8"))
    return {_anchor(heading) for heading in re.findall(r"^#+ (.+)$", text, re.MULTILINE)}


def _links(path):
    """(target, anchor) of every local link and picture; web addresses left out."""
    text = _without_code_blocks(path.read_text(encoding="utf-8"))
    found = re.findall(r"\]\(([^)\s#]*)(?:#([^)\s]+))?\)", text)
    return [(target, anchor) for target, anchor in found
            if not target.startswith(("http://", "https://", "mailto:"))]


@pytest.mark.parametrize("page", PAGES)
def test_links_and_anchors_resolve(repo_root, page):
    links = _links(repo_root / page)
    assert len(links) >= 10
    for target, anchor in links:
        path = (repo_root / target).resolve() if target else repo_root / page
        assert path.exists(), f"{page}: {target} doesn't exist"
        if anchor:
            assert path.is_file() and anchor in _anchors(path), f"{page}: {target}#{anchor}"


def test_readme_points_to_every_document(repo_root):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## 📚 Documentation"):]
    section = section[:section.index("\n---")]
    linked = {target for target, _ in _links(repo_root / "README.md") if target in section}
    assert set(DOCUMENTS) <= linked, sorted(set(DOCUMENTS) - linked)
    assert "(#-documentation)" in readme, "in the table of contents"
    assert "docs/V5.1_CHECKLIST.md" in readme[:readme.index("### Known limitations")]


def test_install_steps_cover_what_a_fresh_windows_install_meets(repo_root):
    """5.1 gate (found in 1.5h): Windows' default PowerShell policy refuses Activate.ps1, and a
    Python 3.11 venv starts with setuptools 65.5.0; the install steps say what to do."""
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    steps = readme[readme.index("### Steps"):readme.index("### Dependencies")]
    assert "python -m pip install --upgrade pip setuptools" in steps
    assert ".venv\\Scripts\\activate.bat" in steps.split("```")[2], "the note after the steps"
    assert "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned" in steps
    guide = (repo_root / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert "If PowerShell refuses `Activate.ps1`" in guide


def test_changelog_starts_with_5_1(repo_root):
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = re.findall(r"^## \[([^\]]+)\]", changelog, re.MULTILINE)
    assert versions[:2] == ["5.1", "5.0.1"]
    section = changelog[changelog.index("## [5.1]"):changelog.index("## [5.0.1]")]
    for heading in ("Added", "Changed", "Security", "Removed", "Upgrading from 5.0",
                    "Known limitations"):
        assert f"\n### {heading}\n" in section, heading
    assert "docs/V5.1_CHECKLIST.md" in changelog[:changelog.index("## [")]
