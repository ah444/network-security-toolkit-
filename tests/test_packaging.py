"""pyproject packaging (v5.1 step 1.5b, plan D18): ``pip install .`` on Windows, Linux and macOS.

The pins in pyproject.toml are the ones in requirements.txt / requirements-dev.txt (the
files ``pip install -r`` uses), so the two can't drift apart.
"""

import ast
import re
import sys
import tomllib

import pytest

# import name -> requirement name, where they differ
REQUIREMENT_NAMES = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn"}
BOOST = {"xgboost", "lightgbm"}


@pytest.fixture(scope="module")
def pyproject(request):
    return tomllib.loads((request.config.rootpath / "pyproject.toml").read_text(encoding="utf-8"))


def _pins(path):
    """``name==version`` lines of a requirements file (comments and ``-r`` lines left out)."""
    pins = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            pins.add(line.lower())
    return pins


def _name(requirement):
    return re.split(r"[\[=<>!~ ]", requirement, maxsplit=1)[0].lower()


def test_dependencies_are_the_requirements_pins(repo_root, pyproject):
    project = pyproject["project"]
    runtime = {d.lower() for d in project["dependencies"]}
    boost = {d.lower() for d in project["optional-dependencies"]["boost"]}
    assert all("==" in d for d in runtime | boost), "pinned exactly, as requirements.txt"
    assert {_name(d) for d in boost} == BOOST
    assert not {_name(d) for d in runtime} & BOOST, "the boosted models are the [boost] extra"
    assert runtime | boost == _pins(repo_root / "requirements.txt")


def test_dev_extra_is_the_dev_requirements(repo_root, pyproject):
    dev = {d.lower() for d in pyproject["project"]["optional-dependencies"]["dev"]}
    assert dev == _pins(repo_root / "requirements-dev.txt")


def test_every_third_party_import_is_declared(repo_root, pyproject):
    """What toolkit imports itself is declared, not only reached through another
    package's dependencies (joblib, threadpoolctl, werkzeug: step 1.5b)."""
    imported = set()
    for path in (repo_root / "toolkit").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    third_party = {m for m in imported if m not in sys.stdlib_module_names and m != "toolkit"}
    project = pyproject["project"]
    declared = {_name(d) for d in project["dependencies"]
                + project["optional-dependencies"]["boost"]}
    missing = {REQUIREMENT_NAMES.get(m, m) for m in third_party} - declared
    assert not missing, f"imported but not declared: {sorted(missing)}"


def test_build_backend_package_and_command(pyproject):
    assert pyproject["build-system"] == {"requires": ["setuptools>=78.1.1"],
                                         "build-backend": "setuptools.build_meta"}
    project = pyproject["project"]
    assert project["requires-python"] == ">=3.11"
    assert project["scripts"] == {"toolkit": "toolkit.controllers.cli.main:main"}
    # the project folder also holds data/, docs/, scripts/, tests/: only toolkit is the package
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == ["toolkit*"]


def test_licence_is_mit_in_the_authors_name(repo_root, pyproject):
    """The code's licence (step 1.5g): the LICENSE file, and the wheel's metadata names it."""
    project = pyproject["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Ahmed Mohamed Abdalrazek"}]
    text = (repo_root / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("The MIT License (MIT)\n\nCopyright (c) 2025-2026 Ahmed Mohamed "
                           "Abdalrazek\n\nPermission is hereby granted, free of charge,")
    assert 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND' in text


def test_package_data_covers_every_file_the_web_trainer_serves(repo_root, pyproject):
    """The templates and static files (CSS, JS modules) are in the wheel."""
    patterns = pyproject["tool"]["setuptools"]["package-data"]
    covered = set()
    for package, globs in patterns.items():
        folder = repo_root.joinpath(*package.split("."))
        for pattern in globs:
            covered |= {p for p in folder.glob(pattern) if p.is_file()}
    files = {p for p in (repo_root / "toolkit").rglob("*")
             if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts}
    assert files, "the web trainer has templates and static files"
    assert files <= covered, f"not packaged: {sorted(str(p) for p in files - covered)}"


def test_optional_libraries_may_be_missing_for_mypy(pyproject):
    """Without the [boost] extra, mypy must not fail on the boosted models' imports."""
    ignored = set()
    for override in pyproject["tool"]["mypy"]["overrides"]:
        if override.get("ignore_missing_imports"):
            ignored |= set(override["module"])
    assert BOOST <= ignored
