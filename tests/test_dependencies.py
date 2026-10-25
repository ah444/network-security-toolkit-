"""Runtime dependencies are only what the code uses (5.0-rc1: dask removed, plan D16, D18)."""

import re

import pytest

# requirements.txt name -> import name
IMPORT_NAMES = {"scikit-learn": "sklearn", "imbalanced-learn": "imblearn"}
# dask[dataframe] and the packages only it needed (nothing else installed requires them)
REMOVED = {"dask", "pyarrow", "fsspec", "partd", "locket", "toolz", "pyyaml"}
CODE_DIRS = ("toolkit", "scripts")


def _requirements(path):
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(re.split(r"[\[=<>!~ ]", line, maxsplit=1)[0].lower())
    return names


@pytest.fixture(scope="module")
def code(request):
    root = request.config.rootpath
    files = [root / "main.py"]
    for folder in CODE_DIRS:
        files += sorted((root / folder).rglob("*.py"))
    return "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)


def test_every_runtime_requirement_is_imported(repo_root, code):
    for name in _requirements(repo_root / "requirements.txt"):
        module = IMPORT_NAMES.get(name, name.replace("-", "_"))
        assert re.search(rf"^\s*(import|from)\s+{module}\b", code, re.M), \
            f"{name} is in requirements.txt but nothing imports {module}"


@pytest.mark.parametrize("filename", ["requirements.txt", "requirements.lock"])
def test_dask_and_its_dependencies_removed(repo_root, filename):
    listed = set(_requirements(repo_root / filename))
    assert not listed & REMOVED, f"{filename} still lists {sorted(listed & REMOVED)}"


def test_no_code_imports_dask(code):
    assert not re.search(r"^\s*(import|from)\s+(dask|pyarrow)\b", code, re.M)
