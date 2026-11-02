"""HC4: every test file has one level (tests/levels.py), and every test carries it."""

from pathlib import Path

from levels import E2E, INTEGRATION, LEVELS, UNIT

TESTS = Path(__file__).resolve().parent
NAMES = ("unit", "integration", "e2e")


def test_every_test_file_has_one_level():
    files = {path.relative_to(TESTS).as_posix() for path in TESTS.rglob("test_*.py")}
    assert files - set(LEVELS) == set(), "add each new test file to tests/levels.py"
    assert set(LEVELS) - files == set(), "tests/levels.py names a file that doesn't exist"
    assert len(UNIT) + len(INTEGRATION) + len(E2E) == len(LEVELS), "a file in two levels"


def test_the_levels_are_registered_markers(pytestconfig):
    registered = {line.split(":")[0] for line in pytestconfig.getini("markers")}
    assert set(NAMES) <= registered


def test_every_test_carries_its_files_level(request):
    marks = {mark.name for mark in request.node.iter_markers()}
    assert marks & set(NAMES) == {"unit"}
