"""v5.2 step 2.3a: the data functions give exactly the results the code before the step 2.3
refactor gave (``tests/fixtures/golden/outputs.json``, see ``tests/golden_outputs.py``)."""

import json
from pathlib import Path

import pytest

import golden_outputs

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def golden():
    return json.loads((FIXTURES_DIR / "golden" / "outputs.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def now():
    """Computed once for the module (the leakage checks fit small trees)."""
    return json.loads(json.dumps(golden_outputs.outputs(FIXTURES_DIR), sort_keys=True))


@pytest.mark.parametrize("name", golden_outputs.FIXTURES)
def test_each_file_as_before(golden, now, name):
    for part, value in golden["files"][name].items():
        assert now["files"][name][part] == value, (name, part)


def test_the_official_split_as_before(golden, now):
    assert now["official"] == golden["official"]


def test_the_balancing_rules_as_before(golden, now):
    assert now["rule_table"] == golden["rule_table"]


def test_the_rule_table_reaches_every_reachable_answer(golden):
    """15 of the 16 reason codes; DEFAULT is unreachable (every severity has its own case)."""
    codes = {row[-1][1] for row in golden["rule_table"]}
    assert len(codes) == 15 and "DEFAULT" not in codes
