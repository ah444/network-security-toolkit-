"""v5.1 step 1.5d: a bundle saved before the layered layout still loads
(toolkit.services.ids.legacy_names).

Its pickle refers to classes and functions by their old module names
(``toolkit.ids_core.pipeline``, ``toolkit.ids_core.registry``); the fixture was made by the
old code (``tests/fixtures/legacy/make_legacy_bundle.py``) and is never rebuilt by newer code.
"""

import importlib
import importlib.util
import json
import subprocess
import sys
import zlib

import pandas as pd
import pytest

from toolkit.data.ids import bundle_files
from toolkit.services.ids import legacy_names
from toolkit.services.ids.bundle import _layout, unpack
from toolkit.services.ids.pipeline import FittedModel

KEY = bytes(range(32))  # the key the fixture was signed with
OLD_NAMES = (b"toolkit.ids_core.pipeline", b"toolkit.ids_core.registry")


@pytest.fixture
def legacy(fixtures_dir):
    folder = fixtures_dir / "legacy"
    expected = json.loads((folder / "old_names.expected.json").read_text(encoding="utf-8"))
    return folder / "old_names.bundle", expected


def test_fixture_refers_to_the_old_module_names(legacy):
    path, _ = legacy
    pickled = zlib.decompress(_layout(path.read_bytes())[1])  # read only, nothing unpickled
    assert all(name in pickled for name in OLD_NAMES)


def test_old_bundle_loads_and_predicts(legacy):
    path, expected = legacy
    loaded = unpack(path.read_bytes(), key=KEY)
    assert loaded.run_id == expected["run_id"]
    rows = pd.DataFrame(expected["rows"])
    for name, labels in expected["predictions"].items():
        assert [str(label) for label in loaded.models[name].predict(rows)] == labels, name


def test_old_bundle_through_the_ids_command(legacy, tmp_path):
    """In a new process, so no module is already imported under its old name."""
    path, expected = legacy
    key = bundle_files.key_path()
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(KEY.hex(), encoding="ascii")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(expected["rows"]), encoding="utf-8")
    for name, labels in expected["predictions"].items():
        done = subprocess.run([sys.executable, "-m", "toolkit", "ids", "--bundle", str(path),
                               "--model", name, "--json", str(rows)],
                              capture_output=True, text=True, encoding="utf-8", check=False)
        assert done.returncode == 0, done.stdout + done.stderr
        assert [answer["prediction"] for answer in json.loads(done.stdout)] == labels, name


def test_old_names_exist_only_while_a_bundle_loads(legacy):
    path, _ = legacy
    loaded = unpack(path.read_bytes(), key=KEY)
    assert all(isinstance(m, FittedModel) for m in loaded.models.values())  # today's class
    assert not [name for name in sys.modules if name.startswith("toolkit.ids_core")]
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("toolkit.ids_core.pipeline")


def test_every_old_module_name_has_a_module(repo_root):
    """The 14 modules of toolkit/ids_core (v5.1 step 1.4) are all still in services/ids."""
    assert len(legacy_names.MODULES) == 14
    for old, new in legacy_names.OLD_TO_NEW.items():
        assert importlib.util.find_spec(new) is not None, (old, new)
