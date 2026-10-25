"""v5.1 step 1.5d and v5.2 step 2.3: bundles saved with older module names still load
(toolkit.core.services.ids.legacy_names).

Two generations of fixtures, each made by the code of its time and never rebuilt by newer code:
``old_names.bundle`` refers to ``toolkit.ids_core.*`` (before v5.1 step 1.5d,
``tests/fixtures/legacy/make_legacy_bundle.py``) and ``v51_names.bundle`` to
``toolkit.services.ids.*`` (v5.1 step 1.5d to v5.2 step 2.3a, ``make_v51_bundle.py``).
"""

import importlib
import importlib.util
import json
import subprocess
import sys
import zlib

import pandas as pd
import pytest

from toolkit.core.data.ids import bundle_files
from toolkit.core.services.ids import legacy_names
from toolkit.core.services.ids.bundle import _layout, unpack
from toolkit.core.services.ids.pipeline import FittedModel

KEY = bytes(range(32))  # the key the fixtures were signed with
GENERATIONS = {
    "old_names": (b"toolkit.ids_core.pipeline", b"toolkit.ids_core.registry"),
    "v51_names": (b"toolkit.services.ids.pipeline", b"toolkit.services.ids.registry"),
}


@pytest.fixture(params=sorted(GENERATIONS))
def legacy(request, fixtures_dir):
    folder = fixtures_dir / "legacy"
    expected = json.loads((folder / f"{request.param}.expected.json").read_text(encoding="utf-8"))
    return folder / f"{request.param}.bundle", expected, GENERATIONS[request.param]


def test_fixture_refers_to_its_module_names(legacy):
    path, _, names = legacy
    pickled = zlib.decompress(_layout(path.read_bytes())[1])  # read only, nothing unpickled
    assert all(name in pickled for name in names)


def test_old_bundle_loads_and_predicts(legacy):
    path, expected, _ = legacy
    loaded = unpack(path.read_bytes(), key=KEY)
    assert loaded.run_id == expected["run_id"]
    rows = pd.DataFrame(expected["rows"])
    for name, labels in expected["predictions"].items():
        assert [str(label) for label in loaded.models[name].predict(rows)] == labels, name


def test_old_bundle_through_the_ids_command(legacy, tmp_path):
    """In a new process, so no module is already imported under its old name."""
    path, expected, _ = legacy
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


@pytest.mark.parametrize("old", ["toolkit.ids_core", "toolkit.services.ids"])
def test_old_names_exist_only_while_a_bundle_loads(legacy, old):
    path, _, _ = legacy
    loaded = unpack(path.read_bytes(), key=KEY)
    assert all(isinstance(m, FittedModel) for m in loaded.models.values())  # today's class
    assert not [name for name in sys.modules if name == old or name.startswith(old + ".")]
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{old}.pipeline")


def test_every_old_module_name_has_a_module(repo_root):
    """The 14 modules of toolkit/ids_core (v5.1 step 1.4) and the 15 of toolkit/services/ids
    (v5.1 step 1.5d) are all in toolkit/core/services/ids."""
    assert len(legacy_names.MODULES) == 14 and len(legacy_names.V51_MODULES) == 15
    assert len(legacy_names.OLD_TO_NEW) == 1 + 14 + 1 + 15
    for old, new in legacy_names.OLD_TO_NEW.items():
        assert importlib.util.find_spec(new) is not None, (old, new)
