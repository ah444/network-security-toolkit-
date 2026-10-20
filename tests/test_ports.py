"""The 3 ports of plan §2 and their adapters (v5.1 step 1.1c)."""

import numpy as np
import pandas as pd
import pytest

from toolkit import bundle_files, ports
from toolkit.ids_core import bundle
from toolkit.ids_core.bundle import BundleError
from toolkit.ids_core.runs import DatasetRef, TrainSettings, train

# --- ModelStore ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(0)
    label = rng.choice(["BENIGN", "DoS"], size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "b": rng.normal(size=200), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified", models=("decision_tree",),
                             workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=200, columns=3))


def test_bundle_store_is_a_model_store():
    assert isinstance(bundle_files.BundleStore(), ports.ModelStore)


def test_bundle_store_round_trip(tmp_path, monkeypatch, run):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    store: ports.ModelStore = bundle_files.BundleStore()

    saved = store.save(run)
    assert saved.path == tmp_path / "bundles" / f"{run.run_id}.bundle"
    assert [info.path for info in store.list_bundles()] == [saved.path]
    assert store.load(saved.path).sha256 == saved.sha256
    assert store.open(saved.path.read_bytes()).sha256 == saved.sha256
    with pytest.raises(FileExistsError):
        store.save(run)  # never overwritten


def test_bundle_store_follows_its_folder(tmp_path, monkeypatch, run):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path / "instance"))
    store = bundle_files.BundleStore(tmp_path / "elsewhere", key_file=tmp_path / "other.key")
    saved = store.save(run)
    assert saved.path.parent == tmp_path / "elsewhere" and (tmp_path / "other.key").is_file()
    assert store.list_bundles()[0].path == saved.path
    assert bundle_files.BundleStore().list_bundles() == []  # the default folder is untouched


def test_bundle_store_refuses_tampered_before_loading(tmp_path, monkeypatch, run):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    store = bundle_files.BundleStore()
    data = bytearray(store.save(run).path.read_bytes())
    data[-1] ^= 0xFF

    def refuse(*args, **kwargs):
        raise AssertionError("joblib.load ran on a tampered bundle")

    monkeypatch.setattr(bundle.joblib, "load", refuse)
    with pytest.raises(BundleError) as excinfo:
        store.open(bytes(data))
    assert excinfo.value.code == "BAD_SIGNATURE"
