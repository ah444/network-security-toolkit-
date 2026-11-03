"""v5.3 step HR4b (RD4): a bundle kept when its run was deleted (an unregistered bundle) is
registered again. A bundle carries the copy of its run's record that it was signed with, and
records never change, so once the whole file's signature is checked (read in pieces: a
bundle can be gigabytes, and nothing in it is unpickled) that copy, with the bundle's
SHA-256, is published as the record: the same record as before, byte for byte. It adds a
file and never replaces one."""

import hashlib
import hmac
import json
import threading

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids import bundle_files, importance_files
from toolkit.core.data.ids.bundle_files import BundleError, bundles_dir, signing_key
from toolkit.core.data.ids.run_files import runs_dir
from toolkit.core.services.ids import bundle, registration, run_removal
from toolkit.core.services.ids.bundle import HEADER_BYTES, MAGIC
from toolkit.core.services.ids.comparison import NotARun, saved_runs
from toolkit.core.services.ids.exports import save_bundle, write_run
from toolkit.core.services.ids.registration import NotRegistered
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train


def _run(seed):
    rng = np.random.default_rng(seed)
    label = rng.choice(["BENIGN", "DoS"], size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "noise": rng.normal(size=200), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified",
                             models=("decision_tree", "logistic_regression"), workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=200, columns=3))


@pytest.fixture(scope="module")
def trained():
    return _run(0)


@pytest.fixture
def kept(trained):
    """A run exported, then deleted with its bundle kept: (run id, the record's bytes)."""
    saved = save_bundle(trained)
    write_run(saved.run)
    record = (runs_dir() / f"{trained.run_id}.json").read_bytes()
    run_removal.remove_run(trained.run_id, with_bundle=False)
    return trained.run_id, record


def _bundle(run_id):
    return bundles_dir() / f"{run_id}.bundle"


def _resign(run_id, change):
    """The bundle of ``run_id`` with its manifest changed by ``change`` and signed again with
    this install's key (a bundle this toolkit could have made)."""
    data = _bundle(run_id).read_bytes()
    length = int.from_bytes(data[len(MAGIC):HEADER_BYTES], "big")
    manifest = json.loads(data[HEADER_BYTES:HEADER_BYTES + length])
    change(manifest)
    body = json.dumps(manifest).encode("utf-8")
    unsigned = MAGIC + len(body).to_bytes(8, "big") + body + data[HEADER_BYTES + length:-32]
    path = _bundle(run_id)
    path.unlink()
    path.write_bytes(unsigned + hmac.new(signing_key(), unsigned, hashlib.sha256).digest())


def test_the_same_record_as_before_byte_for_byte(kept):
    run_id, record = kept
    path = registration.register_bundle(run_id)
    assert path == runs_dir() / f"{run_id}.json" and path.read_bytes() == record
    assert [r["run_id"] for r in saved_runs()["runs"]] == [run_id]


def test_it_is_not_changed_to_name_its_bundle(kept):
    """HR5: a record is written once, by its training; its bundle is found by its file."""
    run_id, _record = kept
    registration.register_bundle(run_id)
    stored = json.loads((runs_dir() / f"{run_id}.json").read_text(encoding="utf-8"))
    assert stored["bundle_sha256"] is None


@pytest.mark.parametrize("where", [-1, HEADER_BYTES + 5, "payload"])
def test_a_changed_byte_is_refused(kept, where):
    run_id, _record = kept
    data = bytearray(_bundle(run_id).read_bytes())
    at = len(data) - 100 if where == "payload" else where
    data[at] ^= 1
    _bundle(run_id).unlink()
    _bundle(run_id).write_bytes(bytes(data))
    with pytest.raises(BundleError) as refused:
        registration.register_bundle(run_id)
    assert refused.value.code == "BAD_SIGNATURE" or isinstance(refused.value, NotRegistered)
    assert not (runs_dir() / f"{run_id}.json").exists()


def test_it_reads_in_pieces_and_unpickles_nothing(kept, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("read whole or unpickled")
    monkeypatch.setattr(bundle_files, "read_all", refuse)
    monkeypatch.setattr(bundle, "unpack", refuse)
    monkeypatch.setattr(bundle.joblib, "load", refuse)
    monkeypatch.setattr(bundle_files, "CHUNK_BYTES", 4096)  # many pieces
    assert registration.register_bundle(kept[0]).read_bytes() == kept[1]


def test_a_saved_run_is_never_replaced(trained):
    saved = save_bundle(trained)
    write_run(saved.run)
    before = (runs_dir() / f"{trained.run_id}.json").read_bytes()
    with pytest.raises(FileExistsError):
        registration.register_bundle(trained.run_id)
    assert (runs_dir() / f"{trained.run_id}.json").read_bytes() == before


def test_no_bundle_an_incomplete_one_and_bad_ids(kept):
    with pytest.raises(FileNotFoundError):
        registration.register_bundle("20200101T000000Z-0000000c")
    run_id = kept[0]
    whole = _bundle(run_id).read_bytes()
    _bundle(run_id).unlink()
    _bundle(run_id).write_bytes(whole[:-1000])
    with pytest.raises(NotRegistered, match="incomplete"):
        registration.register_bundle(run_id)
    for bad in ("../x", "", "20261020T125523Z-7baf0b28/../../a"):
        with pytest.raises(NotARun):
            registration.register_bundle(bad)


def test_a_bundle_without_its_record_is_refused(kept):
    run_id = kept[0]
    _resign(run_id, lambda manifest: manifest.pop("run"))
    with pytest.raises(NotRegistered, match="no copy of its run's record"):
        registration.register_bundle(run_id)
    assert not (runs_dir() / f"{run_id}.json").exists()


@pytest.mark.parametrize("change", [
    lambda m: m["run"].update(run_id="20200101T000000Z-0000000c"),  # another run's record
    lambda m: m["run"].update(schema_version=999),  # newer than this toolkit
    lambda m: m.update(models=["knn"]),  # a model its record doesn't have
])
def test_a_copy_that_isnt_this_runs_record_is_refused(kept, change):
    run_id = kept[0]
    _resign(run_id, change)
    with pytest.raises(NotRegistered, match="not one this toolkit can save"):
        registration.register_bundle(run_id)
    assert not (runs_dir() / f"{run_id}.json").exists()


def test_it_waits_for_the_publishing_lock(kept):
    done = threading.Event()
    with importance_files.PUBLISHING:
        worker = threading.Thread(target=lambda: (
            registration.register_bundle(kept[0]), done.set()))
        worker.start()
        assert not done.wait(0.3), "it waited"
        assert not (runs_dir() / f"{kept[0]}.json").exists()
    worker.join(10)
    assert done.is_set() and (runs_dir() / f"{kept[0]}.json").exists()
