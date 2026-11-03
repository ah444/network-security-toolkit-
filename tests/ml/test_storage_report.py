"""v5.3 step HR4b (RD4: IA9): the storage report. It counts every file of the instance by
kind, with their sizes, the total and the disk's free space, and names the leftovers: an
importance result without a record or for a model not in it, an unregistered bundle, an
incomplete bundle, a damaged file, a temporary file. It only reads: it deletes nothing, and
it never unpickles a bundle (a bundle's state comes from its header)."""

import json

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids import importance_files
from toolkit.core.data.ids.bundle_files import bundles_dir
from toolkit.core.data.ids.run_files import runs_dir
from toolkit.core.data.settings import instance_dir
from toolkit.core.services import storage
from toolkit.core.services.ids import bundle
from toolkit.core.services.ids.exports import save_bundle, write_run
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train

MODELS = ("decision_tree", "logistic_regression")
NO_RECORD_ID = "20200101T000000Z-0000000c"
DAMAGED_ID = "20200101T000000Z-0000000d"


def _run(seed):
    rng = np.random.default_rng(seed)
    label = rng.choice(["BENIGN", "DoS"], size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "noise": rng.normal(size=200), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified", models=MODELS, workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=200, columns=3))


@pytest.fixture(scope="module")
def runs():
    first = _run(0)
    return first, _run(1)


@pytest.fixture
def instance(runs):
    """One file of every kind: what each kind should hold (its file names)."""
    saved_run, unregistered = runs
    saved = save_bundle(saved_run)
    write_run(saved.run)
    lone = save_bundle(unregistered)  # a bundle whose record isn't there
    folder = runs_dir()
    importance_files.write_importance(folder, saved_run.run_id, "decision_tree", {})
    importance_files.write_importance(folder, saved_run.run_id, "knn", {})  # not in the record
    importance_files.write_importance(folder, NO_RECORD_ID, "decision_tree", {})
    (folder / f"{DAMAGED_ID}.json").write_text("{not json", encoding="utf-8")
    (folder / f"{saved_run.run_id}.json.0123abcd.tmp").write_bytes(b"x" * 10)
    whole = lone.path.read_bytes()
    incomplete = bundles_dir() / f"{NO_RECORD_ID}.bundle"
    incomplete.write_bytes(whole[: len(whole) // 2])
    (bundles_dir() / "notes.bundle").write_bytes(b"not a bundle")
    for folder_name, name in (("scans", "scan-20261025T200015Z.txt"), ("uploads", "u.csv"),
                              ("logs", "web.log")):
        (instance_dir() / folder_name).mkdir(exist_ok=True)
        (instance_dir() / folder_name / name).write_bytes(b"y" * 100)
    (instance_dir() / "jobs.journal").write_bytes(b"{}\n")
    sid = saved_run.run_id
    return {
        "records": [f"{sid}.json"],
        "importance": [f"{sid}.importance.decision_tree.json"],
        "orphan_importance": sorted([f"{sid}.importance.knn.json",
                                     f"{NO_RECORD_ID}.importance.decision_tree.json"]),
        "bundles": [f"{sid}.bundle"],
        "unregistered_bundles": [f"{unregistered.run_id}.bundle"],
        "incomplete_bundles": [f"{NO_RECORD_ID}.bundle"],
        "damaged": sorted([f"{DAMAGED_ID}.json", "notes.bundle"]),
        "temporary": [f"{sid}.json.0123abcd.tmp"],
        "scans": ["scan-20261025T200015Z.txt"],
        "uploads": ["u.csv"],
        "logs": ["web.log"],
        "other": sorted(["jobs.journal", "signing.key"]),
    }


def _every_file():
    return {path: path.stat().st_size for path in instance_dir().rglob("*") if path.is_file()}


def test_every_file_is_counted_once_by_kind(instance):
    report = storage.storage_report()
    kinds = {item["kind"]: item for item in report["kinds"]}
    assert list(kinds) == list(instance), "every kind, in this order"
    files = _every_file()
    by_name = {path.name: size for path, size in files.items()}
    for kind, names in instance.items():
        assert kinds[kind]["count"] == len(names), kind
        assert kinds[kind]["bytes"] == sum(by_name[name] for name in names), kind
    assert report["total_bytes"] == sum(files.values())
    assert sum(item["count"] for item in report["kinds"]) == len(files)
    assert report["free_bytes"] > 0


def test_the_leftovers_are_named_and_the_rest_counted(instance):
    kinds = {item["kind"]: item for item in storage.storage_report()["kinds"]}
    for kind in storage.LEFTOVERS:
        assert sorted(kinds[kind]["files"]) == instance[kind], kind
        assert kinds[kind]["leftover"] is True
    for kind in set(instance) - set(storage.LEFTOVERS):
        assert kinds[kind]["files"] == [] and kinds[kind]["leftover"] is False, kind
    assert all(item["title"] for item in kinds.values())


def test_it_deletes_nothing_and_unpickles_nothing(instance, monkeypatch):
    before = _every_file()

    def refuse(*_args, **_kwargs):
        raise AssertionError("a bundle was opened")
    monkeypatch.setattr(bundle, "unpack", refuse)
    monkeypatch.setattr(bundle.joblib, "load", refuse)
    storage.storage_report()
    assert _every_file() == before


def test_an_empty_instance():
    report = storage.storage_report()
    assert report["total_bytes"] == 0 and all(item["count"] == 0 for item in report["kinds"])


def test_a_record_that_is_not_a_run_is_damaged(runs):
    write_run(runs[0])
    path = runs_dir() / f"{runs[0].run_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["schema_version"] = 999  # newer than this toolkit: not shown, so not owned
    path.unlink()
    path.write_text(json.dumps(record), encoding="utf-8")
    kinds = {item["kind"]: item for item in storage.storage_report()["kinds"]}
    assert kinds["damaged"]["files"] == [path.name] and kinds["records"]["count"] == 0
