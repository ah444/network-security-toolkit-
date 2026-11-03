"""v5.3 step HR5b: Saved Runs follow every training, and say each run's bundle.

Every record is listed, exported or not; ``bundle`` comes from the bundle file's first bytes
(HR3d), never its models: "exported", "incomplete", "unreadable", or None (not exported).
Records exported before HR5 (their ``bundle_sha256`` set) still list."""

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids.bundle_files import bundles_dir
from toolkit.core.data.ids.run_files import runs_dir
from toolkit.core.services.ids import comparison
from toolkit.core.services.ids.exports import save_bundle, write_run
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train

IDS = {name: f"20261103T12000{i}Z-0000000{i}"
       for i, name in enumerate(("none", "exported", "incomplete", "unreadable", "old"), 1)}


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(0)
    label = rng.choice(("BENIGN", "DoS"), size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified", models=("decision_tree",),
                             balancer="none", workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="two.csv", sha256="cd" * 32, rows=200, columns=2))


@pytest.fixture
def saved(run, monkeypatch):
    """One run of each kind of bundle; unpickling fails the test."""
    for name, run_id in IDS.items():
        write_run(dataclasses.replace(run, run_id=run_id,
                                      bundle_sha256="ef" * 32 if name == "old" else None))
    exported = save_bundle(dataclasses.replace(run, run_id=IDS["exported"])).path
    data = exported.read_bytes()
    (bundles_dir() / f"{IDS['incomplete']}.bundle").write_bytes(data[:len(data) // 2])
    (bundles_dir() / f"{IDS['unreadable']}.bundle").write_bytes(b"not a bundle")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a bundle was unpickled")
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)


def test_every_record_is_listed_with_its_bundle(saved):
    listed = {run["run_id"]: run["bundle"] for run in comparison.saved_runs()["runs"]}
    assert listed == {IDS["none"]: None, IDS["exported"]: "exported",
                      IDS["incomplete"]: "incomplete", IDS["unreadable"]: "unreadable",
                      IDS["old"]: None}


def test_the_state_of_one_run(saved):
    assert comparison.bundle_state(IDS["exported"]) == "exported"
    assert comparison.bundle_state(IDS["none"]) is None
    assert not list(runs_dir().glob("*.bundle")), "only read"
