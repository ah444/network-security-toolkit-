"""v5.3 step HR4a (RD4: IA7, IA8): deleting a saved run. Its importance results go first, then
its bundle when the user chose to remove it (kept, it is an unregistered bundle on purpose),
then its record last; a file that can't be removed keeps the record, so the run is still
listed and the delete can be tried again. It runs under the one lock of the run files."""

import threading

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids import importance_files
from toolkit.core.data.ids.bundle_files import bundles_dir
from toolkit.core.data.ids.run_files import runs_dir
from toolkit.core.services.ids import run_removal, saved_importance
from toolkit.core.services.ids.comparison import NotARun, saved_runs
from toolkit.core.services.ids.exports import list_bundles, save_bundle, write_run
from toolkit.core.services.ids.importance import permutation_importances
from toolkit.core.services.ids.run_removal import BUNDLE, IMPORTANCE, RECORD
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train

MODELS = ("decision_tree", "logistic_regression")


def _run(seed):
    rng = np.random.default_rng(seed)
    label = rng.choice(["BENIGN", "DoS"], size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "noise": rng.normal(size=200), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified", models=MODELS, workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=200, columns=3))


@pytest.fixture(scope="module")
def trained():
    return _run(0)


@pytest.fixture(scope="module")
def other_trained():
    return _run(1)


def _save(run):
    """The run exported as the web trainer does: its bundle, then its record, and an
    importance result of each model."""
    saved = save_bundle(run)
    write_run(saved.run)
    for model in MODELS:
        importance_files.write_importance(runs_dir(), run.run_id, model,
                                          {"run_id": run.run_id, "model": model})
    return run


@pytest.fixture
def run(trained):
    return _save(trained)


def _paths(run):
    return [runs_dir() / f"{run.run_id}.importance.{model}.json" for model in MODELS] + [
        bundles_dir() / f"{run.run_id}.bundle", runs_dir() / f"{run.run_id}.json"]


def _removals(monkeypatch, refuse=None):
    """What the delete removes, in order; ``refuse``: the file it can't remove."""
    removed = []
    real = run_removal.remove_file

    def remove(path):
        if refuse is not None and path.name == refuse:
            raise PermissionError(13, "used by another process", str(path))
        real(path)
        removed.append(path.name)
    remove.__wrapped__ = real  # the real one, to try again with
    monkeypatch.setattr(run_removal, "remove_file", remove)
    return removed


def test_the_files_of_a_run_in_the_order_they_go(run):
    found = run_removal.run_files(run.run_id)
    assert [(f.kind, f.path) for f in found] == list(zip(
        [IMPORTANCE, IMPORTANCE, BUNDLE, RECORD], _paths(run), strict=True))
    assert [f.size for f in found] == [path.stat().st_size for path in _paths(run)]
    assert found[0].to_dict() == {"kind": IMPORTANCE, "name": _paths(run)[0].name,
                                  "size": found[0].size}, "a name, never a path"


def test_a_delete_removes_every_file_in_that_order(run, monkeypatch):
    removed = _removals(monkeypatch)
    removal = run_removal.remove_run(run.run_id, with_bundle=True)
    assert removed == [path.name for path in _paths(run)]
    assert [f.name for f in removal.removed] == removed
    assert removal.kept is None and removal.failed is None
    assert not any(path.exists() for path in _paths(run))
    assert saved_runs()["runs"] == [] and list_bundles() == []


def test_a_bundle_kept_becomes_an_unregistered_bundle(run, monkeypatch):
    removed = _removals(monkeypatch)
    removal = run_removal.remove_run(run.run_id, with_bundle=False)
    bundle = bundles_dir() / f"{run.run_id}.bundle"
    assert removed == [path.name for path in _paths(run) if path != bundle]
    assert removal.kept is not None and removal.kept.path == bundle and bundle.exists()
    assert saved_runs()["runs"] == []
    assert [info.path for info in list_bundles()] == [bundle], "still listed as a bundle"


def test_a_file_that_cant_be_removed_keeps_the_record(run, monkeypatch):
    bundle = f"{run.run_id}.bundle"
    removed = _removals(monkeypatch, refuse=bundle)
    removal = run_removal.remove_run(run.run_id, with_bundle=True)
    assert removal.failed is not None and removal.failed.name == bundle
    assert removed == [path.name for path in _paths(run)[:2]], "stopped at the bundle"
    assert (runs_dir() / f"{run.run_id}.json").exists()
    assert [r["run_id"] for r in saved_runs()["runs"]] == [run.run_id], "still listed"
    message = run_removal.not_removed(removal)
    assert bundle in message and "kept" in message and "again" in message
    assert str(bundles_dir()) not in message, "a name, never a path"

    monkeypatch.setattr(run_removal, "remove_file", run_removal.remove_file.__wrapped__)
    again = run_removal.remove_run(run.run_id, with_bundle=True)
    assert again.failed is None and [f.kind for f in again.removed] == [BUNDLE, RECORD]
    assert not any(path.exists() for path in _paths(run))


def test_another_runs_files_are_left_alone(run, other_trained):
    other = _save(other_trained)
    run_removal.remove_run(run.run_id, with_bundle=True)
    assert all(path.exists() for path in _paths(other))
    assert [r["run_id"] for r in saved_runs()["runs"]] == [other.run_id]


@pytest.mark.parametrize("run_id", ["../x", "20261020T125523Z-7baf0b28/../../a", "",
                                    "20261020T125523Z-7BAF0B28", "x.importance.knn"])
def test_a_bad_run_id_is_refused_before_a_path(run_id):
    with pytest.raises(NotARun):
        run_removal.run_files(run_id)
    with pytest.raises(NotARun):
        run_removal.remove_run(run_id, with_bundle=True)


def test_no_record_no_delete(trained):
    saved = save_bundle(trained)
    importance_files.write_importance(runs_dir(), trained.run_id, "decision_tree", {})
    with pytest.raises(FileNotFoundError):
        run_removal.run_files(trained.run_id)
    with pytest.raises(FileNotFoundError):
        run_removal.remove_run(trained.run_id, with_bundle=True)
    assert saved.path.exists() and _paths(trained)[0].exists(), "orphans: the storage report's"


def test_a_record_that_isnt_one_can_be_deleted(trained):
    record = runs_dir() / f"{trained.run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{not json", encoding="utf-8")
    removal = run_removal.remove_run(trained.run_id, with_bundle=True)
    assert [f.kind for f in removal.removed] == [RECORD] and not record.exists()


def test_the_delete_waits_for_the_publishing_lock(run):
    done = threading.Event()
    with importance_files.PUBLISHING:
        worker = threading.Thread(target=lambda: (
            run_removal.remove_run(run.run_id, with_bundle=True), done.set()))
        worker.start()
        assert not done.wait(0.3), "it waited"
        assert all(path.exists() for path in _paths(run))
    worker.join(10)
    assert done.is_set() and not any(path.exists() for path in _paths(run))


def test_an_importance_computed_meanwhile_is_not_published(run, trained):
    found = permutation_importances(trained.evaluation.training.models["decision_tree"],
                                    trained.split.test, "Label", seed=trained.evaluation.seed)
    run_removal.remove_run(run.run_id, with_bundle=True)
    assert saved_importance.publish(run.run_id, "decision_tree", found) == \
        saved_importance.NO_RECORD
    assert not _paths(run)[0].exists()


def test_a_record_is_written_under_the_same_lock(other_trained):
    done = threading.Event()
    with importance_files.PUBLISHING:
        worker = threading.Thread(target=lambda: (write_run(other_trained), done.set()))
        worker.start()
        assert not done.wait(0.3), "it waited"
        assert not (runs_dir() / f"{other_trained.run_id}.json").exists()
    worker.join(10)
    assert done.is_set() and (runs_dir() / f"{other_trained.run_id}.json").exists()
