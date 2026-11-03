"""v5.3 step HR2 (RD4: IA2, IA5-IA7): each permutation importance result is a file of its
own, ``<run id>.importance.<model id>.json`` beside its run's record. Its name comes from
checked ids only; it is published once, under the one lock of the run files, and only while
the record lists that model; it is never replaced; it is read back as untrusted data."""

import json
import threading

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids import importance_files
from toolkit.core.data.ids.run_files import list_runs, runs_dir
from toolkit.core.services.ids import saved_importance
from toolkit.core.services.ids.comparison import saved_runs
from toolkit.core.services.ids.exports import write_run
from toolkit.core.services.ids.importance import permutation_importances
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
def run():
    return _run(0)


@pytest.fixture(scope="module")
def found(run):
    return permutation_importances(run.evaluation.training.models["decision_tree"],
                                   run.split.test, "Label", seed=run.evaluation.seed)


@pytest.fixture
def saved(run):
    write_run(run)
    return run


def _path(run, model="decision_tree"):
    return runs_dir() / f"{run.run_id}.importance.{model}.json"


def test_a_result_is_published_beside_its_record(saved, found):
    assert saved_importance.publish(saved.run_id, "decision_tree", found) is None
    stored = json.loads(_path(saved).read_text(encoding="utf-8"))
    assert {key: stored[key] for key in ("schema_version", "method", "method_version", "run_id",
                                         "model")} == {
        "schema_version": 1, "method": "permutation_importance", "method_version": 1,
        "run_id": saved.run_id, "model": "decision_tree"}
    assert stored["settings"] == {"max_rows": 5000, "min_per_class": 3, "repeats": 5,
                                  "seed": saved.evaluation.seed, "metric": "macro-F1"}
    assert (stored["rows"], stored["test_rows"]) == (found.rows, found.test_rows)
    assert stored["features"] == [{"feature": f.feature, "mean": f.mean, "std": f.std}
                                  for f in found.features]
    assert stored["baseline_macro_f1"] == found.baseline_macro_f1
    assert stored["concentrated"] == found.concentrated
    assert stored["versions"]["scikit-learn"] and stored["created_at"].endswith("Z")
    assert saved_importance.saved_importance(saved.run_id, "decision_tree") == stored


def test_another_run_leaves_it_alone(saved, found):
    saved_importance.publish(saved.run_id, "decision_tree", found)
    before = _path(saved).read_bytes()
    write_run(_run(1))  # a training after it
    assert _path(saved).read_bytes() == before
    listed = saved_importance.saved_importances(saved.run_id)
    assert [item["model"] for item in listed["importances"]] == ["decision_tree"]
    assert listed["left_out"] == 0


def test_a_published_result_is_never_replaced(saved, found):
    saved_importance.publish(saved.run_id, "decision_tree", found)
    before = _path(saved).read_bytes()
    other = permutation_importances(saved.evaluation.training.models["decision_tree"],
                                    saved.split.test, "Label", seed=7)
    assert saved_importance.publish(saved.run_id, "decision_tree", other) is None, "kept"
    assert _path(saved).read_bytes() == before


def test_without_a_record_nothing_is_published(run, found):
    assert saved_importance.publish(run.run_id, "decision_tree", found) == (
        "Not saved: the run isn't saved yet. Export it to keep its results.")
    assert not _path(run).exists()


def test_a_run_deleted_while_it_computes_gets_nothing(saved, found):
    (runs_dir() / f"{saved.run_id}.json").unlink()  # as HR4's delete will, meanwhile
    assert saved_importance.publish(saved.run_id, "decision_tree", found).startswith("Not saved")
    assert not _path(saved).exists()


def test_a_model_the_record_lacks_is_not_published(saved, found):
    assert saved_importance.publish(saved.run_id, "knn", found) == (
        "Not saved: the run's record has no model knn.")
    assert not _path(saved, "knn").exists()


@pytest.mark.parametrize("run_id, model", [
    ("../20261103T000000Z-00000000", "decision_tree"), ("not-a-run", "decision_tree"),
    ("20261103T000000Z-00000000", "../../x"), ("20261103T000000Z-00000000", "Decision Tree"),
    ("20261103T000000Z-00000000", "knn.json"),
])
def test_a_name_comes_from_checked_ids_only(run_id, model, found, monkeypatch):
    def no_path(*args, **kwargs):
        raise AssertionError("a path was made")

    monkeypatch.setattr(importance_files, "importance_path", no_path)
    for call in (lambda: saved_importance.publish(run_id, model, found),
                 lambda: saved_importance.saved_importance(run_id, model)):
        with pytest.raises(saved_importance.NotAnImportance):
            call()


def test_the_record_check_and_the_publish_hold_the_lock(saved, found, monkeypatch):
    held = []
    real = importance_files.write_importance

    def write(*args, **kwargs):
        held.append(importance_files.PUBLISHING.locked())
        return real(*args, **kwargs)

    monkeypatch.setattr(importance_files, "write_importance", write)
    importance_files.PUBLISHING.acquire()  # a delete in progress (HR4)
    done = threading.Event()
    worker = threading.Thread(target=lambda: (saved_importance.publish(
        saved.run_id, "decision_tree", found), done.set()))
    worker.start()
    assert not done.wait(0.3), "waits for the lock"
    importance_files.PUBLISHING.release()
    worker.join(30)
    assert done.is_set() and held == [True]


def _write(run, model, content):
    path = _path(run, model)
    path.write_text(content if isinstance(content, str) else json.dumps(content),
                    encoding="utf-8")
    return path


def test_bad_files_are_left_out_and_counted(saved, found, monkeypatch):
    saved_importance.publish(saved.run_id, "decision_tree", found)
    good = json.loads(_path(saved).read_text(encoding="utf-8"))
    _write(saved, "logistic_regression", {**good, "model": "decision_tree"})  # not its name
    _write(saved, "knn", {**good, "model": "knn", "schema_version": 2})  # newer
    _write(saved, "naive_bayes", "{not json")
    _write(saved, "random_forest", {**good, "model": "random_forest", "features": [{"x": 1}]})
    _write(saved, "svm", {**good, "model": "svm", "padding": "x" * 2**20})  # too big
    _write(saved, "Not A Model", good)
    listed = saved_importance.saved_importances(saved.run_id)
    assert [item["model"] for item in listed["importances"]] == ["decision_tree"]
    assert listed["left_out"] == 6
    for model in ("logistic_regression", "knn", "naive_bayes", "random_forest", "svm"):
        assert saved_importance.saved_importance(saved.run_id, model) is None
    assert saved_importance.saved_importance(saved.run_id, "xgboost") is None, "none saved"


def test_the_saved_runs_dont_count_importance_files(saved, found):
    saved_importance.publish(saved.run_id, "decision_tree", found)
    assert list_runs() == [runs_dir() / f"{saved.run_id}.json"]
    listed = saved_runs()
    assert [run["run_id"] for run in listed["runs"]] == [saved.run_id]
    assert listed["left_out"] == 0
