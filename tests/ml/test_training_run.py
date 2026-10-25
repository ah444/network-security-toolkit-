"""Step 0.4e: TrainingRun record returned by train(), written by the caller [D21]."""

import json

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids import run_files
from toolkit.core.services.ids import exports as core_exports
from toolkit.core.data.settings import instance_dir
from toolkit.core.services.ids.datasets import load_dataset
from toolkit.core.services.ids.evaluation import EvaluationResult
from toolkit.core.services.ids.runs import (
    SCHEMA_VERSION,
    DatasetRef,
    TrainSettings,
    compare_metrics,
    rerun,
    train,
)

CLASSES = ("BENIGN", "DoS", "PortScan")


def _frame(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.65, 0.25, 0.10])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 2.0, "PortScan": -2.0}).to_numpy()
    return pd.DataFrame({
        "a": shift + rng.normal(size=rows),
        "b": rng.normal(size=rows),
        "proto": np.where(label == "PortScan", "udp", rng.choice(["tcp", "udp"], size=rows)),
        "Flow ID": [f"f{i}" for i in range(rows)],  # blocked by the leakage guard
        "copy": label,  # a renamed copy of the label: blocked unless allowed
        "Label": label,
    })


SETTINGS = TrainSettings(label="Label", strategy="stratified",
                         models=("decision_tree", "logistic_regression"), balancer="recommended",
                         workers=1)
SOURCE = DatasetRef(name="synthetic.csv", sha256="ab" * 32, rows=600, columns=6)


@pytest.fixture(scope="module")
def run():
    return train(_frame(), settings=SETTINGS, source=SOURCE)


def test_record_complete(run):
    record = run.to_dict()
    json.dumps(record, allow_nan=False)  # plain JSON, no NaN / objects
    assert record["schema_version"] == SCHEMA_VERSION == 1
    assert record["run_id"] == run.run_id and record["created_at"].endswith("Z")
    assert record["dataset"] == {"name": "synthetic.csv", "sha256": "ab" * 32, "rows": 600,
                                 "columns": 6}
    assert record["test_dataset"] is None
    assert record["profile"] is None  # in-memory frame, no profile given
    assert record["settings"]["label"] == "Label"
    assert record["settings"]["models"] == ["decision_tree", "logistic_regression"]

    split = record["split"]
    assert (split["strategy"], split["requested"]) == ("stratified", "stratified")
    assert "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC" in split["warnings"]
    assert split["reason_codes"] == []
    assert split["train_rows"] + split["test_rows"] == split["dedupe"]["rows_after"]
    assert split["seed"] == 42 and split["test_size"] == 0.2
    assert {c["label"] for c in split["coverage"]} == set(CLASSES)

    guard = record["leakage"]
    assert set(guard["blocked"]) == {"Flow ID", "copy"}
    assert {(f["column"], f["reason"]) for f in guard["findings"]} >= {
        ("Flow ID", "NAME_PATTERN"), ("copy", "LABEL_RELABELLING")}
    assert record["features"] == ["a", "b", "proto"]

    assert record["balancing"]["method"] == run.evaluation.balancer
    assert record["balancing"]["recommendation"]["reason_code"]
    for name in ("decision_tree", "logistic_regression"):
        model = record["models"][name]
        assert model["params"]["random_state"] == 42
        assert model["steps"] == ["prepare", "select", "sample", "model"]
        assert model["numeric"] and model["categorical"]
    assert record["models"]["logistic_regression"]["numeric"].endswith("standard scaling")
    assert "one-hot" in record["models"]["logistic_regression"]["categorical"]
    assert "ordinal" in record["models"]["decision_tree"]["categorical"]

    evaluation = record["evaluation"]
    assert evaluation["selected"] == run.evaluation.selected
    assert len(evaluation["leaderboard"]) == 2
    first = evaluation["models"][0]
    assert first["cv"]["macro_f1_mean"] == pytest.approx(np.mean(first["cv"]["macro_f1"]))
    assert "training" not in evaluation, "fitted models are not part of the record"

    versions = record["versions"]
    for package in ("python", "scikit-learn", "numpy", "pandas", "imbalanced-learn", "xgboost",
                    "lightgbm", "joblib"):
        assert versions[package], package
    assert record["reproducibility"] == {"workers": 1, "metric_tolerance": 0.0}
    assert record["bundle_sha256"] is None  # filled when a bundle is written (0.4f)


def test_train_returns_fitted_models_and_evaluation(run):
    assert isinstance(run.evaluation, EvaluationResult)
    fitted = run.evaluation.training.models[run.evaluation.selected]
    assert len(fitted.predict(_frame(rows=30, seed=9))) == 30


def test_override_recorded():
    settings = TrainSettings(label="Label", strategy="stratified", models=("decision_tree",),
                             allow={"copy": "kept on purpose for this test"}, workers=1)
    record = train(_frame(), settings=settings).to_dict()
    assert "copy" in record["features"]
    assert record["leakage"]["overrides"] == [{
        "column": "copy", "action": "block", "reason": "LABEL_RELABELLING",
        "score": pytest.approx(1.0), "note": "kept on purpose for this test"}]
    assert record["settings"]["allow"] == {"copy": "kept on purpose for this test"}


def test_profile_and_official_test_file_recorded(fixtures_dir):
    train_file = load_dataset(fixtures_dir / "nslkdd_train_sample.txt")
    test_file = load_dataset(fixtures_dir / "nslkdd_test_sample_noheader.txt")
    settings = TrainSettings(models=("decision_tree",), workers=1)
    run = train(train_file.frame, profile=train_file.profile, settings=settings,
                test_frame=test_file.frame,
                source=DatasetRef.from_loaded("nslkdd_train_sample.txt", train_file),
                test_source=DatasetRef.from_loaded("nslkdd_test_sample_noheader.txt", test_file))
    record = run.to_dict()
    assert record["profile"]["name"] == "nsl_kdd"
    assert record["profile"]["label_column"] == "attack"
    assert record["profile"]["blocked_columns"] == ["level"]
    assert (record["split"]["strategy"], record["split"]["official_mode"]) == (
        "official", "as_published")
    assert record["dataset"]["sha256"] == train_file.report.sha256
    assert record["test_dataset"]["sha256"] == test_file.report.sha256
    assert "level" in record["leakage"]["blocked"] and "level" not in record["features"]


def test_label_required():
    with pytest.raises(ValueError, match="label"):
        train(_frame(), settings=TrainSettings(models=("decision_tree",)))


# --- writer (edge code) ----------------------------------------------------------------

def test_writer_roundtrip(run, tmp_path):
    path = core_exports.write_run(run, tmp_path)
    assert path == tmp_path / f"{run.run_id}.json"
    assert core_exports.read_run(path) == json.loads(json.dumps(run.to_dict()))
    with pytest.raises(FileExistsError):
        core_exports.write_run(run, tmp_path)  # a run record is never overwritten
    assert run_files.list_runs(tmp_path) == [path]


def test_writer_default_is_instance_runs(monkeypatch, tmp_path, repo_root):
    monkeypatch.delenv("TOOLKIT_INSTANCE_DIR", raising=False)
    assert instance_dir() == repo_root / "instance"
    assert run_files.runs_dir() == repo_root / "instance" / "runs"
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    assert run_files.runs_dir() == tmp_path / "runs"


def test_reader_rejects_newer_schema_and_non_records(tmp_path):
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "run_id": "x"}),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        core_exports.read_run(newer)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="TrainingRun"):
        core_exports.read_run(other)


# --- re-run ------------------------------------------------------------------------------

def test_rerun_reproduces_metrics(run, tmp_path):
    record = core_exports.read_run(core_exports.write_run(run, tmp_path))
    again = rerun(record, _frame(), source=SOURCE)
    assert again.run_id != run.run_id
    assert compare_metrics(record, again.to_dict()) == []
    assert again.to_dict()["settings"] == record["settings"]


def test_rerun_refuses_other_data(run):
    record = run.to_dict()
    other = DatasetRef(name="other.csv", sha256="cd" * 32, rows=600, columns=6)
    with pytest.raises(ValueError, match="SHA-256"):
        rerun(record, _frame(), source=other)


def test_compare_metrics_reports_differences(run):
    record = run.to_dict()
    changed = json.loads(json.dumps(record))
    changed["evaluation"]["models"][0]["test"]["macro_f1"] += 0.1
    differences = compare_metrics(record, changed)
    assert len(differences) == 1 and "macro_f1" in differences[0]
    assert compare_metrics(record, changed, tolerance=0.2) == []
