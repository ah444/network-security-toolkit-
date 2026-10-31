"""v5.3 F2b: saved runs side by side (toolkit.core.services.ids.comparison)."""

import copy
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids import bundle_files, run_files
from toolkit.core.services.ids import comparison
from toolkit.core.services.ids import exports as core_exports
from toolkit.core.services.ids.comparison import (
    DIFFERENT_DATASET,
    DIFFERENT_LABEL,
    DIFFERENT_ROWS,
    DIFFERENT_SPLIT,
    DIFFERENT_TEST_FILE,
    NotARun,
)
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train

CLASSES = ("BENIGN", "DoS", "PortScan")


def _frame(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.6, 0.25, 0.15])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 2.0, "PortScan": -2.0}).to_numpy()
    return pd.DataFrame({"a": shift + rng.normal(size=rows), "b": rng.normal(size=rows),
                         "Label": label})


@pytest.fixture(scope="module")
def record():
    settings = TrainSettings(label="Label", strategy="stratified",
                             models=("decision_tree", "logistic_regression"), balancer="none",
                             workers=1)
    source = DatasetRef(name="synthetic.csv", sha256="ab" * 32, rows=400, columns=3)
    return train(_frame(), settings=settings, source=source).to_dict()


def _other(record, run_id="20261030T120000Z-0000000b"):
    """A second run of the same data: the same record with other scores."""
    other = copy.deepcopy(record)
    other["run_id"] = run_id
    for row in other["evaluation"]["leaderboard"]:
        row["test_macro_f1"] -= 0.25
        row["test_recall_per_class"] = [[label, recall - 0.5]
                                        for label, recall in row["test_recall_per_class"]]
    return other


def _save(directory, record, run_id=None, name=None):
    record = copy.deepcopy(record)
    if run_id:
        record["run_id"] = run_id
    path = directory / f"{name or record['run_id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


@pytest.mark.parametrize("text, expected", [
    ("20261020T110444Z-7cf3ced5", True),
    ("20261020T110444Z-7CF3CED5", False),  # train writes lower-case hex
    ("20261020T110444Z-7cf3ced5.json", False),
    ("../20261020T110444Z-7cf3ced5", False),
    ("20261020T110444Z-7cf3ced5\n", False),
    ("20261020T110444Z-7cf3ced", False),
    ("", False),
])
def test_run_id_form(text, expected):
    assert comparison.is_run_id(text) is expected


def test_a_trained_run_id_has_the_form(record):
    assert comparison.is_run_id(record["run_id"])


def test_summary(record):
    summary = comparison.summary(record)
    selected = next(row for row in record["evaluation"]["leaderboard"] if row["selected"])
    assert summary == {
        "run_id": record["run_id"], "created_at": record["created_at"],
        "dataset": {"name": "synthetic.csv", "rows": 400}, "test_dataset": None,
        "profile": None, "label": "Label", "split": "stratified",
        "train_rows": record["split"]["train_rows"], "test_rows": record["split"]["test_rows"],
        "balancing": "none", "models": [r["model"] for r in record["evaluation"]["leaderboard"]],
        "selected": selected["model"], "selected_title": selected["title"],
        "cv_macro_f1_mean": selected["cv_macro_f1_mean"],
        "test_macro_f1": selected["test_macro_f1"],
    }


def test_view_has_every_fold(record):
    view = comparison.view(record)
    assert view["leaderboard"] == record["evaluation"]["leaderboard"]
    assert set(view["folds"]) == {"decision_tree", "logistic_regression"}
    for scores in view["folds"].values():
        assert len(scores) == record["evaluation"]["cv_folds"]
        assert all(0 <= score <= 1 for score in scores)


def test_the_same_test_rows_are_comparable(record):
    assert comparison.comparability(record, _other(record)) == []


@pytest.mark.parametrize("change, reason", [
    (lambda r: r["dataset"].update(sha256="cd" * 32), DIFFERENT_DATASET),
    (lambda r: r.update(test_dataset={"name": "t.csv", "sha256": "ef" * 32, "rows": 9,
                                      "columns": 3}), DIFFERENT_TEST_FILE),
    (lambda r: r["settings"].update(label="attack"), DIFFERENT_LABEL),
    (lambda r: r["split"].update(strategy="temporal"), DIFFERENT_SPLIT),
    (lambda r: r["split"].update(seed=7), DIFFERENT_SPLIT),
    (lambda r: r["split"].update(test_size=0.3), DIFFERENT_SPLIT),
    (lambda r: r["split"].update(official_mode="deduplicated"), DIFFERENT_SPLIT),
    (lambda r: r["split"].update(test_rows=r["split"]["test_rows"] - 1), DIFFERENT_ROWS),
    (lambda r: r["split"].update(train_rows=r["split"]["train_rows"] + 1), DIFFERENT_ROWS),
])
def test_what_makes_two_runs_not_comparable(record, change, reason):
    other = _other(record)
    change(other)
    assert comparison.comparability(record, other) == [reason]


def test_compare_comparable_runs(record):
    other = _other(record)
    other["features"] = ["a"]
    other["balancing"]["method"] = "smote"
    result = comparison.compare(record, other)
    assert result["comparable"] is True and result["reasons"] == []
    assert result["a"]["run_id"] == record["run_id"] and result["b"]["run_id"] == other["run_id"]
    assert [m["model"] for m in result["models"]] == \
        [r["model"] for r in record["evaluation"]["leaderboard"]]
    for model in result["models"]:
        assert set(model["a"]) == set(comparison.METRICS) == set(model["delta"])
        assert model["delta"]["test_macro_f1"] == pytest.approx(-0.25)
        assert model["delta"]["cv_macro_f1_mean"] == 0
        assert [r["label"] for r in model["recall"]] == list(CLASSES)
        assert all(r["delta"] == pytest.approx(-0.5) for r in model["recall"])
    assert result["only_in_a"] == [] and result["only_in_b"] == []
    assert result["settings"] == [
        {"setting": "features", "a": record["features"], "b": ["a"]},
        {"setting": "balancing", "a": "none", "b": "smote"},
    ]


def test_runs_on_other_data_have_no_differences(record):
    other = _other(record)
    other["dataset"]["sha256"] = "cd" * 32
    result = comparison.compare(record, other)
    assert result["comparable"] is False and result["reasons"] == [DIFFERENT_DATASET]
    for model in result["models"]:
        assert model["delta"] is None
        assert model["a"] and model["b"]
        assert all(r["delta"] is None for r in model["recall"])


def test_models_and_classes_of_one_run_only(record):
    other = _other(record)
    board = other["evaluation"]["leaderboard"]
    dropped = board.pop()["model"]
    board.append({**board[0], "model": "random_forest", "title": "Random Forest"})
    board[0]["test_recall_per_class"] = board[0]["test_recall_per_class"][1:]
    result = comparison.compare(record, other)
    assert result["only_in_a"] == [dropped] and result["only_in_b"] == ["random_forest"]
    assert result["titles"][dropped] == record["evaluation"]["leaderboard"][-1]["title"]
    assert result["titles"]["random_forest"] == "Random Forest", "every model of both, titled"
    first = next(m for m in result["models"] if m["model"] == board[0]["model"])
    assert [r["label"] for r in first["recall"]] == list(CLASSES[1:])


def test_saved_runs_newest_first_in_pages(tmp_path, record):
    ids = [f"20261030T1200{n:02d}Z-0000000{n}" for n in range(1, 6)]
    for run_id in ids:
        _save(tmp_path, record, run_id)
    first = comparison.saved_runs(limit=2, directory=tmp_path)
    assert [r["run_id"] for r in first["runs"]] == ids[:-3:-1]
    assert first["more"] is True and first["left_out"] == 0
    second = comparison.saved_runs(before=ids[3], limit=2, directory=tmp_path)
    assert [r["run_id"] for r in second["runs"]] == [ids[2], ids[1]]
    last = comparison.saved_runs(before=ids[1], limit=2, directory=tmp_path)
    assert [r["run_id"] for r in last["runs"]] == [ids[0]] and last["more"] is False


def test_saved_runs_default_folder_and_none_yet(record):
    assert comparison.saved_runs() == {"runs": [], "more": False, "left_out": 0}
    run_files.write_record(record["run_id"], record)
    assert [r["run_id"] for r in comparison.saved_runs()["runs"]] == [record["run_id"]]


def test_bad_files_are_left_out_and_counted(tmp_path, record):
    good = _save(tmp_path, record, "20261030T120001Z-00000001")
    (tmp_path / "20261030T120002Z-00000002.json").write_text("{not json", encoding="utf-8")
    newer = copy.deepcopy(record)
    newer["schema_version"] = 99
    _save(tmp_path, newer, "20261030T120003Z-00000003")
    _save(tmp_path, record, "20261030T120004Z-00000004", name="20261030T120005Z-00000005")
    broken = copy.deepcopy(record)
    del broken["split"]
    _save(tmp_path, broken, "20261030T120006Z-00000006")
    _save(tmp_path, record, "notes", name="notes")
    (tmp_path / "20261030T120007Z-00000007.json").write_text("[1, 2]", encoding="utf-8")
    listed = comparison.saved_runs(directory=tmp_path)
    assert [r["run_id"] for r in listed["runs"]] == [good["run_id"]]
    assert listed["left_out"] == 6 and listed["more"] is False


def test_a_bad_page_start_is_refused(tmp_path):
    with pytest.raises(NotARun):
        comparison.saved_runs(before="../x", directory=tmp_path)


def test_saved_run(tmp_path, record):
    saved = _save(tmp_path, record, "20261030T120001Z-00000001")
    assert comparison.saved_run(saved["run_id"], tmp_path) == saved
    with pytest.raises(FileNotFoundError):
        comparison.saved_run("20261030T120009Z-00000009", tmp_path)
    for bad in ("..\\20261030T120001Z-00000001", "../20261030T120001Z-00000001",
                str(tmp_path / "20261030T120001Z-00000001"), "20261030T120001Z-00000001.json"):
        with pytest.raises(NotARun):
            comparison.saved_run(bad, tmp_path)
    (tmp_path / "20261030T120002Z-00000002.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NotARun, match="not a TrainingRun record"):
        comparison.saved_run("20261030T120002Z-00000002", tmp_path)


def test_compare_saved(tmp_path, record):
    a = _save(tmp_path, record, "20261030T120001Z-00000001")
    b = _save(tmp_path, _other(record), "20261030T120002Z-00000002")
    result = comparison.compare_saved(a["run_id"], b["run_id"], tmp_path)
    assert result == comparison.compare(a, b)


def test_no_bundle_is_read(tmp_path, record, monkeypatch):
    """Only the records: no bundle file is opened, so nothing is unpickled."""
    def refuse(*args, **kwargs):
        raise AssertionError("a bundle was read")
    for module in (bundle_files, core_exports):
        for name in ("read_all", "read_start", "stored_bundles", "signing_key"):
            monkeypatch.setattr(module, name, refuse)
    monkeypatch.setattr(pickle, "loads", refuse)
    monkeypatch.setattr(pickle, "load", refuse)
    a = _save(tmp_path, record, "20261030T120001Z-00000001")
    comparison.saved_runs(directory=tmp_path)
    comparison.compare_saved(a["run_id"], a["run_id"], tmp_path)
