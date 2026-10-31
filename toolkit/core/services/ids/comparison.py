"""Saved runs side by side (v5.3 F2): a run's summary, the view its charts need, whether two
runs can be compared and how they differ.

Works on TrainingRun records as dicts, the way ``exports.read_run`` gives them back, and only
reads the records (``instance/runs/*.json``), never a bundle: nothing is unpickled. A record on
disk is data someone could have edited, so a run id is checked against the form ``runs.train``
writes before any path is made from it, and a record that isn't a TrainingRun is left out of a
list and counted, never an error of the whole list.

Two runs are comparable when they scored the same test rows (the same data, the same test file,
the same split, the same row counts); only then are the differences (Δ) given.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...data.ids.run_files import list_runs, runs_dir
from .exports import read_run

RUN_ID = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")  # runs.train: f"{now:%Y%m%dT%H%M%SZ}-" + 8 hex
PAGE_SIZE = 50  # runs in one list, newest first

# The leaderboard's numbers each model is compared on, in the order the pages show them.
METRICS = ("cv_macro_f1_mean", "cv_balanced_accuracy_mean", "test_macro_f1",
           "test_balanced_accuracy", "test_accuracy", "test_min_recall", "fit_seconds")

# Why two runs are not comparable: each code is a part of what the test rows depend on.
DIFFERENT_DATASET = "DIFFERENT_DATASET"  # the dataset's SHA-256
DIFFERENT_TEST_FILE = "DIFFERENT_TEST_FILE"  # the separate test file's SHA-256, or one has none
DIFFERENT_LABEL = "DIFFERENT_LABEL"  # the label column
DIFFERENT_SPLIT = "DIFFERENT_SPLIT"  # strategy, seed, test size or official mode
DIFFERENT_ROWS = "DIFFERENT_ROWS"  # training or test row counts (deduplication, filters)

_SPLIT_KEYS = ("strategy", "seed", "test_size", "official_mode")


class NotARun(ValueError):
    """A run id in the wrong form, or a file that is not a readable TrainingRun record."""


def is_run_id(text: str) -> bool:
    """Whether ``text`` has the form of a run id (so a path made from it stays in the folder)."""
    return RUN_ID.fullmatch(text) is not None


def _selected_row(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The leaderboard row of the selected model, if any."""
    return next((row for row in record["evaluation"]["leaderboard"] if row["selected"]), None)


def summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """One line of the list: when, on what, how, and the selected model's scores. Raises
    KeyError or TypeError on a record that lacks a part (``saved_runs`` leaves it out)."""
    dataset = record["dataset"] or {}
    test_dataset = record["test_dataset"]
    profile = record["profile"]
    split = record["split"]
    row = _selected_row(record)
    return {
        "run_id": str(record["run_id"]),
        "created_at": str(record["created_at"]),
        "dataset": {"name": dataset.get("name"), "rows": dataset.get("rows")},
        "test_dataset": test_dataset["name"] if test_dataset else None,
        "profile": profile["title"] if profile else None,
        "label": record["settings"]["label"],
        "split": split["strategy"],
        "train_rows": split["train_rows"],
        "test_rows": split["test_rows"],
        "balancing": record["balancing"]["method"],
        "models": [r["model"] for r in record["evaluation"]["leaderboard"]],
        "selected": row["model"] if row else None,
        "selected_title": row["title"] if row else None,
        "cv_macro_f1_mean": row["cv_macro_f1_mean"] if row else None,
        "test_macro_f1": row["test_macro_f1"] if row else None,
    }


def folds(record: Mapping[str, Any]) -> dict[str, list[float]]:
    """Each model's CV macro-F1, fold by fold (the charts' dots)."""
    return {model["name"]: list(model["cv"]["macro_f1"]) for model in record["evaluation"]["models"]}


def view(record: Mapping[str, Any]) -> dict[str, Any]:
    """One saved run as the page shows it: its summary, its leaderboard and the folds."""
    return {"summary": summary(record), "leaderboard": record["evaluation"]["leaderboard"],
            "folds": folds(record)}


def _test_file_hash(record: Mapping[str, Any]) -> str | None:
    """The separate test file's SHA-256, or None when the test rows came from the split."""
    test_dataset = record["test_dataset"]
    return test_dataset["sha256"] if test_dataset else None


def comparability(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Why the two runs did not score the same test rows; empty when they did."""
    reasons = []
    if (a["dataset"] or {}).get("sha256") != (b["dataset"] or {}).get("sha256"):
        reasons.append(DIFFERENT_DATASET)
    if _test_file_hash(a) != _test_file_hash(b):
        reasons.append(DIFFERENT_TEST_FILE)
    if a["settings"]["label"] != b["settings"]["label"]:
        reasons.append(DIFFERENT_LABEL)
    if any(a["split"].get(key) != b["split"].get(key) for key in _SPLIT_KEYS):
        reasons.append(DIFFERENT_SPLIT)
    if any(a["split"][key] != b["split"][key] for key in ("train_rows", "test_rows")):
        reasons.append(DIFFERENT_ROWS)
    return reasons


def _settings(record: Mapping[str, Any]) -> dict[str, Any]:
    """The choices a user varies between two runs, as the comparison lists them."""
    settings = record["settings"]
    return {"features": list(record["features"]),
            "models": list(settings["models"]),
            "balancing": record["balancing"]["method"],
            "selection": [settings["selector"], settings["k_features"]]}


def _recalls(a_row: Mapping[str, Any], b_row: Mapping[str, Any],
             comparable: bool) -> list[dict[str, Any]]:
    """The test recall of the classes present in both runs' test rows, in the first run's order."""
    b_recall = dict(b_row["test_recall_per_class"])
    return [{"label": label, "a": recall, "b": b_recall[label],
             "delta": b_recall[label] - recall if comparable else None}
            for label, recall in a_row["test_recall_per_class"] if label in b_recall]


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """The two runs side by side: each model of both with its numbers (and Δ = b − a when they
    are comparable), the models of one run only, and the settings that differ."""
    reasons = comparability(a, b)
    comparable = not reasons
    a_rows = {row["model"]: row for row in a["evaluation"]["leaderboard"]}
    b_rows = {row["model"]: row for row in b["evaluation"]["leaderboard"]}
    models = []
    for name, a_row in a_rows.items():
        if name not in b_rows:
            continue
        b_row = b_rows[name]
        models.append({
            "model": name, "title": a_row["title"],
            "selected": [a_row["selected"], b_row["selected"]],
            "a": {key: a_row[key] for key in METRICS},
            "b": {key: b_row[key] for key in METRICS},
            "delta": {key: b_row[key] - a_row[key] for key in METRICS} if comparable else None,
            "recall": _recalls(a_row, b_row, comparable),
        })
    a_settings, b_settings = _settings(a), _settings(b)
    return {
        "a": summary(a), "b": summary(b),
        "comparable": comparable, "reasons": reasons,
        "models": models,
        "only_in_a": [name for name in a_rows if name not in b_rows],
        "only_in_b": [name for name in b_rows if name not in a_rows],
        "titles": {row["model"]: row["title"] for row in [*b_rows.values(), *a_rows.values()]},
        "settings": [{"setting": key, "a": a_settings[key], "b": b_settings[key]}
                     for key in a_settings if a_settings[key] != b_settings[key]],
    }


def _record_path(run_id: str, directory: Path) -> Path:
    """``<directory>/<run id>.json``; NotARun unless ``run_id`` has the form of a run id."""
    if not is_run_id(run_id):
        raise NotARun("That is not a run id.")
    return directory / f"{run_id}.json"


def _checked(path: Path) -> dict[str, Any]:
    """The record at ``path`` if it is a TrainingRun the pages can show; NotARun otherwise."""
    try:
        record = read_run(path)
        if record["run_id"] != path.stem:
            raise ValueError("renamed")
        summary(record)
        folds(record)
        return record
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        raise NotARun(f"Run {path.stem} is not a TrainingRun record this toolkit can show.") \
            from error


def saved_run(run_id: str, directory: str | Path | None = None) -> dict[str, Any]:
    """The saved record of ``run_id``: NotARun for a bad id or record, FileNotFoundError when
    there is none."""
    path = _record_path(run_id, Path(directory) if directory is not None else runs_dir())
    if not path.is_file():
        raise FileNotFoundError(run_id)
    return _checked(path)


def saved_runs(before: str | None = None, limit: int = PAGE_SIZE,
               directory: str | Path | None = None) -> dict[str, Any]:
    """The summaries of the saved runs, newest first, at most ``limit``, older than ``before``
    when given: ``{"runs", "more", "left_out"}``. ``left_out`` counts the files met while
    filling this list that are not readable TrainingRun records (or not named as one)."""
    if before is not None and not is_run_id(before):
        raise NotARun("That is not a run id.")
    paths = [path for path in reversed(list_runs(directory))
             if before is None or path.stem < before]
    runs: list[dict[str, Any]] = []
    left_out = 0
    read = 0
    for path in paths:
        if len(runs) == limit:
            break
        read += 1
        if not is_run_id(path.stem):
            left_out += 1
            continue
        try:
            runs.append(summary(_checked(path)))
        except NotARun:
            left_out += 1
    return {"runs": runs, "more": read < len(paths), "left_out": left_out}


def compare_saved(a: str, b: str, directory: str | Path | None = None) -> dict[str, Any]:
    """``compare`` of two saved runs (the same errors as ``saved_run``; both ids are checked
    before either record is looked for)."""
    if not (is_run_id(a) and is_run_id(b)):
        raise NotARun("That is not a run id.")
    return compare(saved_run(a, directory), saved_run(b, directory))

