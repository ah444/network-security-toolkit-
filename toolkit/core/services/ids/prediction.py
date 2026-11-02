"""Predictions with a bundle exported by the web trainer (plan D5, step 0.5): one row of
values, or a whole CSV file labelled into a new file (the same table
as the web trainer's "Predict a CSV file", ``batch``, step 1.4f)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from ...data.files import read_text, write_new_text
from .batch import prediction_csv, prediction_summary, prediction_table
from .bundle import Bundle
from .datasets import load_dataset


def label_column(bundle: Bundle) -> str | None:
    """The run's label column (chosen, else the profile's), or None."""
    record = bundle.run
    return cast(str | None, record["settings"].get("label")
                or (record.get("profile") or {}).get("label_column"))


def model_order(bundle: Bundle) -> list[str]:
    """Model names, best first (the run's leaderboard order)."""
    ranked = [row["model"] for row in bundle.run["evaluation"]["leaderboard"]]
    return [name for name in ranked if name in bundle.models] + \
        [name for name in bundle.models if name not in ranked]


def predict_values(bundle: Bundle, values: Mapping[str, Any],
                   model: str | None = None) -> dict[str, Any]:
    """One row of feature values (text or numbers) -> {prediction, confidence, top}."""
    fitted = bundle.model(model)
    row = fitted.input_frame(values)
    proba = fitted.predict_proba(row)
    result: dict[str, Any] = {"prediction": str(fitted.predict(row)[0]), "model": fitted.name,
                              "confidence": None, "top": []}
    if proba is not None:
        order = np.argsort(proba[0])[::-1][:5]
        result["confidence"] = float(proba[0].max())
        result["top"] = [[fitted.classes[i], float(proba[0][i])] for i in order]
    return result


def output_path(csv_path: str | Path, output: str | Path | None) -> Path:
    """Where ``--csv`` writes: ``output``, else ``<file>.predictions.csv`` next to it."""
    csv_path = Path(csv_path)
    return Path(output) if output else csv_path.with_name(csv_path.stem + ".predictions.csv")


def label_file(bundle: Bundle, csv_path: str | Path, output: Path,
               model: str | None = None) -> dict[str, Any]:
    """Label every row of a CSV file into ``output`` (a new file, never overwritten); the
    summary: rows, counts per class, matches with the label column."""
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another --output")
    frame = load_dataset(csv_path).frame
    table = prediction_table(bundle.model(model), frame, label_column(bundle))
    write_new_text(output, prediction_csv(table))
    return prediction_summary(table)


def json_rows(text: str) -> list[dict[str, Any]]:
    """Rows given as JSON: an object or a list of them, inline or in a .json file."""
    source = Path(text)
    if text.strip()[:1] not in "[{" and source.is_file():
        text = read_text(source)
    rows = json.loads(text)
    rows = [rows] if isinstance(rows, dict) else rows
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("JSON input must be an object of feature values, or a list of them")
    return cast(list[dict[str, Any]], rows)
