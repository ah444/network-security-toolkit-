"""Batch prediction: every row of a table labelled by a fitted model (plan F1, v5.1 step 1.4f).

The CLI (``python -m toolkit ids --csv``) and the web trainer's "Predict a CSV file" both
write ``prediction_table`` with ``prediction_csv``, so they give the same file for the same
model and rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .pipeline import FittedModel


def missing_features(fitted: FittedModel, frame: pd.DataFrame) -> list[str]:
    return [name for name in fitted.features if name not in frame.columns]


def prediction_table(fitted: FittedModel, frame: pd.DataFrame,
                     label: str | None = None) -> pd.DataFrame:
    """``row`` (1-based), ``prediction``, ``confidence`` (the top class probability, empty
    when the model has none) and, when ``frame`` has the label column, ``true_label``.
    ValueError naming the missing feature columns."""
    missing = missing_features(fitted, frame)
    if missing:
        raise ValueError(f"Missing feature column(s): {', '.join(missing)}")
    proba = fitted.predict_proba(frame)
    confidence = proba.max(axis=1) if proba is not None else np.full(len(frame), np.nan)
    table = pd.DataFrame({"row": np.arange(1, len(frame) + 1),
                          "prediction": fitted.predict(frame),
                          "confidence": np.round(confidence, 4)})
    if label and label in frame.columns:
        table["true_label"] = frame[label].astype(str).to_numpy()
    return table


def prediction_csv(table: pd.DataFrame) -> str:
    """The table as CSV text (written with ``newline=""`` by the CLI, sent as UTF-8 by
    the web trainer)."""
    return table.to_csv(index=False)


def prediction_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Rows, per-class counts (largest first) and, with ``true_label``, how many match."""
    rows = len(table)
    counts = table["prediction"].astype(str).value_counts()
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    summary: dict[str, Any] = {
        "rows": rows,
        "counts": [{"label": label, "count": int(count), "share": count / rows if rows else 0.0}
                   for label, count in ordered],
        "matches": None,
    }
    if "true_label" in table:
        matching = int((table["prediction"].astype(str) == table["true_label"]).sum())
        summary["matches"] = {"matching": matching, "rows": rows,
                              "share": matching / rows if rows else 0.0}
    return summary
