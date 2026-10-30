"""Predictions with the session's trained or loaded models: one row, or every row of an
uploaded CSV file (step 1.4f; the labelled CSV is the CLI's, ``toolkit.core.services.ids.batch``)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import IO, Any

import pandas as pd

from ...core.data.ids.dataset_files import DatasetError
from ..data import uploads
from ...core.services.ids import registry
from ...core.services.ids.batch import missing_features, prediction_csv, prediction_summary, prediction_table
from ...core.services.ids.datasets import load_dataset
from ...core.services.ids.pipeline import FittedModel
from .errors import TOO_LARGE, UNKNOWN_COLUMN, UNREADABLE_FILE, Failed, ServiceError
from .workspace import BatchPredictions, Workspace

log = logging.getLogger(__name__)


def predict_one(models: Mapping[str, FittedModel], default: str | None,
                values: Mapping[str, Any] | None, model: str | None) -> dict[str, Any]:
    """Predict one row with ``models[model]`` through its saved pipeline (B2)."""
    if not values:
        raise ServiceError("No input values provided.")
    algorithm = model or default or ""
    if algorithm not in models:
        raise ServiceError(f"Invalid algorithm selection. Available models: {', '.join(models)}")
    fitted = models[algorithm]
    try:
        row = fitted.input_frame(values)
    except ValueError as error:
        message = str(error)  # written by FittedModel.input_frame for the user (no internals)
        raise ServiceError(message) from None
    try:
        prediction = fitted.predict(row)[0]
    except Exception:
        log.exception("Prediction error")
        raise Failed("Prediction failed for these input values.") from None
    return {"prediction": str(prediction), "model": algorithm}


def model_title(name: str) -> str:
    """A model's title from the registry, else its name."""
    return registry.MODELS[name].title if name in registry.MODELS else name


def read_prediction_file(stream: IO[bytes], path: Path, name: str,
                         max_mb: int) -> pd.DataFrame:
    """The uploaded file to predict, read like a dataset upload: on disk at ``path`` only
    while it is read (S2, S3, S11)."""
    try:
        if uploads.save(stream, path) > max_mb * 2**20:
            raise ServiceError(too_large(max_mb), TOO_LARGE)
        return load_dataset(path).frame
    except (DatasetError, ValueError, UnicodeError, OSError):
        log.exception("Could not read a file to predict")
        raise ServiceError(f"Could not read {name} as a CSV/TXT file. Check the file "
                           "format.", UNREADABLE_FILE) from None
    finally:
        uploads.remove(path)


def too_large(max_mb: int) -> str:
    """The message for a file above the prediction limit."""
    return f"File too large to predict (limit: {max_mb} MB, TOOLKIT_MAX_PREDICT_MB)."


def predict_file(ws: Workspace, source: str, fitted: FittedModel, frame: pd.DataFrame,
                 file_name: str, label: str | None) -> dict[str, Any]:
    """Label every row of ``frame``; keeps the labelled CSV in the workspace for the
    download, and answers counts per class and matches with the label column."""
    algorithm = fitted.name
    missing = missing_features(fitted, frame)
    if missing:
        shown = ", ".join(missing[:20]) + (f" and {len(missing) - 20} more" if len(missing) > 20
                                           else "")
        raise ServiceError(f"The file is missing feature column(s) of this model: {shown}.",
                           UNKNOWN_COLUMN)
    try:
        table = prediction_table(fitted, frame, label)
    except Exception:
        log.exception("Batch prediction failed")
        raise ServiceError("Prediction failed for this file: check that its columns hold "
                           "the same kinds of values as the training data.") from None
    summary = prediction_summary(table)
    answer: dict[str, Any] = {
        "message": f"{summary['rows']:,} rows predicted with {model_title(algorithm)}.",
        "source": source,
        "model": algorithm,
        "file": file_name,
        **summary,
        "label_column": label if summary["matches"] else None,
        "download_url": "/api/v1/predictions",
        "download_name": f"{PurePath(file_name).stem or 'predictions'}.predictions.csv",
    }
    ws.predictions = BatchPredictions(prediction_csv(table).encode("utf-8"), answer)
    return answer
