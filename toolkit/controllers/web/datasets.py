"""Dataset routes: upload, preview, class filter, and the analyses that steer training.

Every analysis that could steer training (balancing recommendation, feature scores,
balancing preview) looks at the **training split only**; the dataset itself is never
changed by them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from ...services.ids import registry
from ...services.ids.balancing import recommend_balancing
from ...services.ids.datasets import DatasetError
from ...services.ids.preview import preview_balancing
from ...services.ids.preview import recommend_features as score_features
from ...services.ids.split import SplitResult
from ...services.web.datasets import (
    candidate_features,
    combine,
    label_values,
    read_upload,
    remove_classes,
    require_data,
    require_target,
    training_split,
)
from ...services.web.errors import UNREADABLE_FILE
from .api import Answer, error_response, json_body, query
from .schemas import (
    BalancingPreviewRequest,
    ClassesRequest,
    DistributionRequest,
    FeatureScoresRequest,
    FilterRequest,
    PreviewRequest,
    load,
)
from .security import display_name, safe_upload_name
from .state import state
from ...services.web.workspace import UploadedData, Workspace, dataset_bytes
from .workspace import with_workspace

log = logging.getLogger(__name__)

bp = Blueprint("datasets", __name__, url_prefix="/api/v1")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as JSON-safe dicts: missing -> null, inf -> "inf" (JSON has no NaN/inf)."""
    values = frame.astype(object).where(frame.notna(), None)
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            infinite = np.isinf(frame[column].to_numpy())
            if infinite.any():
                values.loc[infinite, column] = np.where(frame[column][infinite] > 0, "inf", "-inf")
    records: list[dict[str, Any]] = values.to_dict(orient="records")
    return records


def _dataset_summary(data: UploadedData, size: int) -> dict[str, Any]:
    profile = data.profile
    return {
        "files": data.files,
        "rows": len(data.frame),
        "memory_mb": round(size / 2**20, 1),
        "test_file": data.test_source.name if data.test_source else None,
        "test_rows": len(data.test_frame) if data.test_frame is not None else None,
        "profile": {
            "name": profile.profile.name,
            "variant": profile.variant,
            "label_column": profile.label_column,
            "split_strategy": profile.split_strategy,
            "blocked_columns": list(profile.blocked_columns),
            "warnings": list(profile.warnings),
        },
    }


def dataset_answer(data: UploadedData, size: int) -> dict[str, Any]:
    """The upload's answer; GET /api/v1/workspace repeats it after a reload (step 1.3c)."""
    message = f"Successfully processed {len(data.files)} file(s). Total rows: {len(data.frame)}."
    if data.test_source and data.test_frame is not None:
        message += (f" {data.test_source.name} is used as the official test file "
                    f"({len(data.test_frame)} rows).")
    message += f" Memory used: {size / 2**20:,.1f} MB."
    return {"columns": [str(c) for c in data.frame.columns], "message": message,
            "dataset": _dataset_summary(data, size)}


@bp.route("/dataset", methods=["POST"])
@with_workspace(create=True, changes_dataset=True)
def upload(ws: Workspace) -> Answer:
    if "dataset" not in request.files:
        return error_response("No files uploaded.")
    files = [f for f in request.files.getlist("dataset") if f.filename]
    if not files:
        return error_response("No files selected.")

    loaded = []
    for file in files:
        name = display_name(file.filename)
        if not (file.filename or "").lower().endswith((".csv", ".txt")):
            return error_response(f"Unsupported file type: {name} (use .csv or .txt).")
        # Unique, sanitised name inside uploads/ (S2, S3). The file is deleted
        # as soon as it has been read into memory (S11).
        save_path = state().upload_dir / safe_upload_name(file.filename)
        if save_path.resolve().parent != state().upload_dir.resolve():
            return error_response("Invalid file name.")
        try:
            loaded.append((name, read_upload(file.stream, save_path)))
        except (DatasetError, ValueError, UnicodeError, OSError):
            log.exception("Could not read an uploaded file")
            return error_response(f"Could not read {name} as a CSV/TXT dataset. Check the file "
                                  "format.", code=UNREADABLE_FILE)

    try:
        data = combine(loaded)
    except ValueError:
        log.exception("Uploaded files don't fit together")
        return error_response("The files can't be combined: they must be parts of the same dataset "
                              "with identical columns (or an official train file plus a file "
                              "named '*test*').")
    size = dataset_bytes(data)
    state().workspaces.reserve_data(ws, size)  # may drop other idle workspaces, or refuse
    ws.data, ws.split = data, None
    return jsonify(dataset_answer(data, size))


@bp.route("/dataset/preview", methods=["GET"])
@with_workspace()
def preview(ws: Workspace) -> Answer:
    """Return a preview of the uploaded dataset (first N rows)."""
    req = load(PreviewRequest, query())  # rows clamped to 1-100 (S9)
    df = require_data(ws).frame
    return jsonify({
        "preview": _json_records(df.head(req.rows)),
        "columns": [str(c) for c in df.columns],
        "stats": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": int(sum(df[c].dtype.kind in "iufb" for c in df.columns)),
            "categorical_columns": int(sum(df[c].dtype.kind not in "iufb" for c in df.columns)),
        },
    })


@bp.route("/dataset/filter", methods=["POST"])
@with_workspace(changes_dataset=True)
def filter_classes(ws: Workspace) -> Answer:
    """Remove the rows of the selected classes (from the test file too)."""
    req = load(FilterRequest, json_body())
    data = require_data(ws)
    target = require_target(data, req.target)
    to_remove = req.remove
    removed, remaining = remove_classes(ws, data, target, to_remove)
    return jsonify({
        "success": True,
        "removed_values": to_remove,
        "removed_rows": removed,
        "remaining_values": remaining,
        "remaining_count": len(remaining),
        "total_rows": len(data.frame),
        "new_class_counts": {k: int(v) for k, v in
                             label_values(data.frame, target).value_counts().items()},
        "message": f"Successfully removed {len(to_remove)} class(es) ({removed:,} rows). "
                   f"{len(remaining)} classes remaining with {len(data.frame):,} total rows.",
    })


@bp.route("/dataset/classes", methods=["GET"])
@with_workspace()
def classes(ws: Workspace) -> Answer:
    """Values of the target column with their counts."""
    req = load(ClassesRequest, query())
    data = require_data(ws)
    target = require_target(data, req.target)
    counts = {k: int(v) for k, v in label_values(data.frame, target).value_counts().items()}
    return jsonify({"values": list(counts), "value_counts": counts, "count": len(counts)})


def _split_summary(split: SplitResult) -> dict[str, Any]:
    return {"strategy": split.strategy, "train_rows": len(split.train),
            "test_rows": len(split.test), "test_size": split.test_size,
            "warnings": list(split.warnings)}


@bp.route("/dataset/distribution", methods=["GET"])
@with_workspace()
def distribution(ws: Workspace) -> Answer:
    """Class counts of the dataset; the balancing recommendation uses the training split."""
    req = load(DistributionRequest, query())
    data = require_data(ws)
    target = require_target(data, req.target)
    counts = label_values(data.frame, target).value_counts()
    total = int(counts.sum())
    percentages = {k: float(v) / total * 100 for k, v in counts.items()}
    try:
        split = training_split(ws, target, req.test_size)
        recommendation = recommend_balancing(split.train[target],
                                             len(candidate_features(data, target)))
    except ValueError:
        log.exception("Could not split the dataset for the recommendation")
        return error_response("Could not make a training split with this target (is there more "
                              "than one class?).")
    return jsonify({
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_percentages": percentages,
        "sorted_classes": sorted(percentages.items(), key=lambda item: -item[1]),
        "total_samples": total,
        "test_file_counts": ({k: int(v) for k, v in
                              label_values(data.test_frame, target).value_counts().items()}
                             if data.test_frame is not None and target in data.test_frame
                             else None),
        "recommendation": asdict(recommendation),
        "split": _split_summary(split),
    })


@bp.route("/dataset/balancing-preview", methods=["POST"])
@with_workspace()
def balancing_preview(ws: Workspace) -> Answer:
    """Preview a balancing method on the **training split** (nothing is stored or changed).

    Training applies the method inside each pipeline, to training rows only; this shows
    the class counts it would produce.
    """
    req = load(BalancingPreviewRequest, json_body())
    data = require_data(ws)
    target = require_target(data, req.target)
    method = req.balancing or "smote"
    if method != "recommended" and method not in registry.BALANCERS:
        return error_response(f"Unknown balancing method: {method}")
    features = candidate_features(data, target, req.features)
    try:
        split = training_split(ws, target, req.test_size)
        preview = preview_balancing(split.train, target, method, features=features)
    except ValueError:
        log.exception("Balancing preview failed")
        return error_response("The balancing preview could not run on this data.")
    return jsonify({
        "message": "Preview on the training split only: your dataset and the test split are "
                   "unchanged. Training applies the method inside each model's pipeline.",
        "requested": preview.requested,
        "balancer": preview.balancer,
        "before": dict(preview.before),
        "after": dict(preview.after),
        "changes_rows": preview.changes_rows,
        "rows": preview.rows,
        "train_rows": preview.train_rows,
        "sampled": preview.sampled,
        "notes": list(preview.notes),
        "recommendation": asdict(preview.recommendation) if preview.recommendation else None,
    })


@bp.route("/dataset/feature-scores", methods=["POST"])
@with_workspace()
def feature_scores(ws: Workspace) -> Answer:
    """Feature scores on the training split, after the leakage guard."""
    req = load(FeatureScoresRequest, json_body())  # k clamped to 1-10,000 (S9)
    data = require_data(ws)
    target = require_target(data, req.target)
    try:
        split = training_split(ws, target, req.test_size)
        result = score_features(split.train, target,
                                profile_blocked=data.profile.blocked_columns)
    except ValueError:
        log.exception("Feature scoring failed")
        return error_response("Feature scores could not be computed for this dataset.")
    scores = [(name, "inf" if np.isinf(score) else score) for name, score in result.scores]
    return jsonify({  # JSON has no inf: a feature with one value per class scores "inf"
        "recommended_features": [[name, score] for name, score in scores[:req.k]],
        "all_feature_scores": dict(scores),
        "blocked": [{"column": c, "reason": r} for c, r in result.blocked],
        "method": result.method,
        "rows": result.rows,
        "train_rows": result.train_rows,
    })
