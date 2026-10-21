"""Dataset routes: upload, preview, class filter, and the analyses that steer training.

Every analysis that could steer training (balancing recommendation, feature scores,
balancing preview) looks at the **training split only**; the dataset itself is never
changed by them.
"""

import hashlib
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from ..config import SplitConfig
from ..ids_core import registry
from ..ids_core.balancing import recommend_balancing
from ..ids_core.datasets import DatasetError, load_dataset
from ..ids_core.preview import preview_balancing
from ..ids_core.preview import recommend_features as score_features
from ..ids_core.runs import DatasetRef
from ..ids_core.split import split_dataset
from .api import BadRequestBody, error_response, int_param, json_body, test_size_param
from .security import display_name, safe_upload_name
from .state import state
from .workspace import UploadedData, current_workspace

bp = Blueprint("datasets", __name__)


def label_values(frame, target):
    return frame[target].dropna().astype(str)


def _json_records(frame):
    """Rows as JSON-safe dicts: missing -> null, inf -> "inf" (JSON has no NaN/inf)."""
    values = frame.astype(object).where(frame.notna(), None)
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            infinite = np.isinf(frame[column].to_numpy())
            if infinite.any():
                values.loc[infinite, column] = np.where(frame[column][infinite] > 0, "inf", "-inf")
    return values.to_dict(orient="records")


def _combine(loaded):
    """One UploadedData from the loaded files: an official train/test pair (NSL-KDD with a
    file named '*test*'), or several files with the same columns stacked together."""
    names = [name for name, _ in loaded]
    first = loaded[0][1]
    for name, dataset in loaded[1:]:
        if dataset.profile.profile.name != first.profile.profile.name:
            raise ValueError(f"{name} looks like a different dataset than {names[0]}.")
        if list(dataset.frame.columns) != list(first.frame.columns):
            raise ValueError(f"Column mismatch in {name}. All files must have identical columns.")
    test_files = [i for i, name in enumerate(names) if "test" in name.lower()]
    if len(loaded) == 2 and first.profile.split_strategy == "official" and len(test_files) == 1:
        test_name, test = loaded[test_files[0]]
        train_name, train = loaded[1 - test_files[0]]
        return UploadedData(frame=train.frame, profile=train.profile, files=names,
                            source=DatasetRef.from_loaded(train_name, train),
                            test_frame=test.frame,
                            test_source=DatasetRef.from_loaded(test_name, test))
    if len(loaded) == 1:
        return UploadedData(frame=first.frame, profile=first.profile, files=names,
                            source=DatasetRef.from_loaded(names[0], first))
    frame = pd.concat([dataset.frame for _, dataset in loaded], ignore_index=True)
    combined = hashlib.sha256("\n".join(d.report.sha256 for _, d in loaded).encode()).hexdigest()
    sources = pd.Series(np.repeat(names, [len(d.frame) for _, d in loaded]), index=frame.index)
    return UploadedData(frame=frame, profile=first.profile, files=names, sources=sources,
                        source=DatasetRef(" + ".join(names), combined, len(frame),
                                          len(frame.columns)))


def _dataset_summary(data):
    profile = data.profile
    return {
        "files": data.files,
        "rows": len(data.frame),
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


@bp.route("/upload", methods=["POST"])
def upload():
    ws = current_workspace()
    if "dataset" not in request.files:
        return error_response("No files uploaded.")
    files = [f for f in request.files.getlist("dataset") if f.filename]
    if not files:
        return error_response("No files selected.")

    loaded = []
    for file in files:
        name = display_name(file.filename)
        if not file.filename.lower().endswith((".csv", ".txt")):
            return error_response(f"Unsupported file type: {name} (use .csv or .txt).")
        # Unique, sanitised name inside uploads/ (S2, S3). The file is deleted
        # as soon as it has been read into memory (S11).
        save_path = state().upload_dir / safe_upload_name(file.filename)
        if save_path.resolve().parent != state().upload_dir.resolve():
            return error_response("Invalid file name.")
        try:
            file.save(save_path)
            loaded.append((name, load_dataset(save_path)))
        except (DatasetError, ValueError, UnicodeError, OSError):
            logging.exception("Could not read an uploaded file")
            return error_response(f"Could not read {name} as a CSV/TXT dataset. Check the file format.")
        finally:
            save_path.unlink(missing_ok=True)

    try:
        data = _combine(loaded)
    except ValueError:
        logging.exception("Uploaded files don't fit together")
        return error_response("The files can't be combined: they must be parts of the same dataset with "
                      "identical columns (or an official train file plus a file named '*test*').")
    ws.data, ws.split = data, None
    message = f"Successfully processed {len(files)} file(s). Total rows: {len(data.frame)}."
    if data.test_source:
        message += (f" {data.test_source.name} is used as the official test file "
                    f"({len(data.test_frame)} rows).")
    return jsonify({"columns": [str(c) for c in data.frame.columns], "message": message,
                    "dataset": _dataset_summary(data)})


def require_data(ws):
    if ws.data is None:
        raise BadRequestBody("No dataset uploaded.")
    return ws.data


def _require_target(data, target):
    if not target or target not in data.frame.columns:
        raise BadRequestBody("Target column not specified or not found.")
    return target


@bp.route("/preview_data", methods=["POST"])
def preview_data():
    """Return a preview of the uploaded dataset (first N rows)."""
    ws = current_workspace()
    req = json_body()
    num_rows = int_param(req, "num_rows", 10, 1, 100)  # clamped (S9)
    df = require_data(ws).frame
    return jsonify({
        "preview": _json_records(df.head(num_rows)),
        "columns": [str(c) for c in df.columns],
        "stats": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": int(sum(df[c].dtype.kind in "iufb" for c in df.columns)),
            "categorical_columns": int(sum(df[c].dtype.kind not in "iufb" for c in df.columns)),
        },
    })


@bp.route("/filter_target_values", methods=["POST"])
def filter_target_values():
    """Remove the rows of the selected classes (from the test file too)."""
    ws = current_workspace()
    req = json_body()
    data = require_data(ws)
    target = _require_target(data, req.get("target_column"))
    to_remove = req.get("values_to_remove") or []
    if not isinstance(to_remove, list) or not to_remove:
        return error_response("No values selected for removal.")
    to_remove = [str(v) for v in to_remove]
    current = set(label_values(data.frame, target))
    unknown = [v for v in to_remove if v not in current]
    if unknown:
        return error_response(f"Values not found in target: {', '.join(unknown)}")
    remaining = sorted(current - set(to_remove))
    if not remaining:
        return error_response("Cannot remove all target values. At least 1 class must remain.")

    before = len(data.frame)
    keep = ~data.frame[target].astype(str).isin(to_remove)
    data.frame = data.frame[keep].reset_index(drop=True)
    if data.sources is not None:
        data.sources = data.sources[keep].reset_index(drop=True)
    if data.test_frame is not None and target in data.test_frame.columns:
        test_keep = ~data.test_frame[target].astype(str).isin(to_remove)
        data.test_frame = data.test_frame[test_keep].reset_index(drop=True)
    ws.split = None
    removed = before - len(data.frame)
    logging.info("Removed %d target values (%d rows)", len(to_remove), removed)
    return jsonify({
        "success": True,
        "removed_values": to_remove,
        "removed_rows": removed,
        "remaining_values": remaining,
        "remaining_count": len(remaining),
        "total_rows": len(data.frame),
        "new_class_counts": {k: int(v) for k, v in label_values(data.frame, target).value_counts().items()},
        "message": f"Successfully removed {len(to_remove)} class(es) ({removed:,} rows). "
                   f"{len(remaining)} classes remaining with {len(data.frame):,} total rows.",
    })


@bp.route("/get_target_values", methods=["POST"])
def get_target_values():
    """Values of the target column with their counts."""
    ws = current_workspace()
    req = json_body()
    data = require_data(ws)
    target = _require_target(data, req.get("target_column"))
    counts = {k: int(v) for k, v in label_values(data.frame, target).value_counts().items()}
    return jsonify({"values": list(counts), "value_counts": counts, "count": len(counts)})


def _training_split(ws, target, test_size):
    """The split training will use (same profile, strategy, seed), cached for the previews."""
    data = ws.data
    key = (id(data.frame), target, test_size)
    if ws.split is None or ws.split[0] != key:
        ws.split = None  # free the old split first (big datasets)
        split = split_dataset(data.frame, target, profile=data.profile,
                              test_frame=data.test_frame, sources=data.sources,
                              config=SplitConfig(test_size=test_size))
        ws.split = (key, split)
    return ws.split[1]


def _split_summary(split):
    return {"strategy": split.strategy, "train_rows": len(split.train),
            "test_rows": len(split.test), "test_size": split.test_size,
            "warnings": list(split.warnings)}


def _candidate_features(data, target, chosen=None):
    blocked = set(data.profile.blocked_columns)
    if chosen:
        if not isinstance(chosen, list) or any(c not in data.frame.columns for c in chosen):
            raise BadRequestBody("Unknown input feature.")
        return [c for c in chosen if c != target]
    return [c for c in data.frame.columns if c != target and c not in blocked]


@bp.route("/analyze_class_distribution", methods=["POST"])
def analyze_class_distribution():
    """Class counts of the dataset; the balancing recommendation uses the training split."""
    ws = current_workspace()
    req = json_body()
    data = require_data(ws)
    target = _require_target(data, req.get("target"))
    counts = label_values(data.frame, target).value_counts()
    total = int(counts.sum())
    percentages = {k: float(v) / total * 100 for k, v in counts.items()}
    try:
        split = _training_split(ws, target, test_size_param(req))
        recommendation = recommend_balancing(split.train[target],
                                             len(_candidate_features(data, target)))
    except ValueError:
        logging.exception("Could not split the dataset for the recommendation")
        return error_response("Could not make a training split with this target (is there more than "
                      "one class?).")
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


@bp.route("/apply_smote", methods=["POST"])
def apply_smote_endpoint():
    """Preview a balancing method on the **training split** (nothing is stored or changed).

    Training applies the method inside each pipeline, to training rows only; this shows
    the class counts it would produce.
    """
    ws = current_workspace()
    req = json_body()
    data = require_data(ws)
    target = _require_target(data, req.get("target"))
    method = req.get("balancing_method") or "smote"
    if method != "recommended" and method not in registry.BALANCERS:
        return error_response(f"Unknown balancing method: {method}")
    features = _candidate_features(data, target, req.get("input_features"))
    try:
        split = _training_split(ws, target, test_size_param(req))
        preview = preview_balancing(split.train, target, method, features=features)
    except ValueError:
        logging.exception("Balancing preview failed")
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


@bp.route("/recommend_features", methods=["POST"])
def recommend_features():
    """Feature scores on the training split, after the leakage guard."""
    ws = current_workspace()
    req = json_body()
    data = require_data(ws)
    target = _require_target(data, req.get("target"))
    k_val = int_param(req, "k", 10, 1, 10_000)  # clamped (S9)
    try:
        split = _training_split(ws, target, test_size_param(req))
        result = score_features(split.train, target,
                                profile_blocked=data.profile.blocked_columns)
    except ValueError:
        logging.exception("Feature scoring failed")
        return error_response("Feature scores could not be computed for this dataset.")
    return jsonify({
        "recommended_features": [[name, score] for name, score in result.scores[:k_val]],
        "all_feature_scores": dict(result.scores),
        "blocked": [{"column": c, "reason": r} for c, r in result.blocked],
        "method": result.method,
        "rows": result.rows,
        "train_rows": result.train_rows,
    })
