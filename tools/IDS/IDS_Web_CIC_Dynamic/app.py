"""Dynamic IDS web trainer (Flask).

Since v5.0 (step 0.4g) the routes are a thin layer over ``toolkit.ids_core``: the loader
reads uploads, the split, leakage guard, training pipeline and evaluation run in
``train(...)``, and models are saved as signed bundles. Nothing here preprocesses data
itself, so the web trainer, a saved bundle and the CLI all use the same pipeline.

Every analysis that could steer training (balancing recommendation, feature scores,
balancing preview) looks at the **training split only**; the dataset itself is never
changed by them.
"""

import hashlib
import logging
import os
import re
import secrets
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from uuid import uuid4

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session
from werkzeug.utils import secure_filename

# The toolkit package lives at the repo root and isn't installed yet (v5.1 packaging).
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit import bundle_files, run_files  # noqa: E402
from toolkit.config import SplitConfig  # noqa: E402
from toolkit.ids_core import registry  # noqa: E402
from toolkit.ids_core.balancing import recommend_balancing  # noqa: E402
from toolkit.ids_core.bundle import BundleError, BundleVersionWarning  # noqa: E402
from toolkit.ids_core.datasets import DatasetError, load_dataset  # noqa: E402
from toolkit.ids_core.preview import preview_balancing, recommend_features as score_features  # noqa: E402
from toolkit.ids_core.runs import DatasetRef, TrainSettings  # noqa: E402
from toolkit.ids_core.runs import train as train_run  # noqa: E402
from toolkit.ids_core.split import split_dataset  # noqa: E402
from toolkit.logging_setup import setup_logging  # noqa: E402

# ---------------------------
# Logging setup
# ---------------------------
# Configured in run_server() (log file instance/logs/web.log), never on import.
# Library warnings are deliberately NOT silenced (B19): they flag problems
# such as a model saved with a different scikit-learn version.

# ---------------------------
# Flask app creation
# ---------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# Upload size limit (S9). Large enough for the full CIC-IDS2017 file (~1 GB);
# change with the TOOLKIT_MAX_UPLOAD_MB environment variable.
MAX_UPLOAD_MB = int(os.environ.get("TOOLKIT_MAX_UPLOAD_MB", "2048"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


class BadRequestBody(Exception):
    """The request body or one of its values is invalid (answered with HTTP 400)."""


def json_body():
    """Return the request's JSON object, or raise BadRequestBody (S9, S10 interim).

    Requiring a JSON object (Content-Type: application/json) also means a plain
    HTML form on another website can't trigger these routes.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequestBody("Request body must be a JSON object (Content-Type: application/json).")
    return data


def int_param(data, key, default, low, high):
    """Read an integer value from a JSON body, clamped to [low, high]."""
    value = data.get(key, default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise BadRequestBody(f"'{key}' must be a whole number.") from None
    return max(low, min(high, number))


def test_size_param(data):
    """Share of rows held out for the test split (clamped to 10-50 %)."""
    value = data.get("test_size", SplitConfig().test_size)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BadRequestBody("'test_size' must be a number between 0.1 and 0.5.") from None
    return max(0.1, min(0.5, number))


@app.errorhandler(BadRequestBody)
def _bad_request_body(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(413)
def _upload_too_large(_error):
    return jsonify({"error": f"Upload too large (limit: {MAX_UPLOAD_MB} MB)."}), 413


CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",                  # no inline scripts, no eval (S5 second layer)
    "style-src 'self' 'unsafe-inline'",   # inline style="" is still used; tightened in v5.1
    "img-src 'self' data:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


@app.after_request
def _security_headers(response):
    """Security headers on every response (S5, S10)."""
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.before_request
def _block_cross_site_requests():
    """Interim CSRF protection (S10) until the v5.1 token: reject state-changing
    requests whose Origin header names another site."""
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        origin = request.headers.get("Origin")
        if origin is not None and origin.rstrip("/") != request.host_url.rstrip("/"):
            return jsonify({"error": "Cross-site request blocked."}), 403
    return None

# ---------------------------
# Workspaces: one per browser session (S7, R4)
# ---------------------------
# The session cookie only holds a random workspace ID; the data stays in this
# process's memory. The signing key is random per process unless
# TOOLKIT_SECRET_KEY is set (workspaces don't survive a restart anyway).
app.secret_key = os.environ.get("TOOLKIT_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    # Only behind HTTPS: browsers drop Secure cookies on plain http://127.0.0.1.
    SESSION_COOKIE_SECURE=os.environ.get("TOOLKIT_HTTPS") == "1",
)

# Workspaces unused for this long are dropped to free memory (not while training).
WORKSPACE_IDLE_SECONDS = int(os.environ.get("TOOLKIT_WORKSPACE_IDLE_MIN", "240")) * 60


@dataclass
class UploadedData:
    """The uploaded dataset as the v5 loader read it."""

    frame: pd.DataFrame
    profile: object  # ids_core.profiles.ProfileMatch
    source: DatasetRef
    files: list
    test_frame: pd.DataFrame = None  # official split: the separate test file
    test_source: DatasetRef = None
    sources: pd.Series = None  # file name per row when several files were combined


class Workspace:
    """Everything one browser session has uploaded and trained."""

    def __init__(self):
        self.data = None    # UploadedData
        self.split = None   # (key, SplitResult): last training split used by the previews
        self.run = None     # TrainingRun of the last /configure
        self.export = None  # [SavedBundle, download name] once the run was exported
        self.loaded = None  # Bundle loaded with /load_model
        self.training = threading.Lock()  # held while /configure trains (one at a time)
        self.last_used = time.monotonic()


_workspaces = {}
_workspaces_lock = threading.Lock()


def _drop_idle_workspaces(now):
    for ws_id, ws in list(_workspaces.items()):
        if now - ws.last_used > WORKSPACE_IDLE_SECONDS and not ws.training.locked():
            del _workspaces[ws_id]
            logging.info("Dropped idle workspace")


def current_workspace():
    """The calling browser session's workspace, created on first use."""
    now = time.monotonic()
    with _workspaces_lock:
        _drop_idle_workspaces(now)
        ws = _workspaces.get(session.get("ws"))
        if ws is None:
            ws_id = secrets.token_urlsafe(16)
            ws = _workspaces[ws_id] = Workspace()
            session["ws"] = ws_id
        ws.last_used = now
        return ws


@app.route("/workspace", methods=["DELETE"])
def clear_workspace():
    """Forget this session's dataset and models; the next request starts a new workspace."""
    with _workspaces_lock:
        _workspaces.pop(session.pop("ws", None), None)
    return jsonify({"message": "Workspace cleared."})

# Uploads are written here only while the loader reads them (S11).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
uploads_dir = os.path.join(BASE_DIR, "uploads")
os.makedirs(uploads_dir, exist_ok=True)

_EXPORT_NAME = re.compile(r"[A-Za-z0-9_-]{1,80}")


def safe_upload_name(original):
    """Unique, sanitised file name for an upload (no path parts, no '..'), S2/S3."""
    cleaned = secure_filename(original or "") or "upload"
    return f"{uuid4().hex}_{cleaned}"


def safe_model_basename(name):
    """Return the export base name if it only uses letters, digits, '_' and '-' (1-80 chars).

    Raises ValueError otherwise, so a download name can never carry a path (S4).
    """
    if not isinstance(name, str) or not _EXPORT_NAME.fullmatch(name):
        raise ValueError("Model name may only use letters, digits, '_' and '-' (max 80).")
    return name


def display_name(original):
    """The uploaded file's own name for messages and the run record: no folder parts
    (from either OS), no control characters. Only safe_upload_name() names files on disk."""
    name = PureWindowsPath(original or "").name  # splits on both '\' and '/'
    name = "".join(ch for ch in name if ch.isprintable()).strip()[:120]
    return name or "upload"


def _error(message, status=400, **extra):
    return jsonify({"error": message, **extra}), status


def _labels(frame, target):
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

# ---------------------------
# Dataset upload and exploration
# ---------------------------

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


@app.route("/upload", methods=["POST"])
def upload():
    ws = current_workspace()
    if "dataset" not in request.files:
        return _error("No files uploaded.")
    files = [f for f in request.files.getlist("dataset") if f.filename]
    if not files:
        return _error("No files selected.")

    loaded = []
    for file in files:
        name = display_name(file.filename)
        if not file.filename.lower().endswith((".csv", ".txt")):
            return _error(f"Unsupported file type: {name} (use .csv or .txt).")
        # Unique, sanitised name inside uploads/ (S2, S3). The file is deleted
        # as soon as it has been read into memory (S11).
        save_path = Path(uploads_dir) / safe_upload_name(file.filename)
        if save_path.resolve().parent != Path(uploads_dir).resolve():
            return _error("Invalid file name.")
        try:
            file.save(save_path)
            loaded.append((name, load_dataset(save_path)))
        except (DatasetError, ValueError, UnicodeError, OSError):
            logging.exception("Could not read an uploaded file")
            return _error(f"Could not read {name} as a CSV/TXT dataset. Check the file format.")
        finally:
            save_path.unlink(missing_ok=True)

    try:
        data = _combine(loaded)
    except ValueError:
        logging.exception("Uploaded files don't fit together")
        return _error("The files can't be combined: they must be parts of the same dataset with "
                      "identical columns (or an official train file plus a file named '*test*').")
    ws.data, ws.split = data, None
    message = f"Successfully processed {len(files)} file(s). Total rows: {len(data.frame)}."
    if data.test_source:
        message += (f" {data.test_source.name} is used as the official test file "
                    f"({len(data.test_frame)} rows).")
    return jsonify({"columns": [str(c) for c in data.frame.columns], "message": message,
                    "dataset": _dataset_summary(data)})


def _require_data(ws):
    if ws.data is None:
        raise BadRequestBody("No dataset uploaded.")
    return ws.data


def _require_target(data, target):
    if not target or target not in data.frame.columns:
        raise BadRequestBody("Target column not specified or not found.")
    return target


@app.route("/preview_data", methods=["POST"])
def preview_data():
    """Return a preview of the uploaded dataset (first N rows)."""
    ws = current_workspace()
    req = json_body()
    num_rows = int_param(req, "num_rows", 10, 1, 100)  # clamped (S9)
    df = _require_data(ws).frame
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


@app.route("/filter_target_values", methods=["POST"])
def filter_target_values():
    """Remove the rows of the selected classes (from the test file too)."""
    ws = current_workspace()
    req = json_body()
    data = _require_data(ws)
    target = _require_target(data, req.get("target_column"))
    to_remove = req.get("values_to_remove") or []
    if not isinstance(to_remove, list) or not to_remove:
        return _error("No values selected for removal.")
    to_remove = [str(v) for v in to_remove]
    current = set(_labels(data.frame, target))
    unknown = [v for v in to_remove if v not in current]
    if unknown:
        return _error(f"Values not found in target: {', '.join(unknown)}")
    remaining = sorted(current - set(to_remove))
    if not remaining:
        return _error("Cannot remove all target values. At least 1 class must remain.")

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
        "new_class_counts": {k: int(v) for k, v in _labels(data.frame, target).value_counts().items()},
        "message": f"Successfully removed {len(to_remove)} class(es) ({removed:,} rows). "
                   f"{len(remaining)} classes remaining with {len(data.frame):,} total rows.",
    })


@app.route("/get_target_values", methods=["POST"])
def get_target_values():
    """Values of the target column with their counts."""
    ws = current_workspace()
    req = json_body()
    data = _require_data(ws)
    target = _require_target(data, req.get("target_column"))
    counts = {k: int(v) for k, v in _labels(data.frame, target).value_counts().items()}
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


@app.route("/analyze_class_distribution", methods=["POST"])
def analyze_class_distribution():
    """Class counts of the dataset; the balancing recommendation uses the training split."""
    ws = current_workspace()
    req = json_body()
    data = _require_data(ws)
    target = _require_target(data, req.get("target"))
    counts = _labels(data.frame, target).value_counts()
    total = int(counts.sum())
    percentages = {k: float(v) / total * 100 for k, v in counts.items()}
    try:
        split = _training_split(ws, target, test_size_param(req))
        recommendation = recommend_balancing(split.train[target],
                                             len(_candidate_features(data, target)))
    except ValueError:
        logging.exception("Could not split the dataset for the recommendation")
        return _error("Could not make a training split with this target (is there more than "
                      "one class?).")
    return jsonify({
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_percentages": percentages,
        "sorted_classes": sorted(percentages.items(), key=lambda item: -item[1]),
        "total_samples": total,
        "test_file_counts": ({k: int(v) for k, v in
                              _labels(data.test_frame, target).value_counts().items()}
                             if data.test_frame is not None and target in data.test_frame
                             else None),
        "recommendation": asdict(recommendation),
        "split": _split_summary(split),
    })


@app.route("/apply_smote", methods=["POST"])
def apply_smote_endpoint():
    """Preview a balancing method on the **training split** (nothing is stored or changed).

    Training applies the method inside each pipeline, to training rows only; this shows
    the class counts it would produce.
    """
    ws = current_workspace()
    req = json_body()
    data = _require_data(ws)
    target = _require_target(data, req.get("target"))
    method = req.get("balancing_method") or "smote"
    if method != "recommended" and method not in registry.BALANCERS:
        return _error(f"Unknown balancing method: {method}")
    features = _candidate_features(data, target, req.get("input_features"))
    try:
        split = _training_split(ws, target, test_size_param(req))
        preview = preview_balancing(split.train, target, method, features=features)
    except ValueError:
        logging.exception("Balancing preview failed")
        return _error("The balancing preview could not run on this data.")
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


@app.route("/recommend_features", methods=["POST"])
def recommend_features():
    """Feature scores on the training split, after the leakage guard."""
    ws = current_workspace()
    req = json_body()
    data = _require_data(ws)
    target = _require_target(data, req.get("target"))
    k_val = int_param(req, "k", 10, 1, 10_000)  # clamped (S9)
    try:
        split = _training_split(ws, target, test_size_param(req))
        result = score_features(split.train, target,
                                profile_blocked=data.profile.blocked_columns)
    except ValueError:
        logging.exception("Feature scoring failed")
        return _error("Feature scores could not be computed for this dataset.")
    return jsonify({
        "recommended_features": [[name, score] for name, score in result.scores[:k_val]],
        "all_feature_scores": dict(result.scores),
        "blocked": [{"column": c, "reason": r} for c, r in result.blocked],
        "method": result.method,
        "rows": result.rows,
        "train_rows": result.train_rows,
    })

# ---------------------------
# Training
# ---------------------------

@app.route("/get_available_models", methods=["GET"])
def get_available_models():
    """Models that can be trained here (optional libraries installed)."""
    return jsonify({"models": [
        {"id": name, "name": registry.MODELS[name].title,
         "description": registry.MODELS[name].description,
         "default": registry.MODELS[name].default}
        for name in registry.available_models()
    ]})


@app.route("/configure", methods=["POST"])
def configure():
    ws = current_workspace()
    if not ws.training.acquire(blocking=False):
        return jsonify({"error": "A training run is already in progress in this workspace. "
                                 "Wait for it to finish."}), 409
    try:
        return _configure_training(ws)
    finally:
        ws.training.release()


def _training_settings(data, req):
    """TrainSettings from the request, checked here so errors read well (S8)."""
    target = req.get("target")
    features = req.get("input_features")
    models = req.get("selected_models") or list(registry.default_models())
    balancer = req.get("balancing_method") or "none"
    if not target or target not in data.frame.columns:
        raise BadRequestBody("Choose the target column.")
    if not isinstance(features, list) or not features:
        raise BadRequestBody("Choose at least one input feature.")
    missing = [str(c) for c in features if c not in data.frame.columns]
    if missing:
        raise BadRequestBody(f"Unknown input feature(s): {', '.join(missing)}")
    if target in features:
        raise BadRequestBody("The target can't also be an input feature.")
    if not isinstance(models, list) or not models:
        raise BadRequestBody("Choose at least one model.")
    unknown = [str(m) for m in models if m not in registry.available_models()]
    if unknown:
        raise BadRequestBody(f"Unknown or unavailable model(s): {', '.join(unknown)}")
    if balancer != "recommended" and balancer not in registry.BALANCERS:
        raise BadRequestBody(f"Unknown balancing method: {balancer}")
    if _labels(data.frame, target).nunique() < 2:
        raise BadRequestBody("The target needs at least two classes.")
    return TrainSettings(label=target, features=tuple(features), models=tuple(models),
                         balancer=balancer, split=SplitConfig(test_size=test_size_param(req)))


def _features_info(fitted):
    return [{"name": f.name, "type": f.kind, "classes": list(f.categories)}
            for f in fitted.feature_schema()]


def _blocked(leakage_record):
    reasons = {}
    for finding in leakage_record["findings"]:
        if finding["action"] == "block":
            reasons.setdefault(finding["column"], finding["reason"])
    return [{"column": c, "reason": reasons.get(c, "BLOCKED")} for c in leakage_record["blocked"]]


def _configure_training(ws):
    req = json_body()
    data = _require_data(ws)
    settings = _training_settings(data, req)
    ws.split = None  # free the preview split: training makes its own
    try:
        run = train_run(data.frame, profile=data.profile, settings=settings,
                        test_frame=data.test_frame, source=data.source,
                        test_source=data.test_source, sources=data.sources)
    except ValueError:
        logging.exception("Training failed")
        return _error("Training could not run with these settings (for example: every chosen "
                      "feature was blocked by the leakage guard, or a class is too small for "
                      "the split). Details are in the server log.")
    evaluation = run.evaluation
    if not evaluation.training.models:
        return _error("No model could be trained. Details are in the server log.",
                      failures=[asdict(f) for f in evaluation.failures])
    ws.run, ws.export = run, None

    record = run.to_dict()
    result = record["evaluation"]
    selected = evaluation.training.models[evaluation.selected]
    title = registry.MODELS[evaluation.selected].title
    return jsonify({
        "message": f"Trained {len(evaluation.training.models)} model(s). Selected: {title} "
                   f"(best mean macro-F1 over {evaluation.cv_folds} CV folds of the training "
                   "split); the test split was scored once.",
        "run_id": run.run_id,
        "selected": evaluation.selected,
        "selection_metric": evaluation.selection_metric,
        "leaderboard": result["leaderboard"],
        "split": record["split"],
        "cv": {"folds": evaluation.cv_folds, "rows": evaluation.cv_rows,
               "sampled": evaluation.cv_sampled},
        "leakage": {"blocked": _blocked(record["leakage"]),
                    "warned": list(record["leakage"]["warned"]),
                    "features": len(record["features"])},
        "balancing": record["balancing"],
        "failures": result["failures"],
        "skipped": result["skipped"],
        "classes_only_in_test": result["classes_only_in_test"],
        "confusion_matrices": {m["name"]: {"labels": m["test"]["labels"],
                                           "matrix": m["test"]["confusion"]}
                               for m in result["models"]},
        "features_info": _features_info(selected),
        "notes": result["notes"],
    })


@app.route("/get_confusion_matrix", methods=["POST"])
def get_confusion_matrix():
    """Test-split confusion matrix of one trained model."""
    ws = current_workspace()
    req = json_body()
    if ws.run is None:
        return _error("No models trained yet.")
    name = req.get("model_name") or ws.run.evaluation.selected
    for model in ws.run.evaluation.models:
        if model.name == name:
            return jsonify({"confusion_matrix": [list(r) for r in model.test.confusion],
                            "class_labels": list(model.test.labels), "model_name": name,
                            "dataset_type": "test"})
    return _error(f"Model '{name}' not found.")

# ---------------------------
# Predicting
# ---------------------------

def _predict(models, default, req):
    """Predict one row with ``models[algorithm]`` through its saved pipeline (B2)."""
    values = req.get("values")
    if not isinstance(values, dict) or not values:
        return _error("No input values provided.")
    algorithm = req.get("algorithm") or default
    if algorithm not in models:
        return _error(f"Invalid algorithm selection. Available models: {', '.join(models)}")
    fitted = models[algorithm]
    try:
        row = fitted.input_frame(values)
    except ValueError as error:
        message = str(error)  # written by FittedModel.input_frame for the user (no internals)
        return _error(message)
    try:
        prediction = fitted.predict(row)[0]
    except Exception:
        logging.exception("Prediction error")
        return _error("Prediction failed for these input values.", 500)
    return jsonify({"prediction": str(prediction), "model": algorithm})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ws = current_workspace()
    req = json_body()
    if ws.run is None:
        return _error("Models not trained yet. Train a model first.")
    return _predict(ws.run.evaluation.training.models, ws.run.evaluation.selected, req)


@app.route("/predict_loaded", methods=["POST"])
def predict_loaded_endpoint():
    ws = current_workspace()
    req = json_body()
    if ws.loaded is None:
        return _error("No model loaded. Load a bundle exported by this toolkit first.")
    return _predict(ws.loaded.models, ws.loaded.selected, req)

# ---------------------------
# Signed bundles: export and load (D11, S1, S4)
# ---------------------------

def _bundle_refused(error):
    """BundleError messages are written for users and never contain paths (tested)."""
    return _error(str(error), code=error.code)


@app.route("/export_model", methods=["POST"])
def export_model():
    """Save the session's run as a signed bundle in instance/bundles/ (never overwritten)
    and its TrainingRun in instance/runs/; the browser downloads it from /download_bundle."""
    ws = current_workspace()
    req = json_body()
    try:
        name = safe_model_basename(req.get("model_name"))
    except ValueError:
        return _error("Model name may only use letters, digits, '_' and '-' (max 80).")
    if ws.run is None:
        return _error("Train a model first, then export it.")
    if ws.export is None:
        try:
            saved = bundle_files.save_bundle(ws.run)
            run_files.write_run(saved.run)
        except BundleError as error:
            return _bundle_refused(error)
        except FileExistsError:
            return _error("A bundle for this run already exists; it was not overwritten.", 409)
        ws.export = [saved, name]
    ws.export[1] = name
    saved = ws.export[0]
    return jsonify({
        "message": "Saved as a signed bundle. Download it below; it loads only on this "
                   "computer's toolkit install (keep a backup of the instance/ folder).",
        "file": saved.path.name,
        "run_id": saved.run.run_id,
        "sha256": saved.sha256,
        "download_url": "/download_bundle",
        "download_name": f"{name}.bundle",
    })


@app.route("/download_bundle", methods=["GET"])
def download_bundle():
    ws = current_workspace()
    if ws.export is None:
        return _error("Export the model first.", 404)
    saved, name = ws.export
    return send_file(saved.path, as_attachment=True, download_name=f"{name}.bundle",
                     mimetype="application/octet-stream")


@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Load a bundle exported by this install. Every check (format, signature, versions)
    runs before anything in the file is deserialized (S1); only signed bundles load."""
    ws = current_workspace()
    file = request.files.get("model_file")
    if file is None or not file.filename:
        return _error("Choose a .bundle file exported by this toolkit.")
    data = file.read()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BundleVersionWarning)  # listed in the reply instead
            bundle = bundle_files.open_bundle(data)
    except BundleError as error:
        return _bundle_refused(error)
    ws.loaded = bundle
    run = bundle.run
    rows = {row["model"]: row for row in run["evaluation"]["leaderboard"]}
    return jsonify({
        "message": f"Loaded {len(bundle.models)} model(s) from run {bundle.run_id}.",
        "run_id": bundle.run_id,
        "created_at": run["created_at"],
        "dataset": (run.get("dataset") or {}).get("name"),
        "selected": bundle.selected,
        "models": [{"id": name, "name": registry.MODELS[name].title if name in registry.MODELS
                    else name,
                    "cv_macro_f1": rows.get(name, {}).get("cv_macro_f1_mean"),
                    "test_macro_f1": rows.get(name, {}).get("test_macro_f1")}
                   for name in bundle.models],
        "features_info": _features_info(bundle.model()),
        "warnings": list(bundle.warnings),
    })

# ---------------------------
# Pages
# ---------------------------
@app.route("/")
def homepage():
    return render_template("home.html")

@app.route("/dynamic_ids")
def dynamic_ids_page():
    return render_template("dynamic_ids.html", idle_minutes=WORKSPACE_IDLE_SECONDS // 60)


# ---------------------------
# Main entry point
# ---------------------------
def run_server():
    """Serve the app on this computer only (127.0.0.1), S6.

    - Default: waitress, a production-grade pure-Python server (works on every OS).
    - TOOLKIT_DEBUG=1: Flask's debug server, for development only. Never use it
      when other people can reach the machine: its debugger can run code.
    - TOOLKIT_PORT: port number (default 5000).
    """
    setup_logging("web", console_level=logging.INFO)
    host = "127.0.0.1"
    port = int(os.environ.get("TOOLKIT_PORT", "5000"))
    print(f"IDS web app running at http://{host}:{port}/  (press Ctrl+C to stop)")
    if os.environ.get("TOOLKIT_DEBUG") == "1":
        print("WARNING: debug mode is ON (TOOLKIT_DEBUG=1). Development use only.")
        app.run(host=host, port=port, debug=True)
    else:
        from waitress import serve
        serve(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
