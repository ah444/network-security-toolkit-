"""Signed bundles (D11, S1, S4): export, download and load back; predictions with the
session's trained or loaded models, one row or a whole CSV file (step 1.4f)."""

from __future__ import annotations

import io
import logging
import warnings
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import pandas as pd
from flask import Blueprint, jsonify, request, send_file

from .. import run_files
from ..ids_core import registry
from ..ids_core.batch import (missing_features, prediction_csv, prediction_summary,
                              prediction_table)
from ..ids_core.bundle import Bundle, BundleError, BundleVersionWarning
from ..ids_core.datasets import DatasetError, load_dataset
from ..ids_core.pipeline import FittedModel
from ..ids_core.runs import TrainingRun
from .api import (NO_MODEL_LOADED, NOT_FOUND, NOT_TRAINED, TOO_LARGE, UNKNOWN_COLUMN,
                  UNREADABLE_FILE, Answer, error_response, json_body)
from .schemas import ExportRequest, PredictRequest, load
from .security import display_name, safe_model_basename, safe_upload_name
from .state import state
from .training import features_info
from .workspace import BatchPredictions, ExportedBundle, Workspace, with_workspace

log = logging.getLogger(__name__)

bp = Blueprint("bundles", __name__, url_prefix="/api/v1")


def _predict(models: Mapping[str, FittedModel], default: str | None,
             req: PredictRequest) -> Answer:
    """Predict one row with ``models[model]`` through its saved pipeline (B2)."""
    values = req.values
    if not values:
        return error_response("No input values provided.")
    algorithm = req.model or default or ""
    if algorithm not in models:
        return error_response(f"Invalid algorithm selection. Available models: {', '.join(models)}")
    fitted = models[algorithm]
    try:
        row = fitted.input_frame(values)
    except ValueError as error:
        message = str(error)  # written by FittedModel.input_frame for the user (no internals)
        return error_response(message)
    try:
        prediction = fitted.predict(row)[0]
    except Exception:
        log.exception("Prediction error")
        return error_response("Prediction failed for these input values.", 500)
    return jsonify({"prediction": str(prediction), "model": algorithm})


@bp.route("/run/predict", methods=["POST"])
@with_workspace()
def predict(ws: Workspace) -> Answer:
    req = load(PredictRequest, json_body())
    if ws.run is None:
        return error_response("Models not trained yet. Train a model first.", code=NOT_TRAINED)
    return _predict(ws.run.evaluation.training.models, ws.run.evaluation.selected, req)


@bp.route("/bundle/predict", methods=["POST"])
@with_workspace()
def predict_loaded(ws: Workspace) -> Answer:
    req = load(PredictRequest, json_body())
    if ws.loaded is None:
        return error_response("No model loaded. Load a bundle exported by this toolkit first.",
                              code=NO_MODEL_LOADED)
    return _predict(ws.loaded.models, ws.loaded.selected, req)


# --- Batch prediction: a whole CSV file (plan F1, step 1.4f) ---------------------------------

def _model_title(name: str) -> str:
    return registry.MODELS[name].title if name in registry.MODELS else name


def _too_large() -> Answer:
    return error_response(f"File too large to predict (limit: {state().config.max_predict_mb} "
                          "MB, TOOLKIT_MAX_PREDICT_MB).", 413, TOO_LARGE)


def _read_prediction_file() -> pd.DataFrame | Answer:
    """The uploaded ``file`` as a table, read like a dataset upload: a unique sanitised
    name in uploads/, deleted as soon as it is read (S2, S3, S11); or an error answer."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return error_response("Choose a .csv or .txt file to predict.")
    name = display_name(file.filename)
    if not file.filename.lower().endswith((".csv", ".txt")):
        return error_response(f"Unsupported file type: {name} (use .csv or .txt).")
    save_path = state().upload_dir / safe_upload_name(file.filename)
    if save_path.resolve().parent != state().upload_dir.resolve():
        return error_response("Invalid file name.")
    try:
        file.save(save_path)
        if save_path.stat().st_size > state().config.max_predict_mb * 2**20:
            return _too_large()
        return load_dataset(save_path).frame
    except (DatasetError, ValueError, UnicodeError, OSError):
        log.exception("Could not read a file to predict")
        return error_response(f"Could not read {name} as a CSV/TXT file. Check the file "
                              "format.", code=UNREADABLE_FILE)
    finally:
        save_path.unlink(missing_ok=True)


def _predict_file(ws: Workspace, source: str, models: Mapping[str, FittedModel],
                  default: str | None, label: str | None) -> Answer:
    """Label every row of the uploaded file; keeps the labelled CSV for GET /predictions."""
    if (request.content_length or 0) > state().config.max_predict_mb * 2**20:
        return _too_large()  # before request.form reads the body
    algorithm = request.form.get("model") or default or ""
    if algorithm not in models:
        return error_response(f"Invalid algorithm selection. Available models: {', '.join(models)}")
    fitted = models[algorithm]
    frame = _read_prediction_file()
    if not isinstance(frame, pd.DataFrame):
        return frame
    missing = missing_features(fitted, frame)
    if missing:
        shown = ", ".join(missing[:20]) + (f" and {len(missing) - 20} more" if len(missing) > 20
                                           else "")
        return error_response(f"The file is missing feature column(s) of this model: {shown}.",
                              code=UNKNOWN_COLUMN)
    try:
        table = prediction_table(fitted, frame, label)
    except Exception:
        log.exception("Batch prediction failed")
        return error_response("Prediction failed for this file: check that its columns hold "
                              "the same kinds of values as the training data.")
    summary = prediction_summary(table)
    file_name = display_name(request.files["file"].filename or "")
    answer: dict[str, Any] = {
        "message": f"{summary['rows']:,} rows predicted with {_model_title(algorithm)}.",
        "source": source,
        "model": algorithm,
        "file": file_name,
        **summary,
        "label_column": label if summary["matches"] else None,
        "download_url": "/api/v1/predictions",
        "download_name": f"{PurePath(file_name).stem or 'predictions'}.predictions.csv",
    }
    ws.predictions = BatchPredictions(prediction_csv(table).encode("utf-8"), answer)
    return jsonify(answer)


def _run_label(run: TrainingRun) -> str | None:
    return run.settings.label or (run.profile.label_column if run.profile else None)


def _record_label(record: Mapping[str, Any]) -> str | None:
    label = record["settings"].get("label") or (record.get("profile") or {}).get("label_column")
    return str(label) if label else None


@bp.route("/run/predict-file", methods=["POST"])
@with_workspace()
def predict_file(ws: Workspace) -> Answer:
    if ws.run is None:
        return error_response("Models not trained yet. Train a model first.", code=NOT_TRAINED)
    return _predict_file(ws, "run", ws.run.evaluation.training.models,
                         ws.run.evaluation.selected, _run_label(ws.run))


@bp.route("/bundle/predict-file", methods=["POST"])
@with_workspace()
def predict_file_loaded(ws: Workspace) -> Answer:
    if ws.loaded is None:
        return error_response("No model loaded. Load a bundle exported by this toolkit first.",
                              code=NO_MODEL_LOADED)
    return _predict_file(ws, "bundle", ws.loaded.models, ws.loaded.selected,
                         _record_label(ws.loaded.run))


@bp.route("/predictions", methods=["GET"])
@with_workspace()
def download_predictions(ws: Workspace) -> Answer:
    if ws.predictions is None:
        return error_response("Predict a CSV file first.", 404, NOT_FOUND)
    return send_file(io.BytesIO(ws.predictions.csv), as_attachment=True,
                     download_name=ws.predictions.answer["download_name"],
                     mimetype="text/csv")


def _bundle_refused(error: BundleError) -> Answer:
    """BundleError messages are written for users and never contain paths (tested)."""
    return error_response(str(error), code=error.code)


@bp.route("/run/export", methods=["POST"])
@with_workspace()
def export(ws: Workspace) -> Answer:
    """Save the session's run as a signed bundle in instance/bundles/ (never overwritten)
    and its TrainingRun in instance/runs/; the browser downloads it from /api/v1/run/bundle."""
    req = load(ExportRequest, json_body())
    try:
        name = safe_model_basename(req.name)
    except ValueError:
        return error_response("Model name may only use letters, digits, '_' and '-' (max 80).")
    if ws.run is None:
        return error_response("Train a model first, then export it.", code=NOT_TRAINED)
    if ws.export is None:
        try:
            saved = state().model_store.save(ws.run)
            run_files.write_run(saved.run)
        except BundleError as error:
            return _bundle_refused(error)
        except FileExistsError:
            return error_response("A bundle for this run already exists; it was not overwritten.",
                                  409)
        ws.export = ExportedBundle(saved, name)
    ws.export.name = name
    return jsonify(export_answer(ws.export))


def export_answer(export: ExportedBundle) -> dict[str, Any]:
    """The export's answer; GET /api/v1/workspace repeats it after a reload (step 1.3c)."""
    saved = export.saved
    return {
        "message": "Saved as a signed bundle. Download it below; it loads only on this "
                   "computer's toolkit install (keep a backup of the instance/ folder).",
        "file": saved.path.name,
        "run_id": saved.run.run_id,
        "sha256": saved.sha256,
        "download_url": "/api/v1/run/bundle",
        "download_name": f"{export.name}.bundle",
    }


@bp.route("/run/bundle", methods=["GET"])
@with_workspace()
def download(ws: Workspace) -> Answer:
    if ws.export is None:
        return error_response("Export the model first.", 404, NOT_TRAINED)
    return send_file(ws.export.saved.path, as_attachment=True,
                     download_name=f"{ws.export.name}.bundle",
                     mimetype="application/octet-stream")


@bp.route("/bundle", methods=["POST"])
@with_workspace(create=True)
def load_bundle(ws: Workspace) -> Answer:
    """Load a bundle exported by this install. Every check (format, signature, versions)
    runs before anything in the file is deserialized (S1); only signed bundles load."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return error_response("Choose a .bundle file exported by this toolkit.")
    data = file.read()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BundleVersionWarning)  # listed in the reply instead
            bundle = state().model_store.open(data)
    except BundleError as error:
        return _bundle_refused(error)
    ws.loaded = bundle
    return jsonify(bundle_answer(bundle))


def bundle_answer(bundle: Bundle) -> dict[str, Any]:
    """The bundle load's answer; GET /api/v1/workspace repeats it after a reload
    (step 1.3c)."""
    run = bundle.run
    rows = {row["model"]: row for row in run["evaluation"]["leaderboard"]}
    return {
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
        "features_info": features_info(bundle.model()),
        "warnings": list(bundle.warnings),
    }
