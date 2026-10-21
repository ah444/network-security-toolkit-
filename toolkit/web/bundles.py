"""Signed bundles (D11, S1, S4): export, download and load back; predictions with the
session's trained or loaded models."""

import logging
import warnings

from flask import Blueprint, jsonify, request, send_file

from .. import run_files
from ..ids_core import registry
from ..ids_core.bundle import BundleError, BundleVersionWarning
from .api import error_response, json_body
from .security import safe_model_basename
from .state import state
from .training import features_info
from .workspace import current_workspace

bp = Blueprint("bundles", __name__)


def _predict(models, default, req):
    """Predict one row with ``models[algorithm]`` through its saved pipeline (B2)."""
    values = req.get("values")
    if not isinstance(values, dict) or not values:
        return error_response("No input values provided.")
    algorithm = req.get("algorithm") or default
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
        logging.exception("Prediction error")
        return error_response("Prediction failed for these input values.", 500)
    return jsonify({"prediction": str(prediction), "model": algorithm})


@bp.route("/predict", methods=["POST"])
def predict_endpoint():
    ws = current_workspace()
    req = json_body()
    if ws.run is None:
        return error_response("Models not trained yet. Train a model first.")
    return _predict(ws.run.evaluation.training.models, ws.run.evaluation.selected, req)


@bp.route("/predict_loaded", methods=["POST"])
def predict_loaded_endpoint():
    ws = current_workspace()
    req = json_body()
    if ws.loaded is None:
        return error_response("No model loaded. Load a bundle exported by this toolkit first.")
    return _predict(ws.loaded.models, ws.loaded.selected, req)


def _bundle_refused(error):
    """BundleError messages are written for users and never contain paths (tested)."""
    return error_response(str(error), code=error.code)


@bp.route("/export_model", methods=["POST"])
def export_model():
    """Save the session's run as a signed bundle in instance/bundles/ (never overwritten)
    and its TrainingRun in instance/runs/; the browser downloads it from /download_bundle."""
    ws = current_workspace()
    req = json_body()
    try:
        name = safe_model_basename(req.get("model_name"))
    except ValueError:
        return error_response("Model name may only use letters, digits, '_' and '-' (max 80).")
    if ws.run is None:
        return error_response("Train a model first, then export it.")
    if ws.export is None:
        try:
            saved = state().model_store.save(ws.run)
            run_files.write_run(saved.run)
        except BundleError as error:
            return _bundle_refused(error)
        except FileExistsError:
            return error_response("A bundle for this run already exists; it was not overwritten.", 409)
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


@bp.route("/download_bundle", methods=["GET"])
def download_bundle():
    ws = current_workspace()
    if ws.export is None:
        return error_response("Export the model first.", 404)
    saved, name = ws.export
    return send_file(saved.path, as_attachment=True, download_name=f"{name}.bundle",
                     mimetype="application/octet-stream")


@bp.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Load a bundle exported by this install. Every check (format, signature, versions)
    runs before anything in the file is deserialized (S1); only signed bundles load."""
    ws = current_workspace()
    file = request.files.get("model_file")
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
        "features_info": features_info(bundle.model()),
        "warnings": list(bundle.warnings),
    })
