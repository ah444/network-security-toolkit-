"""Signed bundles (D11, S1, S4): export, download and load back; predictions with the
session's trained or loaded models."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import Any

from flask import Blueprint, jsonify, request, send_file

from .. import run_files
from ..ids_core import registry
from ..ids_core.bundle import Bundle, BundleError, BundleVersionWarning
from ..ids_core.pipeline import FittedModel
from .api import NO_MODEL_LOADED, NOT_TRAINED, Answer, error_response, json_body
from .schemas import ExportRequest, PredictRequest, load
from .security import safe_model_basename
from .state import state
from .training import features_info
from .workspace import ExportedBundle, Workspace, with_workspace

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
        logging.exception("Prediction error")
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
