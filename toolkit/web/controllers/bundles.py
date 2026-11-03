"""Signed bundles (D11, S1, S4): export, download and load back; predictions with the
session's trained or loaded models, one row or a whole CSV file (step 1.4f)."""

from __future__ import annotations

import io
import logging
import warnings
from collections.abc import Mapping
from typing import Any

from flask import Blueprint, jsonify, request, send_file

from ...core.services.ids.exports import INCOMPLETE, BundleExists
from ...core.services.ids import registry
from ...core.services.ids.bundle import Bundle, BundleError, BundleVersionWarning
from ...core.services.ids.pipeline import FittedModel
from ...core.services.ids.runs import TrainingRun
from ...core.services.storage import NoSpace
from ..services.errors import NO_MODEL_LOADED, NO_SPACE, NOT_FOUND, NOT_TRAINED, TOO_LARGE
from ..services import prediction
from .api import Answer, error_response, json_body
from .schemas import ExportRequest, PredictRequest, load
from .security import display_name, safe_model_basename, safe_upload_name
from .state import state
from .training import features_info
from ..services.workspace import ExportedBundle, Workspace
from .workspace import with_workspace

log = logging.getLogger(__name__)

bp = Blueprint("bundles", __name__, url_prefix="/api/v1")


@bp.route("/run/predict", methods=["POST"])
@with_workspace()
def predict(ws: Workspace) -> Answer:
    """``POST /run/predict``: one row predicted with the session's trained models."""
    req = load(PredictRequest, json_body())
    if ws.run is None:
        return error_response("Models not trained yet. Train a model first.", code=NOT_TRAINED)
    training = {"typical_values": dict(ws.run.evaluation.training.typical)}
    return jsonify(prediction.predict_one(ws.run.evaluation.training.models, ws.run.evaluation.selected,
                               req.values, req.model, training))


@bp.route("/bundle/predict", methods=["POST"])
@with_workspace()
def predict_loaded(ws: Workspace) -> Answer:
    """``POST /bundle/predict``: one row predicted with the loaded bundle's models."""
    req = load(PredictRequest, json_body())
    if ws.loaded is None:
        return error_response("No model loaded. Load a bundle exported by this toolkit first.",
                              code=NO_MODEL_LOADED)
    return jsonify(prediction.predict_one(ws.loaded.models, ws.loaded.selected,
                               req.values, req.model, ws.loaded.run.get("training")))


# --- Batch prediction: a whole CSV file (plan F1, step 1.4f) ---------------------------------

def _predict_file(ws: Workspace, source: str, models: Mapping[str, FittedModel],
                  default: str | None, label: str | None) -> Answer:
    """Label every row of the uploaded file; keeps the labelled CSV for GET /predictions."""
    if (request.content_length or 0) > state().config.max_predict_mb * 2**20:
        return error_response(prediction.too_large(state().config.max_predict_mb), 413, TOO_LARGE)  # before request.form reads the body
    algorithm = request.form.get("model") or default or ""
    if algorithm not in models:
        return error_response(f"Invalid algorithm selection. Available models: {', '.join(models)}")
    fitted = models[algorithm]
    file = request.files.get("file")
    if file is None or not file.filename:
        return error_response("Choose a .csv or .txt file to predict.")
    name = display_name(file.filename)
    if not file.filename.lower().endswith((".csv", ".txt")):
        return error_response(f"Unsupported file type: {name} (use .csv or .txt).")
    save_path = state().upload_dir / safe_upload_name(file.filename)
    if save_path.resolve().parent != state().upload_dir.resolve():
        return error_response("Invalid file name.")
    frame = prediction.read_prediction_file(file.stream, save_path, name, state().config.max_predict_mb)
    return jsonify(prediction.predict_file(ws, source, fitted, frame, name, label))


def _run_label(run: TrainingRun) -> str | None:
    """The trained run's label column (to compare a predicted file with), or None."""
    return run.settings.label or (run.profile.label_column if run.profile else None)


def _record_label(record: Mapping[str, Any]) -> str | None:
    """The loaded bundle's label column, from its run record, or None."""
    label = record["settings"].get("label") or (record.get("profile") or {}).get("label_column")
    return str(label) if label else None


@bp.route("/run/predict-file", methods=["POST"])
@with_workspace()
def predict_file(ws: Workspace) -> Answer:
    """``POST /run/predict-file``: every row of a CSV file, with the trained models."""
    if ws.run is None:
        return error_response("Models not trained yet. Train a model first.", code=NOT_TRAINED)
    return _predict_file(ws, "run", ws.run.evaluation.training.models,
                         ws.run.evaluation.selected, _run_label(ws.run))


@bp.route("/bundle/predict-file", methods=["POST"])
@with_workspace()
def predict_file_loaded(ws: Workspace) -> Answer:
    """``POST /bundle/predict-file``: every row of a CSV file, with the loaded bundle."""
    if ws.loaded is None:
        return error_response("No model loaded. Load a bundle exported by this toolkit first.",
                              code=NO_MODEL_LOADED)
    return _predict_file(ws, "bundle", ws.loaded.models, ws.loaded.selected,
                         _record_label(ws.loaded.run))


@bp.route("/predictions", methods=["GET"])
@with_workspace()
def download_predictions(ws: Workspace) -> Answer:
    """``GET /predictions``: the last labelled file of the session, as CSV."""
    if ws.predictions is None:
        return error_response("Predict a CSV file first.", 404, NOT_FOUND)
    return send_file(io.BytesIO(ws.predictions.csv), as_attachment=True,
                     download_name=ws.predictions.answer["download_name"],
                     mimetype="text/csv")


def _exists_message(error: FileExistsError) -> str:
    """What blocks an export: an incomplete bundle of the run (HR3d), never removed on its own,
    or a whole one, never overwritten. The file's name, not its path."""
    if isinstance(error, BundleExists) and error.state == INCOMPLETE:
        return (f"An incomplete bundle of this run is in the bundles folder: its export was "
                f"interrupted. Remove {error.path.name} from instance/bundles, then export "
                "again (it is never removed on its own).")
    return "A bundle for this run already exists; it was not overwritten."


def _bundle_refused(error: BundleError) -> Answer:
    """BundleError messages are written for users and never contain paths (tested)."""
    return error_response(str(error), code=error.code)


@bp.route("/run/export", methods=["POST"])
@with_workspace()
def export(ws: Workspace) -> Answer:
    """Save the session's run as a signed bundle in instance/bundles/ (never overwritten);
    the browser downloads it from /api/v1/run/bundle. Its record was saved by its training
    (HR5): Export never touches it."""
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
        except BundleError as error:
            return _bundle_refused(error)
        except FileExistsError as error:
            return error_response(_exists_message(error), 409)
        except NoSpace as error:  # HR4c: the disk is too full to save it
            return error_response(str(error), 507, NO_SPACE)
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
    """``GET /run/bundle``: the session's exported bundle, under the name the user gave (checked by
    ``safe_model_basename``, so it can't carry a path)."""
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
