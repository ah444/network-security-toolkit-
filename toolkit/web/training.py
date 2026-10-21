"""Training routes: the models that can be trained, training itself, and its results."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from flask import Blueprint, jsonify

from ..config import SplitConfig
from ..ids_core import registry
from ..ids_core.pipeline import FittedModel
from ..ids_core.runs import TrainSettings
from ..ids_core.runs import train as train_run
from .api import (
    BUSY,
    NOT_TRAINED,
    TRAINING_FAILED,
    UNKNOWN_COLUMN,
    Answer,
    ApiError,
    error_response,
    json_body,
)
from .datasets import label_values, require_data
from .schemas import ConfusionRequest, TrainRequest, load
from .workspace import UploadedData, Workspace, current_workspace

bp = Blueprint("training", __name__)


@bp.route("/get_available_models", methods=["GET"])
def get_available_models() -> Answer:
    """Models that can be trained here (optional libraries installed)."""
    return jsonify({"models": [
        {"id": name, "name": registry.MODELS[name].title,
         "description": registry.MODELS[name].description,
         "default": registry.MODELS[name].default}
        for name in registry.available_models()
    ]})


@bp.route("/configure", methods=["POST"])
def configure() -> Answer:
    ws = current_workspace()
    if not ws.training.acquire(blocking=False):
        return error_response("A training run is already in progress in this workspace. "
                              "Wait for it to finish.", 409, BUSY)
    try:
        return _configure_training(ws)
    finally:
        ws.training.release()


def _training_settings(data: UploadedData, req: TrainRequest) -> TrainSettings:
    """TrainSettings from the request, checked here so errors read well (S8)."""
    target = req.target
    features = req.input_features
    models = req.selected_models or list(registry.default_models())
    balancer = req.balancing_method or "none"
    if not target or target not in data.frame.columns:
        raise ApiError("Choose the target column.", UNKNOWN_COLUMN)
    if not features:
        raise ApiError("Choose at least one input feature.")
    missing = [c for c in features if c not in data.frame.columns]
    if missing:
        raise ApiError(f"Unknown input feature(s): {', '.join(missing)}", UNKNOWN_COLUMN)
    if target in features:
        raise ApiError("The target can't also be an input feature.")
    unknown = [m for m in models if m not in registry.available_models()]
    if unknown:
        raise ApiError(f"Unknown or unavailable model(s): {', '.join(unknown)}")
    if balancer != "recommended" and balancer not in registry.BALANCERS:
        raise ApiError(f"Unknown balancing method: {balancer}")
    if label_values(data.frame, target).nunique() < 2:
        raise ApiError("The target needs at least two classes.")
    return TrainSettings(label=target, features=tuple(features), models=tuple(models),
                         balancer=balancer, split=SplitConfig(test_size=req.test_size))


def features_info(fitted: FittedModel) -> list[dict[str, Any]]:
    return [{"name": f.name, "type": f.kind, "classes": list(f.categories)}
            for f in fitted.feature_schema()]


def _blocked(leakage_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons: dict[str, str] = {}
    for finding in leakage_record["findings"]:
        if finding["action"] == "block":
            reasons.setdefault(finding["column"], finding["reason"])
    return [{"column": c, "reason": reasons.get(c, "BLOCKED")} for c in leakage_record["blocked"]]


def _configure_training(ws: Workspace) -> Answer:
    req = load(TrainRequest, json_body())
    data = require_data(ws)
    settings = _training_settings(data, req)
    ws.split = None  # free the preview split: training makes its own
    try:
        run = train_run(data.frame, profile=data.profile, settings=settings,
                        test_frame=data.test_frame, source=data.source,
                        test_source=data.test_source, sources=data.sources)
    except ValueError:
        logging.exception("Training failed")
        return error_response("Training could not run with these settings (for example: every "
                              "chosen feature was blocked by the leakage guard, or a class is too "
                              "small for the split). Details are in the server log.",
                              code=TRAINING_FAILED)
    evaluation = run.evaluation
    if not evaluation.training.models or evaluation.selected is None:
        return error_response("No model could be trained. Details are in the server log.",
                              code=TRAINING_FAILED,
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
        "features_info": features_info(selected),
        "notes": result["notes"],
    })


@bp.route("/get_confusion_matrix", methods=["POST"])
def get_confusion_matrix() -> Answer:
    """Test-split confusion matrix of one trained model."""
    ws = current_workspace()
    req = load(ConfusionRequest, json_body())
    if ws.run is None:
        return error_response("No models trained yet.", code=NOT_TRAINED)
    name = req.model_name or ws.run.evaluation.selected
    for model in ws.run.evaluation.models:
        if model.name == name:
            return jsonify({"confusion_matrix": [list(r) for r in model.test.confusion],
                            "class_labels": list(model.test.labels), "model_name": name,
                            "dataset_type": "test"})
    return error_response(f"Model '{name}' not found.")
