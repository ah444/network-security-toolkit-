"""Training routes: the models that can be trained, training itself, and its results."""

import logging
from dataclasses import asdict

from flask import Blueprint, jsonify

from ..config import SplitConfig
from ..ids_core import registry
from ..ids_core.runs import TrainSettings
from ..ids_core.runs import train as train_run
from .api import BadRequestBody, error_response, json_body, test_size_param
from .datasets import label_values, require_data
from .workspace import current_workspace

bp = Blueprint("training", __name__)


@bp.route("/get_available_models", methods=["GET"])
def get_available_models():
    """Models that can be trained here (optional libraries installed)."""
    return jsonify({"models": [
        {"id": name, "name": registry.MODELS[name].title,
         "description": registry.MODELS[name].description,
         "default": registry.MODELS[name].default}
        for name in registry.available_models()
    ]})


@bp.route("/configure", methods=["POST"])
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
    if label_values(data.frame, target).nunique() < 2:
        raise BadRequestBody("The target needs at least two classes.")
    return TrainSettings(label=target, features=tuple(features), models=tuple(models),
                         balancer=balancer, split=SplitConfig(test_size=test_size_param(req)))


def features_info(fitted):
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
    data = require_data(ws)
    settings = _training_settings(data, req)
    ws.split = None  # free the preview split: training makes its own
    try:
        run = train_run(data.frame, profile=data.profile, settings=settings,
                        test_frame=data.test_frame, source=data.source,
                        test_source=data.test_source, sources=data.sources)
    except ValueError:
        logging.exception("Training failed")
        return error_response("Training could not run with these settings (for example: every chosen "
                      "feature was blocked by the leakage guard, or a class is too small for "
                      "the split). Details are in the server log.")
    evaluation = run.evaluation
    if not evaluation.training.models:
        return error_response("No model could be trained. Details are in the server log.",
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
def get_confusion_matrix():
    """Test-split confusion matrix of one trained model."""
    ws = current_workspace()
    req = json_body()
    if ws.run is None:
        return error_response("No models trained yet.")
    name = req.get("model_name") or ws.run.evaluation.selected
    for model in ws.run.evaluation.models:
        if model.name == name:
            return jsonify({"confusion_matrix": [list(r) for r in model.test.confusion],
                            "class_labels": list(model.test.labels), "model_name": name,
                            "dataset_type": "test"})
    return error_response(f"Model '{name}' not found.")
