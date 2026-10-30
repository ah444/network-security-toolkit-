"""Training routes: the models that can be trained, a training job started, and its results
(the job's state and cancel: ``jobs.py``)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from flask import Blueprint, jsonify

from ...core.services.ids import registry
from ...core.services.ids.pipeline import FittedModel
from ...core.services.ids.runs import TrainingRun
from ..services.datasets import require_data
from ..services.errors import NOT_TRAINED, TRAINING_FAILED
from .api import Answer, ApiError, error_response, json_body, query
from .jobs import job_view
from ..services.training import start_job, training_settings
from .schemas import ConfusionRequest, TrainRequest, load
from .state import state
from ..services.workspace import Workspace
from .workspace import with_workspace, workspace

log = logging.getLogger(__name__)

bp = Blueprint("training", __name__, url_prefix="/api/v1")


@bp.route("/models", methods=["GET"])
def models() -> Answer:
    """Models that can be trained here (optional libraries installed)."""
    return jsonify({"models": [
        {"id": name, "name": registry.MODELS[name].title,
         "description": registry.MODELS[name].description,
         "default": registry.MODELS[name].default}
        for name in registry.available_models()
    ]})


@bp.route("/jobs/training", methods=["POST"])
def start_training() -> Answer:
    """Queue a training of the chosen models on the session dataset (202 + the job).

    The request and settings are checked here, so mistakes get a 400 at once. One training
    at a time per workspace (``training`` lock, else 409 BUSY); the lock and a lease
    (D22) are held from now until the job ends, so the dataset can't change and the
    workspace isn't dropped while the job waits or runs.
    """
    req = load(TrainRequest, json_body())
    with workspace() as ws:
        data = require_data(ws)
        settings = training_settings(data, req.target, req.features, req.models,
                                     req.balancing, req.test_size)
        job = start_job(ws, data, settings, state().training_jobs,
                        state().config.job_max_minutes * 60, run_result)
        return jsonify(job_view(job, state().training_jobs)), 202


def features_info(fitted: FittedModel) -> list[dict[str, Any]]:
    """Each input feature of a model as the prediction form needs it."""
    return [{"name": f.name, "type": f.kind, "classes": list(f.categories)}
            for f in fitted.feature_schema()]


def _blocked(leakage_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The leakage guard's blocked columns from a run record, each with its first reason."""
    reasons: dict[str, str] = {}
    for finding in leakage_record["findings"]:
        if finding["action"] == "block":
            reasons.setdefault(finding["column"], finding["reason"])
    return [{"column": c, "reason": reasons.get(c, "BLOCKED")} for c in leakage_record["blocked"]]


def run_result(run: TrainingRun) -> dict[str, Any]:
    """A finished training's result: the job's result, and the run part of
    GET /api/v1/workspace after a reload (step 1.3c)."""
    evaluation = run.evaluation
    if evaluation.selected is None:  # never stored: _run_training refuses such runs
        raise ApiError("No model could be trained.", TRAINING_FAILED)  # never stored
    record = run.to_dict()
    result = record["evaluation"]
    selected = evaluation.training.models[evaluation.selected]
    title = registry.MODELS[evaluation.selected].title
    return {
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
    }


@bp.route("/run/confusion", methods=["GET"])
@with_workspace()
def confusion(ws: Workspace) -> Answer:
    """Test-split confusion matrix of one trained model."""
    req = load(ConfusionRequest, query())
    if ws.run is None:
        return error_response("No models trained yet.", code=NOT_TRAINED)
    name = req.model or ws.run.evaluation.selected
    for model in ws.run.evaluation.models:
        if model.name == name:
            return jsonify({"confusion_matrix": [list(r) for r in model.test.confusion],
                            "class_labels": list(model.test.labels), "model_name": name,
                            "dataset_type": "test"})
    return error_response(f"Model '{name}' not found.")
