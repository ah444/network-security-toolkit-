"""Training routes: the models that can be trained, training jobs, and their results."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from flask import Blueprint, jsonify

from ...services.ids import registry
from ...services.ids.pipeline import FittedModel
from ...services.ids.runs import TrainingRun
from ...services.web.datasets import require_data
from ...services.web.errors import NOT_TRAINED, TRAINING_FAILED
from .api import Answer, ApiError, error_response, json_body, query
from ...services.web.jobs import FINISHED, QUEUED, RUNNING, Job, JobRunner
from ...services.web.training import start_job, training_settings
from .schemas import ConfusionRequest, TrainRequest, load
from .state import state
from ...services.web.workspace import Workspace
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
        job = start_job(ws, data, settings, state().jobs,
                        state().config.job_max_minutes * 60, run_result)
        return jsonify(job_view(job, state().jobs)), 202


@bp.route("/jobs/<job_id>", methods=["GET"])
@with_workspace()
def job_status(ws: Workspace, job_id: str) -> Answer:
    """State, progress and, at the end, the result (or the error) of this session's job."""
    return jsonify(job_view(ws.job(job_id), state().jobs))


@bp.route("/jobs/<job_id>", methods=["DELETE"])
@with_workspace()
def cancel_job(ws: Workspace, job_id: str) -> Answer:
    """Cancel this session's job: a queued one at once, a running one at its next step."""
    job = ws.job(job_id)
    if job.state not in FINISHED:
        state().jobs.cancel(job)
    return jsonify(job_view(job, state().jobs)), 202


def job_view(job: Job, runner: JobRunner) -> dict[str, Any]:
    """What ``GET /api/v1/jobs/<id>`` answers."""
    state = job.state
    if state == RUNNING and job.cancel_requested.is_set():
        state = "cancelling"
    view: dict[str, Any] = {
        "id": job.id,
        "url": f"/api/v1/jobs/{job.id}",
        "state": state,
        "done": job.done,
        "total": job.total,
        "percent": round(100 * job.done / job.total) if job.total else 0,
        "phase": job.phase(),
        "elapsed_seconds": round(job.elapsed(), 1),
    }
    if job.state == QUEUED:
        view["queue_position"] = runner.position(job)
        if view["queue_position"] == 0:  # submitted, the worker has not picked it up yet
            view["phase"] = "Starting"
    if job.result is not None:
        view["result"] = job.result
    if job.error is not None:
        view["error"] = job.error
    return view


def features_info(fitted: FittedModel) -> list[dict[str, Any]]:
    return [{"name": f.name, "type": f.kind, "classes": list(f.categories)}
            for f in fitted.feature_schema()]


def _blocked(leakage_record: Mapping[str, Any]) -> list[dict[str, Any]]:
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
