"""Training routes: the models that can be trained, training jobs, and their results."""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from flask import Blueprint, jsonify

from ..config import SplitConfig
from ..ids_core import registry
from ..ids_core.pipeline import FittedModel
from ..ids_core.runs import TrainingRun, TrainSettings
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
    query,
)
from .datasets import label_values, require_data
from .jobs import FINISHED, Job, job_view
from .schemas import ConfusionRequest, TrainRequest, load
from .state import state
from .workspace import UploadedData, Workspace, with_workspace, workspace

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
        settings = _training_settings(data, req)
        if not ws.training.acquire(blocking=False):
            return error_response("A training run is already in progress in this workspace. "
                                  "Wait for it to finish or cancel it.", 409, BUSY)
        ws.acquire_lease()

        def release() -> None:
            ws.release_lease()
            ws.training.release()

        job = Job(work=functools.partial(_run_training, ws, data, settings), on_end=release,
                  max_seconds=state().config.job_max_minutes * 60,
                  models=len(settings.models or ()), folds=settings.evaluation.cv_folds)
        ws.split = None  # free the preview split: training makes its own
        ws.add_job(job)
        state().jobs.submit(job)
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


def _training_settings(data: UploadedData, req: TrainRequest) -> TrainSettings:
    """TrainSettings from the request, checked here so errors read well (S8)."""
    target = req.target
    features = req.features
    models = req.models or list(registry.default_models())
    balancer = req.balancing or "none"
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


def _run_training(ws: Workspace, data: UploadedData, settings: TrainSettings,
                  job: Job) -> dict[str, Any]:
    """The job's work, in the worker thread. Without the workspace lock: dataset changes
    are refused (409) while the training lock is held. Only a finished training is
    stored; a cancel (JobCancelled from ``job.advance``) or a failure stores nothing."""
    try:
        run = train_run(data.frame, profile=data.profile, settings=settings,
                        test_frame=data.test_frame, source=data.source,
                        test_source=data.test_source, sources=data.sources, progress=job)
    except ValueError:
        log.exception("Training failed")
        raise ApiError("Training could not run with these settings (for example: every chosen "
                       "feature was blocked by the leakage guard, or a class is too small for "
                       "the split). Details are in the server log.", TRAINING_FAILED) from None
    evaluation = run.evaluation
    if not evaluation.training.models or evaluation.selected is None:
        raise ApiError("No model could be trained. Details are in the server log.",
                       TRAINING_FAILED,
                       details={"failures": [asdict(f) for f in evaluation.failures]})
    with ws.lock:
        ws.run, ws.export = run, None
    return run_result(run)


def run_result(run: TrainingRun) -> dict[str, Any]:
    """A finished training's result: the job's result, and the run part of
    GET /api/v1/workspace after a reload (step 1.3c)."""
    evaluation = run.evaluation
    if evaluation.selected is None:  # never stored: _run_training refuses such runs
        raise ApiError("No model could be trained.", TRAINING_FAILED)
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
