"""Job routes for every kind of job (v5.2 step 2.4b): a job's state and progress, and its
cancel. Each kind's route starts its jobs (``POST /api/v1/jobs/training``)."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify

from .api import Answer
from ..services.jobs import FINISHED, QUEUED, RUNNING, Job, JobRunner
from .state import state
from ..services.workspace import Workspace
from .workspace import with_workspace

bp = Blueprint("jobs", __name__, url_prefix="/api/v1")


@bp.route("/jobs/<job_id>", methods=["GET"])
@with_workspace()
def job_status(ws: Workspace, job_id: str) -> Answer:
    """State, progress and, at the end, the result (or the error) of this session's job."""
    job = ws.job(job_id)
    return jsonify(job_view(job, state().runner(job.kind)))


@bp.route("/jobs/<job_id>", methods=["DELETE"])
@with_workspace()
def cancel_job(ws: Workspace, job_id: str) -> Answer:
    """Cancel this session's job: a queued one at once, a running one at its next step."""
    job = ws.job(job_id)
    runner = state().runner(job.kind)
    if job.state not in FINISHED:
        runner.cancel(job)
    return jsonify(job_view(job, runner)), 202


def job_view(job: Job, runner: JobRunner) -> dict[str, Any]:
    """What ``GET /api/v1/jobs/<id>`` answers."""
    state = job.state
    if state == RUNNING and job.cancel_requested.is_set():
        state = "cancelling"
    view: dict[str, Any] = {
        "id": job.id,
        "url": f"/api/v1/jobs/{job.id}",
        "kind": job.kind.name,
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
