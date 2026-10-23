"""GET /api/v1/workspace: what the session's workspace holds, in the same answers the
upload, training, export and bundle routes gave, so the page can resume after a reload
(v5.1 step 1.3c)."""

from __future__ import annotations

from flask import Blueprint, jsonify

from .api import Answer
from .bundles import bundle_answer, export_answer
from .datasets import dataset_answer
from .jobs import FINISHED, job_view
from .state import state
from .training import run_result
from .workspace import Workspace, with_workspace

bp = Blueprint("resume", __name__, url_prefix="/api/v1")


@bp.route("/workspace", methods=["GET"])
@with_workspace()
def workspace_contents(ws: Workspace) -> Answer:
    """Each part is null when the session has none; a session without a workspace gets
    all nulls, and nothing is created (no cookie is set)."""
    active = [job for job in ws.jobs.values() if job.state not in FINISHED]
    return jsonify({
        "dataset": dataset_answer(ws.data, ws.data_bytes) if ws.data is not None else None,
        "run": run_result(ws.run) if ws.run is not None else None,
        "job": job_view(active[-1], state().jobs) if active else None,  # queued or running
        "export": export_answer(ws.export) if ws.export is not None else None,
        "bundle": bundle_answer(ws.loaded) if ws.loaded is not None else None,
        "predictions": ws.predictions.answer if ws.predictions is not None else None,
    })
