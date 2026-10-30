"""Port scan routes (v5.2 step 2.4d): what the page can choose, a scan started as a job, the
session's scan (running or last finished), and the last scan's report as a download. The
job's state, log and cancel are the job routes' (``jobs.py``)."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify

from ...core.data.keys import load_api_keys
from ...core.data.settings import scan_limits
from ..services.jobs import SCAN
from ..services.scan import (
    ScanChoices,
    require_authorized,
    scan_options,
    scan_report,
    start_scan,
)
from ..services.workspace import Workspace
from .api import Answer, json_body, query
from .jobs import job_view
from .schemas import ScanReportRequest, load
from .state import state
from .workspace import with_workspace, workspace

bp = Blueprint("scan", __name__, url_prefix="/api/v1")


@bp.route("/scan/options", methods=["GET"])
def options() -> Answer:
    """The engines (nmap: whether it can run here), the choices, the limits and whether each
    API key is set on the server (never a key)."""
    return jsonify(scan_options(load_api_keys(), scan_limits()))


@bp.route("/jobs/scan", methods=["POST"])
def start() -> Answer:
    """Check a scan and queue it (202 + the job). The authorization is checked first, before
    the session's workspace is even made; nothing reaches the network before it."""
    choices = load(ScanChoices, json_body())
    require_authorized(choices)
    with workspace(create=True) as ws:
        job = start_scan(ws, choices, state().scan_jobs, scan_limits())
        return jsonify(job_view(job, state().scan_jobs)), 202


@bp.route("/scan", methods=["GET"])
@with_workspace()
def session_scan(ws: Workspace) -> Answer:
    """The session's scan that is still queued or running (as ``GET /api/v1/jobs/<id>``) and
    its last finished one (its job's result), each null if none: the page resumes from it."""
    active = ws.active_job(SCAN)
    return jsonify({"job": job_view(active, state().scan_jobs) if active else None,
                    "last": ws.scan.answer if ws.scan is not None else None})


@bp.route("/scan/report", methods=["GET"])
@with_workspace()
def report(ws: Workspace) -> Answer:
    """The last finished scan as a TXT, JSON or CSV file to save (404 NO_SCAN before one)."""
    text, media_type, name = scan_report(ws, load(ScanReportRequest, query()).format)
    return Response(text, content_type=f"{media_type}; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
