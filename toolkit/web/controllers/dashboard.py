"""``GET /api/v1/dashboard`` (v5.4 F10): what the home page shows. It only reads; a session
without a workspace gets nulls and no cookie. See also: web/services/dashboard.py."""

from __future__ import annotations

from flask import Blueprint, jsonify

from ..services.dashboard import runs_part, scans_part, session_part
from .api import Answer
from .workspace import workspace

bp = Blueprint("dashboard", __name__, url_prefix="/api/v1")


@bp.route("/dashboard", methods=["GET"])
def dashboard() -> Answer:
    """The scans and their risks, the runs, and the session. The saved files are read after
    the session's lock is let go, so the session's other requests don't wait for them."""
    with workspace() as ws:
        session = session_part(ws)
    return jsonify({"scans": scans_part(), "runs": runs_part(), "session": session})
