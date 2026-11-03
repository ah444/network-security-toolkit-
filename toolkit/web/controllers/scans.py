"""Saved scans (v5.4 F6b): the list, one scan, its report, and its delete. They only read the
scan history (``instance/scan_history/``, ``toolkit.core.services.recon.saved_scans``), every
finished scan's (F6a), and need no session: a saved scan belongs to this install, not to a
workspace. A scan is deleted only when the user asks, after the page named it."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify

from ...core.services.recon.saved_scans import (
    NotAScan,
    remove_scan,
    saved_report,
    saved_scan,
    saved_scans,
    scan_view,
)
from ..services.errors import NOT_FOUND, NOT_REMOVED
from ..services.scan import MIME_TYPES
from .api import Answer, error_response, query
from .schemas import ScanReportRequest, ScansRequest, load

bp = Blueprint("scans", __name__, url_prefix="/api/v1")

NO_SUCH_SCAN = "There is no saved scan with that id."


@bp.route("/scans", methods=["GET"])
def saved() -> Answer:
    """The saved scans' summaries, newest first, 50 at a time (``before``: the next page)."""
    req = load(ScansRequest, query())
    try:
        return jsonify(saved_scans(before=req.before or None))
    except NotAScan as error:
        return error_response(str(error))


@bp.route("/scans/<scan_id>", methods=["GET"])
def one(scan_id: str) -> Answer:
    """One saved scan as the scan page shows a finished one, with the choices to scan again."""
    try:
        return jsonify(scan_view(saved_scan(scan_id)))
    except NotAScan as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_SCAN, 404, NOT_FOUND)


@bp.route("/scans/<scan_id>/report", methods=["GET"])
def report(scan_id: str) -> Answer:
    """The saved scan's report as a TXT, JSON or CSV file to save."""
    fmt = load(ScanReportRequest, query()).format
    if fmt not in MIME_TYPES:
        return error_response(f"'format' must be one of {', '.join(MIME_TYPES)}.")
    try:
        text = saved_report(scan_id, fmt)
    except NotAScan as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_SCAN, 404, NOT_FOUND)
    return Response(text, content_type=f"{MIME_TYPES[fmt]}; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="scan-{scan_id}.{fmt}"'})


@bp.route("/scans/<scan_id>/delete", methods=["POST"])
def delete(scan_id: str) -> Answer:
    """Delete the saved scan the user confirmed: its record. One that can't be removed is a
    409 naming it, and the scan is kept."""
    try:
        removed = remove_scan(scan_id)
    except NotAScan as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_SCAN, 404, NOT_FOUND)
    except OSError:
        return error_response(f"{scan_id}.json could not be removed: another program may have "
                              "it open. Close it and try again.", 409, NOT_REMOVED)
    return jsonify({"scan_id": scan_id, "removed": removed})
