"""Saved runs (v5.3 F2c): the list, one run, and two side by side. They only read the
TrainingRun records in ``instance/runs/`` (``toolkit.core.services.ids.comparison``), never a
bundle, and need no session: a saved run belongs to this install, not to a workspace."""

from __future__ import annotations

from flask import Blueprint, jsonify

from ...core.services.ids.comparison import NotARun, compare_saved, saved_run, saved_runs, view
from ..services.errors import NOT_FOUND
from .api import Answer, error_response, query
from .schemas import CompareRequest, RunsRequest, load

bp = Blueprint("runs", __name__, url_prefix="/api/v1")

NO_SUCH_RUN = "There is no saved run with that id."


@bp.route("/runs", methods=["GET"])
def saved() -> Answer:
    """The saved runs' summaries, newest first, 50 at a time (``before``: the next page)."""
    req = load(RunsRequest, query())
    try:
        return jsonify(saved_runs(before=req.before or None))
    except NotARun as error:
        return error_response(str(error))


@bp.route("/runs/compare", methods=["GET"])
def compare() -> Answer:
    """Two saved runs side by side; the differences only when they scored the same test rows."""
    req = load(CompareRequest, query())
    try:
        return jsonify(compare_saved(req.a, req.b))
    except NotARun as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_RUN, 404, NOT_FOUND)


@bp.route("/runs/<run_id>", methods=["GET"])
def one(run_id: str) -> Answer:
    """One saved run: its summary, its leaderboard and each model's CV folds."""
    try:
        return jsonify(view(saved_run(run_id)))
    except NotARun as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_RUN, 404, NOT_FOUND)
