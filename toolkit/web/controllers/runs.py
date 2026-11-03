"""Saved runs (v5.3 F2c): the list, one run, and two side by side. They only read the
TrainingRun records in ``instance/runs/`` (``toolkit.core.services.ids.comparison``), never a
bundle, and need no session: a saved run belongs to this install, not to a workspace. A run
is deleted only when the user asks, after the page named its files (HR4a,
``toolkit.core.services.ids.run_removal``). The storage report only reads; registering a kept
bundle again adds its record and deletes nothing (HR4b)."""

from __future__ import annotations

from flask import Blueprint, jsonify

from ...core.services.ids.bundle import BundleError
from ...core.services.ids.comparison import NotARun, compare_saved, saved_run, saved_runs, view
from ...core.services.ids.registration import register_bundle
from ...core.services.ids.run_removal import BUNDLE, not_removed, remove_run, run_files
from ...core.services.storage import NoSpace, storage_report
from ..services.errors import ALREADY_SAVED, NO_SPACE, NOT_FOUND, NOT_REMOVED
from .api import Answer, error_response, json_body, query
from .schemas import CompareRequest, DeleteRunRequest, RunsRequest, load
from .state import state

bp = Blueprint("runs", __name__, url_prefix="/api/v1")

NO_SUCH_RUN = "There is no saved run with that id."
NO_SUCH_BUNDLE = "There is no bundle of that run in instance/bundles."


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


@bp.route("/runs/<run_id>/files", methods=["GET"])
def files(run_id: str) -> Answer:
    """What deleting the run removes, in the order it goes: its importance results, its
    bundle when there is one (the user may keep it), its record last. Names, never paths."""
    try:
        found = run_files(run_id)
    except NotARun as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_RUN, 404, NOT_FOUND)
    return jsonify({"run_id": run_id, "files": [item.to_dict() for item in found]})


@bp.route("/runs/<run_id>/delete", methods=["POST"])
def delete(run_id: str) -> Answer:
    """Delete the run the user confirmed: its bundle too only with ``with_bundle``. A file
    that can't be removed is a 409 naming it, and the run is kept."""
    req = load(DeleteRunRequest, json_body())
    try:
        removal = remove_run(run_id, req.with_bundle)
    except NotARun as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_RUN, 404, NOT_FOUND)
    if any(item.kind == BUNDLE for item in removal.removed):
        state().workspaces.forget_export(run_id)
    if removal.failed is not None:
        return error_response(not_removed(removal), 409, NOT_REMOVED)
    return jsonify({"run_id": run_id, "removed": [item.to_dict() for item in removal.removed],
                    "kept": removal.kept.to_dict() if removal.kept else None})


@bp.route("/storage", methods=["GET"])
def storage() -> Answer:
    """Every file of the instance by kind, with the leftovers named, the total and the disk's
    free space. It only reads: nothing is deleted."""
    return jsonify(storage_report())


@bp.route("/bundles/<run_id>/register", methods=["POST"])
def register(run_id: str) -> Answer:
    """The record a kept bundle carries published again, once its signature is checked; a
    record is never replaced."""
    try:
        record = register_bundle(run_id)
    except NotARun as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_BUNDLE, 404, NOT_FOUND)
    except FileExistsError as error:
        return error_response(str(error), 409, ALREADY_SAVED)
    except NoSpace as error:
        return error_response(str(error), 507, NO_SPACE)
    except BundleError as error:
        return error_response(str(error), code=error.code)
    return jsonify({"run_id": run_id, "record": record.name})


@bp.route("/runs/<run_id>", methods=["GET"])
def one(run_id: str) -> Answer:
    """One saved run: its summary, its leaderboard and each model's CV folds."""
    try:
        return jsonify(view(saved_run(run_id)))
    except NotARun as error:
        return error_response(str(error))
    except FileNotFoundError:
        return error_response(NO_SUCH_RUN, 404, NOT_FOUND)
