"""The unlock page and route (v5.2 step 2.4f): a computer other than this one types the access
token once, when the app is open to the network (``--host``); the session then stays
unlocked until the app stops or a new token is made on this computer.

The new token and the other computers' jobs (HS1, RD1), on this computer only: a new token
locks every other computer out at once, and a job one of them asked for can be stopped here.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, session
from werkzeug.wrappers import Response as BaseResponse  # what redirect() gives

from ..services.access import Access, is_this_computer, lock_out, remote_jobs
from ..services.errors import NOT_FOUND, THIS_COMPUTER_ONLY, ServiceError
from ..services.jobs import HERE, Job, JobRunner, Origin
from ..ui import console as messages
from .api import Answer, json_body
from .jobs import job_view
from .schemas import UnlockRequest, load
from .state import state

bp = Blueprint("access", __name__)


def _needs_no_token() -> bool:
    """This computer's own browser, an app that isn't open to the network, or a session
    already unlocked in this start."""
    access = state().access
    return access is None or is_this_computer(request.remote_addr) or \
        session.get("unlocked") == access.run_id


def origin() -> Origin:
    """Where the job asked for now comes from (HS1): this computer, or the other computer's
    address and the unlock of its session."""
    if state().access is None or is_this_computer(request.remote_addr):
        return HERE
    return Origin(request.remote_addr or "", session.get("unlocked"))


@bp.route("/unlock")
def unlock_page() -> BaseResponse | str:
    """``GET /unlock``: the form for the access token (the home page when none is needed)."""
    if _needs_no_token():
        return redirect("/")
    return render_template("unlock.html")


@bp.route("/api/v1/unlock", methods=["POST"])
def unlock() -> Answer:
    """``POST /api/v1/unlock`` with ``token``: unlocks this session (WRONG_TOKEN, 401, for
    another token; TOO_MANY_TRIES, 429, after five wrong ones in a minute)."""
    req = load(UnlockRequest, json_body())
    access = state().access
    if _needs_no_token() or access is None:
        return jsonify({"message": "No access token is needed here."})
    session["unlocked"] = access.unlock(req.token, request.remote_addr or "")
    return jsonify({"message": "Unlocked: this browser can use the toolkit until it stops."})


def _here_only() -> Access:
    """The access token, for this computer only: NOT_FOUND (404) when the app isn't open to the
    network, THIS_COMPUTER_ONLY (403) from another computer."""
    access = state().access
    if access is None:
        raise ServiceError("No access token is used: the toolkit is open to this computer "
                           "only.", NOT_FOUND)
    if not is_this_computer(request.remote_addr):
        raise ServiceError("Only on the toolkit's own computer.", THIS_COMPUTER_ONLY)
    return access


def _runners() -> tuple[JobRunner, JobRunner]:
    """The app's job runners, every kind."""
    return state().training_jobs, state().scan_jobs


def _remote_view(job: Job, runner: JobRunner) -> dict[str, Any]:
    """A job of another computer as this computer's list shows it: no result and no log."""
    view = job_view(job, runner)
    return {"id": job.id, "url": f"/api/v1/access/jobs/{job.id}", "kind": view["kind"],
            "source": job.origin.source, "state": view["state"], "phase": view["phase"],
            "elapsed_seconds": view["elapsed_seconds"]}


@bp.route("/api/v1/access/token", methods=["POST"])
def new_token() -> Answer:
    """``POST /api/v1/access/token``, on this computer only: a new access token, printed in the
    terminal; every other computer is locked out at once and the jobs they asked for that
    still wait are cancelled (a running one goes on)."""
    token, cancelled = lock_out(_here_only(), _runners())
    messages.new_token(token)
    return jsonify({"message": "New token made: it is in the terminal where the toolkit was "
                               "started. Every other computer is locked out until it types it. "
                               f"Waiting jobs of other computers cancelled: {cancelled}.",
                    "cancelled": cancelled})


@bp.route("/api/v1/access/jobs", methods=["GET"])
def other_computers_jobs() -> Answer:
    """``GET /api/v1/access/jobs``, on this computer only: the jobs other computers asked for
    that are running or waiting, the running one first."""
    _here_only()
    return jsonify({"jobs": [_remote_view(job, runner) for runner, job in remote_jobs(_runners())]})


@bp.route("/api/v1/access/jobs/<job_id>", methods=["DELETE"])
def stop_other_computers_job(job_id: str) -> Answer:
    """``DELETE /api/v1/access/jobs/<id>``, on this computer only: stop a job another computer
    asked for, a waiting one at once, a running one at its next step (NOT_FOUND once it
    ended)."""
    _here_only()
    for runner, job in remote_jobs(_runners()):
        if job.id == job_id:
            runner.cancel(job)
            return jsonify(_remote_view(job, runner)), 202
    raise ServiceError("No job of another computer has that id: it may have ended.", NOT_FOUND)
