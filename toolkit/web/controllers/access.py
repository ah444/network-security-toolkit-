"""The unlock page and route (v5.2 step 2.4f): a computer other than this one types the access
token once, when the app is open to the network (``--host``); the session then stays
unlocked until the app stops."""

from __future__ import annotations

from flask import Blueprint, jsonify, redirect, render_template, request, session
from werkzeug.wrappers import Response as BaseResponse  # what redirect() gives

from ..services.access import is_this_computer
from .api import Answer, json_body
from .schemas import UnlockRequest, load
from .state import state

bp = Blueprint("access", __name__)


def _needs_no_token() -> bool:
    """This computer's own browser, an app that isn't open to the network, or a session
    already unlocked in this start."""
    access = state().access
    return access is None or is_this_computer(request.remote_addr) or \
        session.get("unlocked") == access.run_id


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
