"""The HTML pages: home, the trainer, the port scanner and the saved runs."""

from __future__ import annotations

from flask import Blueprint, Response, make_response, render_template, request

from ..services.access import is_this_computer
from .state import state

bp = Blueprint("pages", __name__)


@bp.route("/")
def homepage() -> Response:
    """``GET /``: the home page; on this computer, while the toolkit is open to the network,
    with the section for a new access token and the other computers' jobs (HS1), which carries
    the session's CSRF token and so is kept out of the browser cache."""
    other_computers = state().access is not None and is_this_computer(request.remote_addr)
    response = make_response(render_template("home.html", other_computers=other_computers))
    if other_computers:
        response.headers["Cache-Control"] = "no-store"
    return response


@bp.route("/dynamic_ids")
def dynamic_ids_page() -> str:
    """``GET /dynamic_ids``: the trainer's page, told when an idle workspace expires."""
    return render_template("dynamic_ids.html", idle_minutes=state().config.workspace_idle_minutes)


@bp.route("/scan")
def scan_page() -> str:
    """``GET /scan``: the port scanner's page (v5.2 step 2.4e)."""
    return render_template("scan.html")


@bp.route("/runs")
def runs_page() -> str:
    """``GET /runs``: the saved runs, one shown or two compared (v5.3 F2e)."""
    return render_template("runs.html")
