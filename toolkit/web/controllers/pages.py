"""The HTML pages: home, the trainer, the port scanner and the saved runs."""

from __future__ import annotations

from flask import Blueprint, Response, make_response, render_template, request, session

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


def interrupted_once(kind: str) -> list[str]:
    """The last server's unfinished jobs of ``kind`` (HR3c), the first time this session
    opens their page; after that, none."""
    lines = state().interrupted.get(kind, [])
    if not lines or session.get(f"seen_interrupted_{kind}"):
        return []
    session[f"seen_interrupted_{kind}"] = True
    return lines


@bp.route("/dynamic_ids")
def dynamic_ids_page() -> str:
    """``GET /dynamic_ids``: the trainer's page, told when an idle workspace expires, and once
    of a training the last server cut off."""
    return render_template("dynamic_ids.html", idle_minutes=state().config.workspace_idle_minutes,
                           interrupted=interrupted_once("training")
                           + interrupted_once("importance"))


@bp.route("/scan")
def scan_page() -> str:
    """``GET /scan``: the port scanner's page (v5.2 step 2.4e), and once of a scan the last
    server cut off."""
    return render_template("scan.html", interrupted=interrupted_once("scan"))


@bp.route("/runs")
def runs_page() -> str:
    """``GET /runs``: the saved runs, one shown or two compared (v5.3 F2e)."""
    return render_template("runs.html")
