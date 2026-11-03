"""The HTML pages: home, the trainer, the port scanner, the saved runs, the saved scans and a
saved scan's report."""

from __future__ import annotations

from flask import Blueprint, Response, make_response, render_template, request, session, url_for

from ...core.services.recon.saved_scans import NotAScan
from ...core.services.recon.scan_report import report_stylesheet, saved_report_html
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


@bp.route("/scans")
def scans_page() -> str:
    """``GET /scans``: the saved scans, one shown (v5.4 F6b)."""
    return render_template("scans.html")


@bp.route("/scans/<scan_id>/report")
def scan_report_page(scan_id: str) -> Response:
    """``GET /scans/<id>/report``: a saved scan's report to read and print (v5.4 F5b).
    A bad id is a 400, an unknown one a 404, as a line of text.
    See also: core/services/recon/scan_report.py (what it says)."""
    try:
        page = saved_report_html(scan_id, stylesheet_url=url_for("pages.scan_report_css"))
    except NotAScan as error:
        return Response(f"{error}\n", 400, mimetype="text/plain")
    except FileNotFoundError:
        return Response("There is no saved scan with that id.\n", 404, mimetype="text/plain")
    return Response(page, mimetype="text/html")


@bp.route("/scans/report.css")
def scan_report_css() -> Response:
    """``GET /scans/report.css``: the report's look, for the page above."""
    return Response(report_stylesheet(), mimetype="text/css")
