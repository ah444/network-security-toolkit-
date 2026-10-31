"""The HTML pages: home, the trainer, the port scanner and the saved runs."""

from __future__ import annotations

from flask import Blueprint, render_template

from .state import state

bp = Blueprint("pages", __name__)


@bp.route("/")
def homepage() -> str:
    """``GET /``: the home page."""
    return render_template("home.html")


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
