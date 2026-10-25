"""The two HTML pages."""

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
