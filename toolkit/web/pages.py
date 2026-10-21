"""The two HTML pages."""

from flask import Blueprint, render_template

from .state import state

bp = Blueprint("pages", __name__)


@bp.route("/")
def homepage():
    return render_template("home.html")


@bp.route("/dynamic_ids")
def dynamic_ids_page():
    return render_template("dynamic_ids.html", idle_minutes=state().config.workspace_idle_minutes)
