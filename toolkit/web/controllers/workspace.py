"""The calling browser session's workspace (the cookie holds only its ID, S7), locked
for the request; and the route that forgets it."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from flask import Blueprint, jsonify, session

from ..services.workspace import Workspace
from .api import Answer
from .state import state

bp = Blueprint("workspace", __name__, url_prefix="/api/v1")


@contextmanager
def workspace(create: bool = False) -> Iterator[Workspace]:
    """The calling session's workspace with its lock held. Without ``create``, a session
    that has none gets an empty, unsaved one, so the routes answer NO_DATASET,
    NOT_TRAINED, ... without creating anything (and without setting a cookie)."""
    store = state().workspaces
    if create:
        ws, ws_id = store.get_or_create(session.get("ws", ""))
        if session.get("ws") != ws_id:
            session["ws"] = ws_id  # only a new workspace sets the cookie
    else:
        ws = store.get(session.get("ws", "")) or Workspace()
    with ws.lock:
        yield ws


View = Callable[..., Answer]


def with_workspace(create: bool = False,
                   changes_dataset: bool = False) -> Callable[[View], Callable[..., Answer]]:
    """Route decorator: the view gets the session's workspace, locked for the request.
    ``changes_dataset`` routes are refused (409 BUSY) while a training runs."""
    def decorate(view: View) -> Callable[..., Answer]:
        @functools.wraps(view)
        def route(**kwargs: str) -> Answer:  # URL parts, e.g. a job id
            with workspace(create) as ws:
                if changes_dataset:
                    ws.require_no_training()
                return view(ws, **kwargs)
        return route
    return decorate


@bp.route("/workspace", methods=["DELETE"])
@with_workspace(changes_dataset=True)
def clear_workspace(ws: Workspace) -> Answer:
    """Forget this session's dataset, models and last scan; the next upload starts a new
    workspace. Refused while a training or a scan runs."""
    ws.require_no_scan()
    state().workspaces.clear(session.pop("ws", ""))
    return jsonify({"message": "Workspace cleared."})
