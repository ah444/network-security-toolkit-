"""Workspaces: one per browser session (S7, R4).

The session cookie only holds a random workspace ID; the data stays in this process's
memory. The signing key is random per start unless TOOLKIT_SECRET_KEY is set
(workspaces don't survive a restart anyway).
"""

import logging
import secrets
import threading
import time
from dataclasses import dataclass

import pandas as pd
from flask import Blueprint, jsonify, session

from ..ids_core.runs import DatasetRef
from .state import state

bp = Blueprint("workspace", __name__)


@dataclass
class UploadedData:
    """The uploaded dataset as the v5 loader read it."""

    frame: pd.DataFrame
    profile: object  # ids_core.profiles.ProfileMatch
    source: DatasetRef
    files: list
    test_frame: pd.DataFrame = None  # official split: the separate test file
    test_source: DatasetRef = None
    sources: pd.Series = None  # file name per row when several files were combined


class Workspace:
    """Everything one browser session has uploaded and trained."""

    def __init__(self):
        self.data = None    # UploadedData
        self.split = None   # (key, SplitResult): last training split used by the previews
        self.run = None     # TrainingRun of the last /configure
        self.export = None  # [SavedBundle, download name] once the run was exported
        self.loaded = None  # Bundle loaded with /load_model
        self.training = threading.Lock()  # held while /configure trains (one at a time)
        self.last_used = time.monotonic()


class WorkspaceStore:
    """The workspaces of one app by ID. Workspaces unused for ``idle_seconds`` are dropped
    to free memory (not while training)."""

    def __init__(self, idle_seconds):
        self.idle_seconds = idle_seconds
        self._workspaces = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._workspaces)

    def values(self):
        return list(self._workspaces.values())

    def _drop_idle(self, now):
        for ws_id, ws in list(self._workspaces.items()):
            if now - ws.last_used > self.idle_seconds and not ws.training.locked():
                del self._workspaces[ws_id]
                logging.info("Dropped idle workspace")

    def current(self):
        """The calling browser session's workspace, created on first use."""
        now = time.monotonic()
        with self._lock:
            self._drop_idle(now)
            ws = self._workspaces.get(session.get("ws"))
            if ws is None:
                ws_id = secrets.token_urlsafe(16)
                ws = self._workspaces[ws_id] = Workspace()
                session["ws"] = ws_id
            ws.last_used = now
            return ws

    def clear(self):
        """Forget the calling browser session's workspace."""
        with self._lock:
            self._workspaces.pop(session.pop("ws", None), None)


def current_workspace():
    """The calling browser session's workspace, created on first use."""
    return state().workspaces.current()


@bp.route("/workspace", methods=["DELETE"])
def clear_workspace():
    """Forget this session's dataset and models; the next request starts a new workspace."""
    state().workspaces.clear()
    return jsonify({"message": "Workspace cleared."})
