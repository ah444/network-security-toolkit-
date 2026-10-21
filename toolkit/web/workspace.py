"""Workspaces: one per browser session (S7, R4).

The session cookie only holds a random workspace ID; the data stays in this process's
memory. The signing key is random per start unless TOOLKIT_SECRET_KEY is set
(workspaces don't survive a restart anyway).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass

import pandas as pd
from flask import Blueprint, Response, jsonify, session

from ..bundle_files import SavedBundle
from ..ids_core.bundle import Bundle
from ..ids_core.profiles import ProfileMatch
from ..ids_core.runs import DatasetRef, TrainingRun
from ..ids_core.split import SplitResult
from .state import state

bp = Blueprint("workspace", __name__, url_prefix="/api/v1")


@dataclass
class UploadedData:
    """The uploaded dataset as the v5 loader read it."""

    frame: pd.DataFrame
    profile: ProfileMatch
    source: DatasetRef
    files: list[str]
    test_frame: pd.DataFrame | None = None  # official split: the separate test file
    test_source: DatasetRef | None = None
    sources: pd.Series | None = None  # file name per row when several files were combined


@dataclass
class ExportedBundle:
    """The session's run, saved once; ``name`` is the download name the user chose last."""

    saved: SavedBundle
    name: str


class Workspace:
    """Everything one browser session has uploaded and trained."""

    def __init__(self) -> None:
        self.data: UploadedData | None = None
        # (key, SplitResult): last training split used by the previews
        self.split: tuple[tuple[int, str, float], SplitResult] | None = None
        self.run: TrainingRun | None = None  # TrainingRun of the last training
        self.export: ExportedBundle | None = None  # once the run was exported
        self.loaded: Bundle | None = None  # bundle loaded back from a file
        self.training = threading.Lock()  # held while a training runs (one at a time)
        self.last_used = time.monotonic()


class WorkspaceStore:
    """The workspaces of one app by ID. Workspaces unused for ``idle_seconds`` are dropped
    to free memory (not while training)."""

    def __init__(self, idle_seconds: float) -> None:
        self.idle_seconds = idle_seconds
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._workspaces)

    def values(self) -> list[Workspace]:
        return list(self._workspaces.values())

    def _drop_idle(self, now: float) -> None:
        for ws_id, ws in list(self._workspaces.items()):
            if now - ws.last_used > self.idle_seconds and not ws.training.locked():
                del self._workspaces[ws_id]
                logging.info("Dropped idle workspace")

    def current(self) -> Workspace:
        """The calling browser session's workspace, created on first use."""
        now = time.monotonic()
        with self._lock:
            self._drop_idle(now)
            ws = self._workspaces.get(session.get("ws", ""))
            if ws is None:
                ws_id = secrets.token_urlsafe(16)
                ws = self._workspaces[ws_id] = Workspace()
                session["ws"] = ws_id
            ws.last_used = now
            return ws

    def clear(self) -> None:
        """Forget the calling browser session's workspace."""
        with self._lock:
            self._workspaces.pop(session.pop("ws", ""), None)


def current_workspace() -> Workspace:
    """The calling browser session's workspace, created on first use."""
    return state().workspaces.current()


@bp.route("/workspace", methods=["DELETE"])
def clear_workspace() -> Response:
    """Forget this session's dataset and models; the next request starts a new workspace."""
    state().workspaces.clear()
    return jsonify({"message": "Workspace cleared."})
