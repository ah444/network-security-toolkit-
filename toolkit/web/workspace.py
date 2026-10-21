"""Workspaces: one per browser session (S7, R4).

The session cookie only holds a random workspace ID; the data stays in this process's
memory. The signing key is random per start unless TOOLKIT_SECRET_KEY is set
(workspaces don't survive a restart anyway).
"""

from __future__ import annotations

import functools
import logging
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
from flask import Blueprint, jsonify, session

from ..bundle_files import SavedBundle
from ..ids_core.bundle import Bundle
from ..ids_core.profiles import ProfileMatch
from ..ids_core.runs import DatasetRef, TrainingRun
from ..ids_core.split import SplitResult
from .api import BUSY, Answer, ApiError
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
    """Everything one browser session has uploaded and trained.

    ``lock`` is held by every request while it reads or changes the workspace, so the
    requests of one session take turns (waitress answers several at once). A training
    holds it only to read the dataset and to store its run; meanwhile it holds the
    ``training`` lock (one training at a time) and a lease (D22).
    """

    def __init__(self) -> None:
        self.data: UploadedData | None = None
        # (key, SplitResult): last training split used by the previews
        self.split: tuple[tuple[int, str, float], SplitResult] | None = None
        self.run: TrainingRun | None = None  # TrainingRun of the last training
        self.export: ExportedBundle | None = None  # once the run was exported
        self.loaded: Bundle | None = None  # bundle loaded back from a file
        self.lock = threading.RLock()
        self.training = threading.Lock()  # held while a training runs (one at a time)
        self.last_used = time.monotonic()
        self._leases = 0
        self._lease_lock = threading.Lock()

    @property
    def leased(self) -> bool:
        """True while a training (later: a job) still needs this workspace."""
        return self._leases > 0 or self.training.locked()

    @contextmanager
    def lease(self) -> Iterator[None]:
        """Keep this workspace from being dropped while the block runs (D22)."""
        with self._lease_lock:
            self._leases += 1
        try:
            yield
        finally:
            with self._lease_lock:
                self._leases -= 1

    def require_no_training(self) -> None:
        """Refuse a dataset change while a training reads the dataset (409 BUSY)."""
        if self.training.locked():
            raise ApiError("A training run is in progress in this workspace. Wait for it to "
                           "finish, then try again.", BUSY, 409)


class WorkspaceStore:
    """The workspaces of one app by ID. Workspaces unused for ``idle_seconds`` are dropped
    to free memory; a leased workspace is never dropped."""

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
            if now - ws.last_used > self.idle_seconds and not ws.leased:
                del self._workspaces[ws_id]
                logging.info("Dropped idle workspace")

    def get(self) -> Workspace | None:
        """The calling browser session's workspace, if it has one."""
        now = time.monotonic()
        with self._lock:
            self._drop_idle(now)
            ws = self._workspaces.get(session.get("ws", ""))
            if ws is not None:
                ws.last_used = now
            return ws

    def get_or_create(self) -> Workspace:
        """The calling browser session's workspace, created if needed (upload, bundle load)."""
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


@contextmanager
def workspace(create: bool = False) -> Iterator[Workspace]:
    """The calling session's workspace with its lock held. Without ``create``, a session
    that has none gets an empty, unsaved one, so the routes answer NO_DATASET,
    NOT_TRAINED, ... without creating anything (and without setting a cookie)."""
    store = state().workspaces
    ws = store.get_or_create() if create else (store.get() or Workspace())
    with ws.lock:
        yield ws


View = Callable[[Workspace], Answer]


def with_workspace(create: bool = False,
                   changes_dataset: bool = False) -> Callable[[View], Callable[[], Answer]]:
    """Route decorator: the view gets the session's workspace, locked for the request.
    ``changes_dataset`` routes are refused (409 BUSY) while a training runs."""
    def decorate(view: View) -> Callable[[], Answer]:
        @functools.wraps(view)
        def route() -> Answer:
            with workspace(create) as ws:
                if changes_dataset:
                    ws.require_no_training()
                return view(ws)
        return route
    return decorate


@bp.route("/workspace", methods=["DELETE"])
@with_workspace(changes_dataset=True)
def clear_workspace(ws: Workspace) -> Answer:
    """Forget this session's dataset and models; the next upload starts a new workspace."""
    state().workspaces.clear()
    return jsonify({"message": "Workspace cleared."})
