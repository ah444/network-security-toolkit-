"""Workspaces: one per browser session (S7, R4), by a random ID.

The session cookie only holds that ID (``toolkit.controllers.web.workspace``); the data
stays in this process's memory, so workspaces don't survive a restart.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ...core.services.ids.exports import SavedBundle
from ...core.services.ids.bundle import Bundle
from ...core.services.ids.profiles import ProfileMatch
from ...core.services.ids.runs import DatasetRef, TrainingRun
from ...core.services.ids.split import SplitResult
from .errors import BUSY, DATA_LIMIT, NOT_FOUND, ServiceError, Unavailable
from .jobs import FINISHED, Job

log = logging.getLogger(__name__)

JOBS_KEPT = 5


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


@dataclass
class BatchPredictions:
    """The last "Predict a CSV file" (step 1.4f): the labelled CSV for the download and
    the answer the route gave (GET /api/v1/workspace repeats it). The uploaded rows are
    not kept."""

    csv: bytes
    answer: dict[str, Any]


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
        self.predictions: BatchPredictions | None = None  # the last "Predict a CSV file"
        self.lock = threading.RLock()
        self.training = threading.Lock()  # held while a training runs (one at a time)
        self.last_used = time.monotonic()
        self.data_bytes = 0  # memory of the loaded dataset (counted by the store's cap)
        self.jobs: dict[str, Job] = {}  # this session's last JOBS_KEPT training jobs
        self._leases = 0
        self._lease_lock = threading.Lock()

    @property
    def leased(self) -> bool:
        """True while a training (later: a job) still needs this workspace."""
        return self._leases > 0 or self.training.locked()

    def acquire_lease(self) -> None:
        """Keep this workspace from being dropped until ``release_lease`` (D22); a job
        takes it when submitted and gives it back in the worker thread."""
        with self._lease_lock:
            self._leases += 1

    def release_lease(self) -> None:
        with self._lease_lock:
            self._leases -= 1

    @contextmanager
    def lease(self) -> Iterator[None]:
        """Keep this workspace from being dropped while the block runs (D22)."""
        self.acquire_lease()
        try:
            yield
        finally:
            self.release_lease()

    def add_job(self, job: Job) -> None:
        """Remember a job; only the last JOBS_KEPT finished ones are kept."""
        self.jobs[job.id] = job
        finished = [j for j in self.jobs.values() if j.state in FINISHED]
        for old in finished[:max(0, len(self.jobs) - JOBS_KEPT)]:
            del self.jobs[old.id]

    def job(self, job_id: str) -> Job:
        """This session's job, else 404 (another session's job id answers the same)."""
        job = self.jobs.get(job_id)
        if job is None:
            raise ServiceError("No such training job in this session.", NOT_FOUND)
        return job

    def require_no_training(self) -> None:
        """Refuse a dataset change while a training reads the dataset (409 BUSY)."""
        if self.training.locked():
            raise ServiceError("A training run is in progress in this workspace. Wait for "
                               "it to finish, then try again.", BUSY)


def dataset_bytes(data: UploadedData) -> int:
    """Memory used by a loaded dataset: its frame, the official test file and the per-row
    file names (pandas ``memory_usage(deep=True)``, so text columns count fully)."""
    total = int(data.frame.memory_usage(deep=True).sum())
    if data.test_frame is not None:
        total += int(data.test_frame.memory_usage(deep=True).sum())
    if data.sources is not None:
        total += int(data.sources.memory_usage(deep=True))
    return total


class WorkspaceStore:
    """The workspaces of one app by ID, with memory caps (R4).

    Workspaces unused for ``idle_seconds`` are dropped. A new workspace above
    ``max_workspaces``, or a dataset that would take the datasets of all workspaces above
    ``max_data_bytes``, first drops other sessions' workspaces, least recently used first.
    A leased workspace (a training still needs it) is never dropped.
    """

    def __init__(self, idle_seconds: float, max_workspaces: int = 4,
                 max_data_bytes: int = 4096 * 2**20) -> None:
        self.idle_seconds = idle_seconds
        self.max_workspaces = max_workspaces
        self.max_data_bytes = max_data_bytes
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._workspaces)

    def values(self) -> list[Workspace]:
        return list(self._workspaces.values())

    def _drop_idle(self, now: float) -> int:
        dropped = 0
        for ws_id, ws in list(self._workspaces.items()):
            if now - ws.last_used > self.idle_seconds and not ws.leased:
                del self._workspaces[ws_id]
                dropped += 1
                log.info("Dropped a workspace (idle for over %d minutes)",
                         self.idle_seconds // 60)
        return dropped

    def drop_idle(self) -> int:
        """Drop the idle workspaces now (the background sweep); how many were dropped."""
        with self._lock:
            return self._drop_idle(time.monotonic())

    def _drop_least_recent(self, keep: Workspace | None, reason: str) -> bool:
        """Drop the least recently used unleased workspace other than ``keep``."""
        candidates = [(ws.last_used, ws_id) for ws_id, ws in self._workspaces.items()
                      if ws is not keep and not ws.leased]
        if not candidates:
            return False
        _, ws_id = min(candidates)
        del self._workspaces[ws_id]
        log.info("Dropped the least recently used workspace (%s)", reason)
        return True

    def reserve_data(self, ws: Workspace, size: int) -> None:
        """Make room for a dataset of ``size`` bytes in ``ws`` (its old one is replaced), or
        refuse it (413 DATA_LIMIT). On success ``ws.data_bytes`` is ``size``."""
        limit_mb = self.max_data_bytes // 2**20
        message = (f"The dataset needs {size / 2**20:,.0f} MB of memory; all open workspaces "
                   f"together may use {limit_mb:,} MB (TOOLKIT_MAX_DATA_MB).")
        with self._lock:
            if size > self.max_data_bytes:
                raise ServiceError(message, DATA_LIMIT)
            while sum(o.data_bytes for o in self._workspaces.values() if o is not ws) + size \
                    > self.max_data_bytes:
                if not self._drop_least_recent(ws, "data limit"):
                    raise ServiceError(message + " The other workspaces are busy training.",
                                       DATA_LIMIT)
            ws.data_bytes = size

    def get(self, ws_id: str) -> Workspace | None:
        """The workspace ``ws_id``, if it still exists."""
        now = time.monotonic()
        with self._lock:
            self._drop_idle(now)
            ws = self._workspaces.get(ws_id)
            if ws is not None:
                ws.last_used = now
            return ws

    def get_or_create(self, ws_id: str) -> tuple[Workspace, str]:
        """The workspace ``ws_id``, or a new one (upload, bundle load); and its ID."""
        now = time.monotonic()
        with self._lock:
            self._drop_idle(now)
            ws = self._workspaces.get(ws_id)
            if ws is None:
                while len(self._workspaces) >= self.max_workspaces:
                    if not self._drop_least_recent(None, "workspace limit"):
                        raise Unavailable(
                            f"{len(self._workspaces)} browser sessions already have a workspace "
                            f"that is training (limit: {self.max_workspaces}, "
                            "TOOLKIT_MAX_WORKSPACES). Try again when one has finished.",
                            BUSY)
                ws_id = secrets.token_urlsafe(16)
                ws = self._workspaces[ws_id] = Workspace()
            ws.last_used = now
            return ws, ws_id

    def clear(self, ws_id: str) -> None:
        """Forget the workspace ``ws_id``."""
        with self._lock:
            self._workspaces.pop(ws_id, None)


def start_idle_sweep(store: WorkspaceStore, interval: float = 300) -> threading.Event:
    """Drop idle workspaces every ``interval`` seconds in a background thread, so memory is
    freed even when no request comes; set the returned event to stop it."""
    stop = threading.Event()

    def sweep() -> None:
        while not stop.wait(interval):
            try:
                store.drop_idle()
            except Exception:  # the sweep must never die silently
                log.exception("Idle workspace sweep failed")

    threading.Thread(target=sweep, name="workspace-sweep", daemon=True).start()
    return stop
