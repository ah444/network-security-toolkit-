"""Background jobs (D8, D22): a training or a scan runs outside the request, with progress
and cancel.

Each kind of job has its own queue (v5.2 step 2.4b): one worker thread per kind runs **one
job of that kind at a time for the whole server**; the others wait, so a laptop never
trains twice at full load (each training already uses ``TOOLKIT_MAX_WORKERS`` cores) nor
scans twice at once, while a scan doesn't wait for a training. A job is its own
``ProgressSink``: every ``advance()`` updates the step count and is the point where a
cancel or the time limit takes effect, so the current step (a model's fit) always
finishes first (cooperative cancel). A job that doesn't end in ``done`` leaves nothing
behind.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import CANCELLED, INTERNAL, TIME_LIMIT, ServiceError

log = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, FAILED, STOPPED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, STOPPED)


@dataclass(frozen=True)
class JobKind:
    """What differs between kinds of job: the name the page and the worker thread use, the
    word that starts its messages and the setting behind its time limit."""

    name: str  # "Waiting for another <name> to finish"; the thread "<name>-jobs"
    title: str  # "<title> cancelled. ..."
    time_limit: str  # the environment variable the job's max_seconds comes from


TRAINING = JobKind("training", "Training", "TOOLKIT_JOB_MAX_MIN")
SCAN = JobKind("scan", "Scan", "TOOLKIT_SCAN_MAX_MIN")


class JobCancelled(Exception):
    """Raised from ``advance()`` when a job was cancelled or ran out of time."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(eq=False)
class Job:
    """One training or scan: its state, its progress (it is the ProgressSink) and its
    outcome."""

    kind: JobKind
    work: Callable[[Job], dict[str, Any]]  # runs in the worker thread; returns the result
    on_end: Callable[[], None]  # always called once at the end (releases lock and lease)
    max_seconds: float
    steps: Callable[[int, int], str]  # the phase text while it runs, from (done, total)
    id: str = field(default_factory=lambda: secrets.token_urlsafe(9))
    state: str = QUEUED
    done: int = 0
    total: int = 0
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    ended: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)

    # --- ProgressSink -------------------------------------------------------------------
    def start(self, total: int) -> None:
        """The job's step count; nothing done yet."""
        self.total, self.done = total, 0

    def advance(self, done: int = 1) -> None:
        """Counts finished steps, then raises JobCancelled if the job was cancelled or ran past
        ``max_seconds``: the job stops between steps, never inside one (a training's fit)."""
        self.done = min(self.total, self.done + done) if self.total else self.done + done
        if self.cancel_requested.is_set():
            raise JobCancelled(CANCELLED, f"{self.kind.title} cancelled. Nothing was kept "
                                          "from it.")
        if self.started is not None and time.monotonic() - self.started > self.max_seconds:
            raise JobCancelled(TIME_LIMIT, f"{self.kind.title} stopped after "
                                           f"{self.max_seconds / 60:,.0f} minutes "
                                           f"({self.kind.time_limit}). Nothing was kept.")

    def finish(self) -> None:
        """Nothing: the worker records how the job ended."""
        pass

    # --- What the page shows ------------------------------------------------------------
    def phase(self) -> str:
        """The page's text for where the job is: its state, or while it runs its kind's
        text for its step count (``steps``)."""
        if self.state == QUEUED:
            return f"Waiting for another {self.kind.name} to finish"
        if self.state in FINISHED:
            return {DONE: "Finished", FAILED: "Failed", STOPPED: "Cancelled"}[self.state]
        return self.steps(self.done, self.total)

    def elapsed(self) -> float:
        """Seconds since the job started running (0 while it waits); final once it ended."""
        if self.started is None:
            return 0.0
        return (self.ended or time.monotonic()) - self.started


class JobRunner:
    """One kind's job queue and its one worker thread (started with the first job)."""

    def __init__(self, kind: JobKind) -> None:
        self.kind = kind
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._waiting: list[Job] = []
        self._running: Job | None = None
        self._thread: threading.Thread | None = None

    def submit(self, job: Job) -> None:
        """Queues ``job``, which must be of this runner's kind; the worker thread is started
        with the first job."""
        if job.kind != self.kind:
            raise ValueError(f"a {job.kind.name} job can't wait in the {self.kind.name} queue")
        with self._lock:
            self._waiting.append(job)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work,
                                                name=f"{self.kind.name}-jobs", daemon=True)
                self._thread.start()
        self._queue.put(job)

    def position(self, job: Job) -> int:
        """How many jobs will run before this one (0 when it is running or finished)."""
        with self._lock:
            if job not in self._waiting:
                return 0
            return self._waiting.index(job) + (1 if self._running is not None else 0)

    def cancel(self, job: Job) -> None:
        """A queued job ends now; a running one at its next step (``advance``)."""
        job.cancel_requested.set()
        with self._lock:
            if job.state != QUEUED:
                return
            self._waiting.remove(job)
            job.state, job.ended = STOPPED, time.monotonic()
            job.error = {"code": CANCELLED,
                         "message": f"{job.kind.title} cancelled before it started."}
        job.on_end()

    def _work(self) -> None:
        """The worker thread: one job at a time, in order, skipping cancelled ones; records the
        result or the error (an unexpected one only as a generic message, the details in the log)
        and always calls ``on_end``."""
        while True:
            job = self._queue.get()
            with self._lock:
                if job.state != QUEUED:  # cancelled while waiting
                    continue
                self._waiting.remove(job)
                self._running = job
                job.state, job.started = RUNNING, time.monotonic()
            try:
                job.result = job.work(job)
                job.state = DONE
            except JobCancelled as stop:
                job.error = {"code": stop.code, "message": stop.message}
                job.state = STOPPED
                log.info("%s job stopped (%s)", job.kind.title, stop.code)
            except ServiceError as error:
                job.error = {"code": error.code, "message": error.message}
                if error.details:
                    job.error["details"] = error.details
                job.state = FAILED
            except Exception:  # the worker must survive anything a job does
                log.exception("%s job failed", job.kind.title)
                job.error = {"code": INTERNAL, "message": "Something went wrong on the server. "
                                                          "Details are in the server log."}
                job.state = FAILED
            finally:
                job.ended = time.monotonic()
                with self._lock:
                    self._running = None
                try:
                    job.on_end()
                except Exception:
                    log.exception("Releasing a finished job's workspace failed")
