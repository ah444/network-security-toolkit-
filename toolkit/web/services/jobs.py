"""Background jobs (D8, D22): a training or a scan runs outside the request, with progress
and cancel.

Each kind of job has its own queue (v5.2 step 2.4b): one worker thread per kind runs **one
job of that kind at a time for the whole server**; the others wait, so a laptop never
trains twice at full load (each training already uses ``TOOLKIT_MAX_WORKERS`` cores) nor
scans twice at once, while a scan doesn't wait for a training. A job is its own
``ProgressSink``: every ``advance()`` updates the step count and is the point where a
cancel or the time limit takes effect, so the current step (a model's fit) always
finishes first (cooperative cancel). A job that doesn't end in ``done`` leaves nothing
behind. Each job keeps where it was asked for (``Origin``, HS1): a runner can be given the
reason a queued job may no longer start, checked as the worker takes it.
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
THIS_COMPUTER = "this computer"  # the source of a job asked for from a loopback address


@dataclass(frozen=True)
class JobKind:
    """What differs between kinds of job: the name the page and the worker thread use, the
    word that starts its messages and the setting behind its time limit."""

    name: str  # "Waiting for another <name> to finish"; the thread "<name>-jobs"
    title: str  # "<title> cancelled. ..."
    time_limit: str  # the environment variable the job's max_seconds comes from
    logged: bool = False  # its view has the lines it wrote so far (``Job.note``)


TRAINING = JobKind("training", "Training", "TOOLKIT_JOB_MAX_MIN")
SCAN = JobKind("scan", "Scan", "TOOLKIT_SCAN_MAX_MIN", logged=True)
MAX_LOG_LINES = 5000  # a job's log stops here, with a last line saying so


class JobCancelled(Exception):
    """Raised from ``advance()`` when a job was cancelled or ran out of time."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Origin:
    """Where a job was asked for (HS1, RD1): this computer, or another computer's address and
    the unlock its session had (``Access.run_id``), so a new access token can cancel it."""

    source: str = THIS_COMPUTER
    unlock: str | None = None  # None: this computer, which needs no access token


HERE = Origin()


@dataclass(eq=False)
class Job:
    """One training or scan: its state, its progress (it is the ProgressSink) and its
    outcome."""

    kind: JobKind
    work: Callable[[Job], dict[str, Any]]  # runs in the worker thread; returns the result
    on_end: Callable[[], None]  # always called once at the end (releases lock and lease)
    max_seconds: float
    steps: Callable[[int, int], str]  # the phase text while it runs, from (done, total)
    origin: Origin = HERE  # who asked for it (HS1)
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
    log: list[str] = field(default_factory=list)  # a logged kind's lines (``note``)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --- ProgressSink -------------------------------------------------------------------
    def start(self, total: int) -> None:
        """The job's step count; nothing done yet."""
        self.total, self.done = total, 0

    def advance(self, done: int = 1) -> None:
        """Counts finished steps, then raises JobCancelled if the job was cancelled or ran past
        ``max_seconds``: the job stops between steps, never inside one (a training's fit)."""
        self.done = min(self.total, self.done + done) if self.total else self.done + done
        if self.cancel_requested.is_set():
            raise self.stopped(CANCELLED)
        if self.out_of_time():
            raise self.stopped(TIME_LIMIT)

    def finish(self) -> None:
        """Nothing: the worker records how the job ended."""
        pass

    # --- For work that stops on its own (a scan stops on ``cancel_requested``) -----------
    def out_of_time(self) -> bool:
        """True once the job has run longer than ``max_seconds``."""
        return self.started is not None and time.monotonic() - self.started > self.max_seconds

    def stopped(self, code: str) -> JobCancelled:
        """The JobCancelled for a cancel (CANCELLED) or the time limit (TIME_LIMIT)."""
        if code == TIME_LIMIT:
            return JobCancelled(TIME_LIMIT, f"{self.kind.title} stopped after "
                                            f"{self.max_seconds / 60:,.0f} minutes "
                                            f"({self.kind.time_limit}). Nothing was kept.")
        return JobCancelled(CANCELLED, f"{self.kind.title} cancelled. Nothing was kept from it.")

    def tally(self, done: int, total: int) -> None:
        """Sets the progress without the cancel check of ``advance`` (any thread may call it)."""
        self.done, self.total = min(done, total), total

    def note(self, line: str) -> None:
        """Adds a line to the job's log (any thread); after MAX_LOG_LINES, one line saying the
        rest was left out."""
        with self._log_lock:
            if len(self.log) < MAX_LOG_LINES:
                self.log.append(line)
            elif len(self.log) == MAX_LOG_LINES:
                self.log.append(f"(the log stops at {MAX_LOG_LINES:,} lines; the result has "
                                "everything)")

    def notes(self, since: int = 0) -> list[str]:
        """The log's lines from line ``since`` (0: all)."""
        return self.log_from(since)[0]

    def log_from(self, since: int) -> tuple[list[str], int]:
        """The log's lines from line ``since`` and how many lines it has, read together."""
        with self._log_lock:
            return self.log[max(0, since):], len(self.log)

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
    """One kind's job queue and its one worker thread (started with the first job).
    ``refusal`` (HS1) says why a queued job may no longer start, or None: the worker asks it
    as it takes each job, and cancels the job with that reason instead of running it."""

    def __init__(self, kind: JobKind,
                 refusal: Callable[[Job], str | None] | None = None) -> None:
        self.kind = kind
        self._refusal = refusal
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

    def active(self) -> list[Job]:
        """The running job, then the queued ones in their order."""
        with self._lock:
            return ([self._running] if self._running is not None else []) + self._waiting

    def cancel(self, job: Job) -> None:
        """A queued job ends now; a running one at its next step (``advance``)."""
        job.cancel_requested.set()
        self.cancel_waiting(job, f"{job.kind.title} cancelled before it started.")

    def cancel_waiting(self, job: Job, message: str) -> bool:
        """Ends ``job`` with ``message`` if it is still queued (True); a running or finished
        one is left as it is, not even asked to stop (False)."""
        with self._lock:
            if job.state != QUEUED:
                return False
            self._waiting.remove(job)
            job.state, job.ended = STOPPED, time.monotonic()
            job.error = {"code": CANCELLED, "message": message}
        job.on_end()
        return True

    def _work(self) -> None:
        """The worker thread: one job at a time, in order, skipping cancelled ones; records the
        result or the error (an unexpected one only as a generic message, the details in the log)
        and always calls ``on_end``, before the job is seen to end: a client that sees the end
        can start the next job at once (v5.2 step 2.4d). A job ``refusal`` refuses never starts."""
        while True:
            job = self._queue.get()
            with self._lock:
                if job.state != QUEUED:  # cancelled while waiting
                    continue
                reason = self._refusal(job) if self._refusal is not None else None
                if reason is None:
                    self._waiting.remove(job)
                    self._running = job
                    job.state, job.started = RUNNING, time.monotonic()
            if reason is not None:
                self.cancel_waiting(job, reason)
                continue
            ended = FAILED
            result: dict[str, Any] | None = None
            error: dict[str, Any] | None = None
            try:
                result, ended = job.work(job), DONE
            except JobCancelled as stop:
                error, ended = {"code": stop.code, "message": stop.message}, STOPPED
                log.info("%s job stopped (%s)", job.kind.title, stop.code)
            except ServiceError as refused:
                error = {"code": refused.code, "message": refused.message}
                if refused.details:
                    error["details"] = refused.details
            except Exception:  # the worker must survive anything a job does
                log.exception("%s job failed", job.kind.title)
                error = {"code": INTERNAL, "message": "Something went wrong on the server. "
                                                      "Details are in the server log."}
            finally:
                job.ended = time.monotonic()
                with self._lock:
                    self._running = None
                try:
                    job.on_end()
                except Exception:
                    log.exception("Releasing a finished job's workspace failed")
                job.result, job.error, job.state = result, error, ended
