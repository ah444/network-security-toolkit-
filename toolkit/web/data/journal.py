"""The journal of the web server's jobs (v5.3 step HR3c, RD4: IA4): ``instance/jobs.journal``,
one JSON line when a job is asked for, when it starts and when it ends.

The jobs and the workspaces live in the server's memory, so a server that stops (Ctrl+C, a
crash, a power cut) loses them. At its next start the holder of the instance's lock
(``instance_lock.take``) reads the journal: a job asked for that never ended is returned as
interrupted, to be told once, and the journal starts again empty. A line cut off by a crash
is not JSON and is left out, so each line counts whole or not at all. Writing a line never
stops a job: a failure is a warning in the log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from ...core.data.settings import instance_dir

JOURNAL_NAME = "jobs.journal"
MAX_BYTES = 10 * 2**20  # read no more of a journal than this (a server runs some thousand jobs)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Interrupted:
    """A job the last server didn't finish: its kind, when it was asked for (seconds since the
    epoch), from where, and whether it had started or was still waiting."""

    kind: str
    asked: float
    source: str
    ran: bool


def journal_path(environ: Mapping[str, str] | None = None) -> Path:
    """``instance/jobs.journal``."""
    return instance_dir(environ) / JOURNAL_NAME


class JobJournal:
    """Appends the lines of one server's jobs; safe from every thread. Without a ``path``
    it writes nothing (a runner with no journal)."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()

    def asked(self, job_id: str, kind: str, source: str) -> None:
        """A job was asked for (queued)."""
        self._write({"job": job_id, "event": "asked", "kind": kind, "source": source,
                     "at": round(time.time(), 3)})

    def started(self, job_id: str) -> None:
        """A job started running."""
        self._write({"job": job_id, "event": "started"})

    def ended(self, job_id: str, state: str) -> None:
        """A job ended: done, failed or cancelled."""
        self._write({"job": job_id, "event": "ended", "state": state})

    def _write(self, line: Mapping[str, object]) -> None:
        """One whole line, flushed to the disk; a failure is logged, never raised."""
        if self.path is None:
            return
        data = (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("ab") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError as error:
            log.warning("Could not write the jobs journal %s: %s", self.path, error)


def recover(path: Path) -> list[Interrupted]:
    """The jobs the journal at ``path`` shows asked for and never ended, oldest first; then
    the journal emptied. Only the instance's lock holder calls it, at its start."""
    try:
        with path.open("rb") as handle:
            text = handle.read(MAX_BYTES).decode("utf-8", "replace")
    except FileNotFoundError:
        return []
    except OSError as error:
        log.warning("Could not read the jobs journal %s: %s", path, error)
        return []
    asked: dict[str, Interrupted] = {}
    for line in text.splitlines():
        try:
            entry = json.loads(line)
            job, event = str(entry["job"]), entry["event"]
        except (ValueError, KeyError, TypeError):
            continue  # cut off by a crash, or not ours: left out
        if event == "asked":
            at = entry.get("at")
            asked[job] = Interrupted(str(entry.get("kind", "job")),
                                     float(at) if isinstance(at, (int, float)) else 0.0,
                                     str(entry.get("source", "")), ran=False)
        elif event == "started" and job in asked:
            asked[job] = replace(asked[job], ran=True)
        elif event == "ended":
            asked.pop(job, None)
    try:
        path.write_bytes(b"")  # a new start: nothing of the last server's jobs is kept
    except OSError as error:
        log.warning("Could not empty the jobs journal %s: %s", path, error)
    return sorted(asked.values(), key=lambda job: job.asked)
