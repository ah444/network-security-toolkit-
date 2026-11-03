"""One web server per instance (v5.3 step HR3b, RD4: IA4): an operating-system lock on
``instance/web.lock`` (``msvcrt`` on Windows, ``fcntl`` on Linux and macOS, both part of
Python), held while the server runs. The system releases it when the process ends, however
it ends (Ctrl+C, the launcher menu, a crash), so it never needs clearing by hand.

The holder writes who it is (its process id and address) after the locked first byte, so a
second server can say which one holds the folder. Only the holder clears the temporary files
a crash left in ``runs/`` and ``bundles/`` (``publish.clear_leftovers``), at its start: the
command line takes no lock, as it writes neither (HR3b, decided with the user).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from ...core.data.ids.bundle_files import bundles_dir
from ...core.data.ids.run_files import runs_dir
from ...core.data.publish import clear_leftovers
from ...core.data.settings import instance_dir

LOCK_NAME = "web.lock"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Holder:
    """The web server that holds an instance: its process id and its address."""

    pid: int
    url: str


class InstanceBusy(Exception):
    """Another web server holds the instance ``folder``; ``holder`` when it says who it is."""

    def __init__(self, folder: Path, holder: Holder | None):
        super().__init__(f"{folder} is held by another web server")
        self.folder, self.holder = folder, holder


class InstanceLock:
    """The lock while it is held; ``release`` (or the end of the process) gives it back."""

    def __init__(self, handle: IO[bytes]):
        self._handle: IO[bytes] | None = handle

    def release(self) -> None:
        """Gives the lock back (closing the file releases it)."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def take(url: str, environ: Mapping[str, str] | None = None) -> InstanceLock:
    """The instance's lock for this web server at ``url``, then the leftovers of a crash
    cleared; InstanceBusy when another server holds it."""
    folder = instance_dir(environ)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOCK_NAME
    handle = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT), "r+b")  # never truncated here
    try:
        _lock(handle)
    except OSError:
        handle.close()
        raise InstanceBusy(folder, holder(path)) from None
    handle.seek(1)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "url": url}).encode("utf-8"))
    handle.flush()
    cleared = sum(clear_leftovers(where) for where in (runs_dir(environ), bundles_dir(environ)))
    if cleared:
        log.info("Removed %d temporary file(s) a stopped export left in %s", cleared, folder)
    return InstanceLock(handle)


def holder(path: Path) -> Holder | None:
    """Who holds the lock at ``path``, from what it wrote after the locked byte; None when
    it can't be read."""
    try:
        with path.open("rb") as handle:
            handle.seek(1)  # the first byte is locked (Windows refuses to read it)
            found = json.loads(handle.read(4096).decode("utf-8"))
        return Holder(int(found["pid"]), str(found["url"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _lock(handle: IO[bytes]) -> None:
    """The lock taken at once, or OSError when another process holds it."""
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # the first byte
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
