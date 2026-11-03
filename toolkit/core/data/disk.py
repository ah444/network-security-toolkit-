"""The instance's files and the disk they are on (v5.3 step HR4b): what the storage report
counts, and the free space a new file must leave (HR4c). It only reads: nothing here removes
or changes a file."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

MB = 2**20


class NoSpace(OSError):
    """A new file refused: the disk would keep less free space than ``TOOLKIT_MIN_FREE_MB``
    (HR4c). Its message is for the user."""

    def __init__(self, message: str):
        super().__init__(message)
        self.errno = errno.ENOSPC


def files_under(folder: str | Path) -> list[tuple[Path, int]]:
    """Every file under ``folder`` (its subfolders too), with its size in bytes, sorted by
    path; none when there is no such folder. A file that vanishes meanwhile is skipped."""
    found = []
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = Path(root) / name
            try:
                found.append((path, path.stat().st_size))
            except OSError:
                continue
    return sorted(found)


def require_free(folder: str | Path, size: int, floor_mb: int) -> None:
    """NoSpace unless writing ``size`` bytes in ``folder`` leaves ``floor_mb`` MB free."""
    free = free_bytes(folder)
    if free - size < floor_mb * MB:
        raise NoSpace(f"Not saved: only {free // MB:,} MB free on this disk, and a new file "
                      f"must leave {floor_mb:,} MB free (TOOLKIT_MIN_FREE_MB). Nothing was "
                      "deleted; free some space, then try again.")


def free_bytes(path: str | Path) -> int:
    """The free space of the disk holding ``path`` (its nearest existing folder)."""
    where = Path(path)
    while not where.exists() and where.parent != where:
        where = where.parent
    return shutil.disk_usage(where).free
