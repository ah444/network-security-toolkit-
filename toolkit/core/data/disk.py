"""The instance's files and the disk they are on (v5.3 step HR4b): what the storage report
counts. It only reads: nothing here removes or changes a file."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


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


def free_bytes(path: str | Path) -> int:
    """The free space of the disk holding ``path`` (its nearest existing folder)."""
    where = Path(path)
    while not where.exists() and where.parent != where:
        where = where.parent
    return shutil.disk_usage(where).free
