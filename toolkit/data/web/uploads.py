"""Uploaded files on disk: written only while they are read, then deleted (S2, S3, S11)."""

import logging
import shutil
import time
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)


def prepare(directory: Path) -> None:
    """The upload folder, made if needed, without the files a crash left there."""
    directory.mkdir(parents=True, exist_ok=True)
    remove_stale_uploads(directory)


def save(stream: IO[bytes], path: Path) -> int:
    """Write an upload to a new file; its size in bytes."""
    with path.open("wb") as handle:
        shutil.copyfileobj(stream, handle, 16384)
    return path.stat().st_size


def remove(path: Path) -> None:
    path.unlink(missing_ok=True)


def remove_stale_uploads(directory: Path, max_age: float = 3600) -> int:
    """Delete files a crash left in the upload folder (S11). Only files older than
    ``max_age`` seconds: another server sharing instance/ may be reading a new one."""
    now, removed = time.time(), 0
    for path in directory.iterdir():
        try:
            if path.is_file() and now - path.stat().st_mtime > max_age:
                path.unlink()
                removed += 1
        except OSError:
            log.warning("Could not remove a leftover upload", exc_info=True)
    if removed:
        log.info("Removed %d leftover upload(s) older than %d minutes", removed,
                 max_age // 60)
    return removed
