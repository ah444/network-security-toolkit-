"""Publishing a new file safely (v5.3 step HR3a, RD4: IA4): a run's record, a bundle.

The bytes go first to a temporary file in the destination's folder (``<name>.<random>.tmp``,
which no listing reads), are flushed to the disk (``fsync``) and their size checked; then the
file is published under its name with a call that never replaces one: ``os.rename`` on
Windows, a hard link (``os.link``) on Linux and macOS, whose ``rename`` would replace. A name
that already exists is a ``FileExistsError`` and the file under it is left alone.

The guarantee: after a crash of the process, the file is under its name whole or not at all
(a temporary file may be left: the web server clears them at its start, HR3b). After a power cut it is best
effort: the disk may not have written what it said it had. A file system without hard links
(on Linux and macOS: some FAT and network drives) is refused with a message rather than
published unsafely.
"""

from __future__ import annotations

import errno
import os
import re
import secrets
import sys
from pathlib import Path

from .disk import require_free
from .settings import min_free_mb

HARD_LINKS = sys.platform != "win32"  # how a file is published without replacing one
# what a hard link is refused with (ENOTSUP and EOPNOTSUPP differ on macOS)
NO_LINKS = {errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS}
LEFTOVER = re.compile(r".+\.[0-9a-f]{8}\.tmp")  # the temporary file's name (token_hex(4))


def publish_bytes(path: str | Path, data: bytes) -> Path:
    """Publish ``data`` as the new file ``path`` (whole or not at all); FileExistsError
    rather than replace a file, NoSpace when the disk would keep less free space than
    ``TOOLKIT_MIN_FREE_MB`` (HR4c)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    require_free(path.parent, len(data), min_free_mb())
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size != len(data):
            raise OSError(f"{temporary.name} has {temporary.stat().st_size} bytes, not "
                          f"{len(data)}")
        _finish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)  # published, or failed: never left by a live process
    return path


def publish_text(path: str | Path, text: str) -> Path:
    """``publish_bytes`` of ``text`` in UTF-8, its line ends exactly as given."""
    return publish_bytes(path, text.encode("utf-8"))


def clear_leftovers(folder: str | Path) -> int:
    """Removes the temporary files a stopped process left in ``folder`` (only theirs:
    ``<name>.<8 hex digits>.tmp``); how many. Only the instance's lock holder calls it
    (HR3b), so no live publish loses its file."""
    folder = Path(folder)
    found = [path for path in folder.iterdir() if LEFTOVER.fullmatch(path.name)] \
        if folder.is_dir() else []
    for path in found:
        path.unlink(missing_ok=True)
    return len(found)


def _finish(temporary: Path, path: Path) -> None:
    """The temporary file published as ``path``, never replacing one (FileExistsError)."""
    if not HARD_LINKS:
        os.rename(temporary, path)  # Windows: refuses an existing name
        return
    try:
        os.link(temporary, path)  # refuses an existing name; the temporary one is removed after
    except OSError as error:  # FileExistsError (EEXIST) is not one of NO_LINKS: raised as it is
        if error.errno in NO_LINKS:
            raise OSError(error.errno, f"{path.parent} is on a file system without hard links, "
                          "where the toolkit can't save a file safely: set "
                          "TOOLKIT_INSTANCE_DIR to a folder on another drive") from error
        raise
    _sync_folder(path.parent)


def _sync_folder(folder: Path) -> None:
    """The new name flushed to the disk too (Linux and macOS; best effort)."""
    try:
        handle = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass  # some file systems can't: the file itself is already on the disk
    finally:
        os.close(handle)
