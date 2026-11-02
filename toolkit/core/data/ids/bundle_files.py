"""Bundle files and the signing key (plan D11): bytes on disk, nothing more.

The key is created once, on the first save, in ``instance/signing.key`` (gitignored;
``TOOLKIT_INSTANCE_DIR`` moves ``instance/``). Loading never creates a key. Bundles live in
``instance/bundles/<run id>.bundle`` and are never overwritten. What the bytes mean (format,
signature, models) is ``toolkit.core.services.ids.bundle``; saving, listing and loading runs is
``toolkit.core.services.ids.exports``.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Mapping
from pathlib import Path

from ..protect import protect
from ..settings import instance_dir

KEY_BYTES = 32


class BundleError(ValueError):
    """A bundle that is not loaded; ``code`` says why (e.g. ``BAD_SIGNATURE``)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def key_path(environ: Mapping[str, str] | None = None) -> Path:
    """``instance/signing.key``: this install's signing key."""
    return instance_dir(environ) / "signing.key"


def bundles_dir(environ: Mapping[str, str] | None = None) -> Path:
    """``instance/bundles``: the exported bundles."""
    return instance_dir(environ) / "bundles"


def signing_key(path: str | Path | None = None, *, create: bool = False) -> bytes:
    """This install's key; with ``create`` it is made on first use (and never replaced)."""
    path = Path(path) if path is not None else key_path()
    if not path.is_file():
        if not create:
            raise BundleError("NO_KEY", f"This install has no signing key ({path.name})")
        path.parent.mkdir(parents=True, exist_ok=True)
        protect(path.parent)  # HS5: the key inherits the folder's permissions
        try:
            with path.open("x", encoding="ascii", newline="\n") as handle:
                handle.write(secrets.token_bytes(KEY_BYTES).hex() + "\n")
        except FileExistsError:
            pass  # made at the same moment by another process: use that one
        else:
            try:
                os.chmod(path, 0o600)  # owner only (on Windows: the folder's ACL, protect)
            except OSError:
                pass
    try:
        key = bytes.fromhex(path.read_text(encoding="ascii").strip())
    except (UnicodeDecodeError, ValueError):
        key = b""
    if len(key) != KEY_BYTES:
        raise BundleError("BAD_KEY", f"{path.name} is not a signing key "
                          f"({KEY_BYTES * 2} hex digits)")
    return key


def new_bundle_path(run_id: str, directory: str | Path | None = None) -> Path:
    """Where ``<run id>.bundle`` goes; raises FileExistsError rather than replace one."""
    directory = Path(directory) if directory is not None else bundles_dir()
    path = directory / f"{run_id}.bundle"
    if path.exists():
        raise FileExistsError(path)
    return path


def write_new(path: Path, data: bytes) -> None:
    """Write a new file (FileExistsError if it appeared meanwhile)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def read_start(path: str | Path, size: int) -> bytes:
    """The first ``size`` bytes of a file (fewer if it is shorter)."""
    with Path(path).open("rb") as handle:
        return handle.read(size)


def read_all(path: str | Path) -> bytes:
    """A bundle file's bytes, read whole; they are checked (``ids.bundle.unpack``) before anything
    in them is deserialized."""
    return Path(path).read_bytes()


def stored_bundles(directory: str | Path | None = None) -> list[tuple[Path, int, float]]:
    """The ``*.bundle`` files in ``instance/bundles``: (path, size in bytes, mtime)."""
    directory = Path(directory) if directory is not None else bundles_dir()
    found = []
    for path in directory.glob("*.bundle") if directory.is_dir() else ():
        stat = path.stat()
        found.append((path, stat.st_size, stat.st_mtime))
    return found
