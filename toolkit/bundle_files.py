"""Signing key and bundle files (edge code for ``toolkit.ids_core.bundle``, plan D11).

The key is created once, on the first save, in ``instance/signing.key`` (gitignored;
``TOOLKIT_INSTANCE_DIR`` moves ``instance/``). Loading never creates a key. If the key is
lost, the bundles saved on this install can no longer be checked, so they are not loaded:
back up ``instance/``, or retrain from the run's TrainingRun record (``instance/runs/``).
Bundles live in ``instance/bundles/<run id>.bundle`` and are never overwritten.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from .config import instance_dir
from .ids_core.bundle import KEY_BYTES, Bundle, BundleError, pack, peek_manifest, unpack
from .ids_core.runs import TrainingRun


@dataclass(frozen=True)
class SavedBundle:
    path: Path
    sha256: str
    run: TrainingRun  # the run with ``bundle_sha256`` set; the caller saves its record


def key_path(environ=None) -> Path:
    return instance_dir(environ) / "signing.key"


def bundles_dir(environ=None) -> Path:
    return instance_dir(environ) / "bundles"


def signing_key(path: str | Path | None = None, *, create: bool = False) -> bytes:
    """This install's key; with ``create`` it is made on first use (and never replaced)."""
    path = Path(path) if path is not None else key_path()
    if not path.is_file():
        if not create:
            raise BundleError("NO_KEY", f"No signing key at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="ascii", newline="\n") as handle:
                handle.write(secrets.token_bytes(KEY_BYTES).hex() + "\n")
        except FileExistsError:
            pass  # made at the same moment by another process: use that one
        else:
            try:
                os.chmod(path, 0o600)  # owner only (Windows keeps its own ACLs)
            except OSError:
                pass
    try:
        key = bytes.fromhex(path.read_text(encoding="ascii").strip())
    except (UnicodeDecodeError, ValueError):
        key = b""
    if len(key) != KEY_BYTES:
        raise BundleError("BAD_KEY", f"{path} is not a signing key ({KEY_BYTES * 2} hex digits)")
    return key


def save_bundle(run: TrainingRun, directory: str | Path | None = None, *,
                key_file: str | Path | None = None) -> SavedBundle:
    """Write ``<run id>.bundle``; raises FileExistsError rather than replace one."""
    directory = Path(directory) if directory is not None else bundles_dir()
    path = directory / f"{run.run_id}.bundle"
    if path.exists():
        raise FileExistsError(path)
    data = pack(run, signing_key(key_file, create=True))
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    sha256 = hashlib.sha256(data).hexdigest()
    return SavedBundle(path, sha256, dataclasses.replace(run, bundle_sha256=sha256))


def load_bundle(path: str | Path, *, key_file: str | Path | None = None,
                pinned_sha256: str | None = None) -> Bundle:
    """Load a bundle saved on this install (or, with ``pinned_sha256``, a shipped one)."""
    data = Path(path).read_bytes()
    if pinned_sha256 is not None:
        return unpack(data, pinned_sha256=pinned_sha256)
    try:
        key = signing_key(key_file)
    except BundleError as error:
        if error.code != "NO_KEY":
            raise
        manifest = peek_manifest(data) or {}  # unchecked: only used to name the run
        run_id = manifest.get("run_id")
        run_id = run_id if isinstance(run_id, str) and re.fullmatch(r"[\w.-]{1,64}", run_id) \
            else None
        record = f"instance/runs/{run_id}.json" if run_id else "instance/runs/"
        raise BundleError("NO_KEY", f"{error} — the signing key is missing, so bundles saved on "
                          "this install can no longer be checked and are not loaded. Retrain "
                          f"from its TrainingRun ({f'run {run_id}, ' if run_id else ''}{record}) "
                          "to make a new bundle. Keep a backup copy of the instance/ folder "
                          "to avoid this.") from None
    return unpack(data, key=key)
