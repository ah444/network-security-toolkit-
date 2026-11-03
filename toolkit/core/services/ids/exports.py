"""Saving, listing and loading this install's runs and bundles (plan D11, D21).

The bytes come from and go to ``toolkit.core.data.ids`` (``bundle_files``, ``run_files``); what
they mean is ``bundle`` (format, signature, library versions) and ``runs`` (the TrainingRun
record). If the signing key is lost, the bundles saved on this install can no longer be
checked, so they are not loaded: back up ``instance/``, or retrain from the run's
TrainingRun record (``instance/runs/``). No model is shipped with the project.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...data.ids.bundle_files import (
    BundleError,
    new_bundle_path,
    read_all,
    read_start,
    signing_key,
    stored_bundles,
    write_new,
)
from ...data.ids.importance_files import PUBLISHING
from ...data.ids.run_files import read_record, write_record
from .bundle import EXPORTED as EXPORTED  # the states, for the callers
from .bundle import INCOMPLETE as INCOMPLETE
from .bundle import VERIFICATION_FAILED as VERIFICATION_FAILED
from .bundle import (
    HEADER_BYTES,
    Bundle,
    header_bytes,
    header_state,
    pack,
    peek_manifest,
    unpack,
)
from .runs import SCHEMA_VERSION, TrainingRun


@dataclass(frozen=True)
class SavedBundle:
    """A bundle just written: its path, its SHA-256 and the run that now records it."""
    path: Path
    sha256: str
    run: TrainingRun  # the run with ``bundle_sha256`` set; the caller saves its record


@dataclass(frozen=True)
class BundleInfo:
    """A bundle file as listed: its path, its size, its manifest, read without checking the
    signature, so for lists and messages only (never to load from), and its state (HR3d):
    EXPORTED, INCOMPLETE (then the manifest may be empty) or VERIFICATION_FAILED."""
    path: Path
    size: int  # bytes
    manifest: dict[str, Any]  # read without checking the signature: for lists and messages only
    state: str = EXPORTED


class BundleExists(FileExistsError):
    """A bundle of this run is already there and is never replaced; ``state`` is its state
    (INCOMPLETE: an interrupted export left it), None when it can't be read."""

    def __init__(self, path: Path, state: str | None):
        super().__init__(path)
        self.path, self.state = path, state


def save_bundle(run: TrainingRun, directory: str | Path | None = None, *,
                key_file: str | Path | None = None) -> SavedBundle:
    """Write ``<run id>.bundle``; raises BundleExists rather than replace one."""
    try:
        path = new_bundle_path(run.run_id, directory)
    except FileExistsError as error:
        raise _exists(Path(error.args[0])) from None
    data = pack(run, signing_key(key_file, create=True))
    try:
        write_new(path, data)
    except FileExistsError:
        raise _exists(path) from None  # it appeared meanwhile
    sha256 = hashlib.sha256(data).hexdigest()
    return SavedBundle(path, sha256, dataclasses.replace(run, bundle_sha256=sha256))


def read_state(path: str | Path) -> tuple[str | None, dict[str, Any] | None]:
    """A bundle file's state and manifest from its first bytes and its size (never its models,
    so listing multi-GB bundles is instant; no length read past the file's size).
    Unchecked: (None, None) if it isn't a bundle or can't be read."""
    try:
        size = Path(path).stat().st_size
        start = read_start(path, min(HEADER_BYTES, size))
        needed = min(header_bytes(start), size)
        if needed > len(start):
            start = read_start(path, needed)
        return header_state(start, size)
    except OSError:
        return None, None


def list_bundles(directory: str | Path | None = None) -> list[BundleInfo]:
    """The bundles in ``instance/bundles`` (this install's exports), newest first, the
    incomplete ones too (never deleted on their own)."""
    found = []
    for path, size, mtime in stored_bundles(directory):
        state, manifest = read_state(path)
        if state is not None:
            found.append((str((manifest or {}).get("created_at", "")), mtime,
                          BundleInfo(path, size, manifest or {}, state)))
    return [info for *_, info in sorted(found, key=lambda item: item[:2], reverse=True)]


def load_bundle(path: str | Path, *, key_file: str | Path | None = None,
                pinned_sha256: str | None = None) -> Bundle:
    """Load a bundle saved on this install (or, with ``pinned_sha256``, a shipped one)."""
    return open_bundle(read_all(path), key_file=key_file, pinned_sha256=pinned_sha256)


class BundleStore:
    """``toolkit.core.services.ports.ModelStore`` over this install's bundle files.

    ``directory`` and ``key_file`` default to ``instance/bundles`` and
    ``instance/signing.key``, resolved at each call (so ``TOOLKIT_INSTANCE_DIR`` applies).
    A file a load refused for its signature is listed as VERIFICATION_FAILED while the store
    lives (HR3d): the file itself is never changed (RD4).
    """

    def __init__(self, directory: str | Path | None = None, *,
                 key_file: str | Path | None = None):
        self.directory = directory
        self.key_file = key_file
        self._failed: set[tuple[str, int, int]] = set()

    def save(self, run: TrainingRun) -> SavedBundle:
        """``save_bundle`` in the store's folder, signed with its key."""
        return save_bundle(run, self.directory, key_file=self.key_file)

    def list_bundles(self) -> list[BundleInfo]:
        """``list_bundles`` of the store's folder, with the files that failed a load."""
        return [dataclasses.replace(info, state=VERIFICATION_FAILED)
                if info.state == EXPORTED and _identity(info.path) in self._failed else info
                for info in list_bundles(self.directory)]

    def load(self, path: str | Path) -> Bundle:
        """``load_bundle`` with the store's key: signed by this install, or refused."""
        try:
            return load_bundle(path, key_file=self.key_file)
        except BundleError as error:
            if error.code == "BAD_SIGNATURE":
                self._failed.add(_identity(path))
            raise

    def open(self, data: bytes) -> Bundle:
        """``open_bundle`` with the store's key (bytes already in memory)."""
        return open_bundle(data, key_file=self.key_file)


def open_bundle(data: bytes, *, key_file: str | Path | None = None,
                pinned_sha256: str | None = None) -> Bundle:
    """``load_bundle`` for bytes already in memory (e.g. a web upload). Error messages
    never contain paths, so they can be shown to the user."""
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


def _exists(path: Path) -> BundleExists:
    """The BundleExists of the file at ``path``, with its state."""
    return BundleExists(path, read_state(path)[0])


def _identity(path: str | Path) -> tuple[str, int, int]:
    """A file as it is now: its path, size and modification time (0, 0 once it is gone)."""
    try:
        stat = Path(path).stat()
        return str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns
    except OSError:
        return str(path), 0, 0


def write_run(run: TrainingRun, directory: str | Path | None = None) -> Path:
    """Write ``<run id>.json``; raises FileExistsError rather than replace a record. Under
    the one lock of the run files, so publishing it and deleting a run take turns (IA7)."""
    with PUBLISHING:
        return write_record(run.run_id, run.to_dict(), directory)


def read_run(path: str | Path) -> dict[str, Any]:
    """A TrainingRun record read back: ValueError if it is not one, or if its schema is newer than
    this toolkit understands."""
    record = read_record(path)
    if not isinstance(record, dict) or "schema_version" not in record or "run_id" not in record:
        raise ValueError(f"{Path(path).name} is not a TrainingRun record")
    version = record["schema_version"]
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise ValueError(f"{Path(path).name} uses TrainingRun schema {version}, newer than this "
                         f"version of the toolkit understands ({SCHEMA_VERSION}); update it")
    return record
