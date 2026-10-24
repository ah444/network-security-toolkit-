"""Saving, listing and loading this install's runs and bundles (plan D11, D21).

The bytes come from and go to ``toolkit.data.ids`` (``bundle_files``, ``run_files``); what
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

from ..data.ids.bundle_files import (
    BundleError,
    new_bundle_path,
    read_all,
    read_start,
    signing_key,
    stored_bundles,
    write_new,
)
from ..data.ids.run_files import read_record, write_record
from .bundle import (
    HEADER_BYTES,
    Bundle,
    manifest_length,
    pack,
    parse_manifest,
    peek_manifest,
    unpack,
)
from .runs import SCHEMA_VERSION, TrainingRun


@dataclass(frozen=True)
class SavedBundle:
    path: Path
    sha256: str
    run: TrainingRun  # the run with ``bundle_sha256`` set; the caller saves its record


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    size: int  # bytes
    manifest: dict[str, Any]  # read without checking the signature: for lists and messages only


def save_bundle(run: TrainingRun, directory: str | Path | None = None, *,
                key_file: str | Path | None = None) -> SavedBundle:
    """Write ``<run id>.bundle``; raises FileExistsError rather than replace one."""
    path = new_bundle_path(run.run_id, directory)
    data = pack(run, signing_key(key_file, create=True))
    write_new(path, data)
    sha256 = hashlib.sha256(data).hexdigest()
    return SavedBundle(path, sha256, dataclasses.replace(run, bundle_sha256=sha256))


def read_manifest_header(path: str | Path) -> dict[str, Any] | None:
    """A bundle file's manifest, reading only the start of the file (never its models, so
    listing multi-GB bundles is instant). Unchecked: None if it isn't a bundle."""
    try:
        length = manifest_length(read_start(path, HEADER_BYTES))
        if length is None:
            return None
        return parse_manifest(read_start(path, HEADER_BYTES + length)[HEADER_BYTES:])
    except OSError:
        return None


def list_bundles(directory: str | Path | None = None) -> list[BundleInfo]:
    """The bundles in ``instance/bundles`` (this install's exports), newest first."""
    found = []
    for path, size, mtime in stored_bundles(directory):
        manifest = read_manifest_header(path)
        if manifest is not None:
            found.append((str(manifest.get("created_at", "")), mtime,
                          BundleInfo(path, size, manifest)))
    return [info for *_, info in sorted(found, key=lambda item: item[:2], reverse=True)]


def load_bundle(path: str | Path, *, key_file: str | Path | None = None,
                pinned_sha256: str | None = None) -> Bundle:
    """Load a bundle saved on this install (or, with ``pinned_sha256``, a shipped one)."""
    return open_bundle(read_all(path), key_file=key_file, pinned_sha256=pinned_sha256)


class BundleStore:
    """``toolkit.ports.ModelStore`` over this install's bundle files.

    ``directory`` and ``key_file`` default to ``instance/bundles`` and
    ``instance/signing.key``, resolved at each call (so ``TOOLKIT_INSTANCE_DIR`` applies).
    """

    def __init__(self, directory: str | Path | None = None, *,
                 key_file: str | Path | None = None):
        self.directory = directory
        self.key_file = key_file

    def save(self, run: TrainingRun) -> SavedBundle:
        return save_bundle(run, self.directory, key_file=self.key_file)

    def list_bundles(self) -> list[BundleInfo]:
        return list_bundles(self.directory)

    def load(self, path: str | Path) -> Bundle:
        return load_bundle(path, key_file=self.key_file)

    def open(self, data: bytes) -> Bundle:
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


def write_run(run: TrainingRun, directory: str | Path | None = None) -> Path:
    """Write ``<run id>.json``; raises FileExistsError rather than replace a record."""
    return write_record(run.run_id, run.to_dict(), directory)


def read_run(path: str | Path) -> dict[str, Any]:
    record = read_record(path)
    if not isinstance(record, dict) or "schema_version" not in record or "run_id" not in record:
        raise ValueError(f"{Path(path).name} is not a TrainingRun record")
    version = record["schema_version"]
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise ValueError(f"{Path(path).name} uses TrainingRun schema {version}, newer than this "
                         f"version of the toolkit understands ({SCHEMA_VERSION}); update it")
    return record
