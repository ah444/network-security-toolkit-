"""Registering a kept bundle again (v5.3 step HR4b, RD4).

A run deleted with its bundle kept leaves an unregistered bundle. The bundle carries the copy
of its run's record it was signed with, and records never change, so the record can be
published again from it: the whole file's signature is checked first, read in pieces (a
bundle can be gigabytes) and with nothing in it deserialized; then the copy must be this
run's record (its run id, a schema this toolkit understands, the bundle's models among its
own), and with the bundle's SHA-256 it is the record as it was, byte for byte. It is
published under the one lock of the run files and never replaces a record. Nothing is
deleted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...data.ids import bundle_files
from ...data.ids.importance_files import PUBLISHING
from ...data.ids.run_files import runs_dir, write_record
from .bundle import signature_matches
from .comparison import NotARun, is_run_id
from .exports import EXPORTED, check_record, read_state
from .saved_importance import record_models

log = logging.getLogger(__name__)

ALREADY_SAVED = "This run is already saved: its record is in instance/runs."


class NotRegistered(bundle_files.BundleError):
    """A bundle whose run can't be registered again (``message`` says why, for the user)."""

    def __init__(self, message: str):
        super().__init__("NOT_REGISTERED", message)


def register_bundle(run_id: str, runs: str | Path | None = None,
                    bundles: str | Path | None = None,
                    key_file: str | Path | None = None) -> Path:
    """Publish the record ``run_id``'s kept bundle carries; the record's path. NotARun for a
    bad id, FileNotFoundError without a bundle, FileExistsError when the run is saved,
    BundleError (``BAD_SIGNATURE``, or NotRegistered) when the bundle can't give it."""
    if not is_run_id(run_id):
        raise NotARun("That is not a run id.")
    path = bundle_files.bundle_path(run_id, bundles)
    folder = Path(runs) if runs is not None else runs_dir()
    if not path.is_file():
        raise FileNotFoundError(run_id)
    if (folder / f"{run_id}.json").exists():
        raise FileExistsError(ALREADY_SAVED)
    state, manifest = read_state(path)
    if state is None or manifest is None:
        raise NotRegistered(f"{path.name} is not a bundle this toolkit can read.")
    if state != EXPORTED:
        raise NotRegistered(f"{path.name} is incomplete: its export was interrupted, so it "
                            "holds no whole record.")
    size = path.stat().st_size
    matches, sha256 = signature_matches(bundle_files.read_chunks(path), size,
                                        bundle_files.signing_key(key_file))
    if not matches:
        raise bundle_files.BundleError(
            "BAD_SIGNATURE", "The bundle's signature does not match this install's signing "
            "key: it was changed after it was saved, or it was made on another install. "
            "Nothing was registered.")
    record = _record(manifest, run_id)
    record["bundle_sha256"] = sha256
    with PUBLISHING:
        try:
            published = write_record(run_id, record, folder)
        except FileExistsError:
            raise FileExistsError(ALREADY_SAVED) from None
    log.info("Registered run %s again from its bundle", run_id)
    return published


def _record(manifest: dict[str, Any], run_id: str) -> dict[str, Any]:
    """The record the (signed) manifest carries, checked as this run's."""
    if "run" not in manifest:
        raise NotRegistered("This bundle holds no copy of its run's record (it was exported "
                            "by an older version), so it can't be registered again.")
    try:
        record = check_record(manifest["run"], "The bundle's record")
        models = manifest.get("models")
        if (record["run_id"] != run_id or manifest.get("run_id") != run_id
                or not isinstance(models, list)
                or not set(models) <= set(record_models(record))):
            raise ValueError("another run's record")
    except (ValueError, TypeError, AttributeError) as error:
        raise NotRegistered("The copy of the record in this bundle is not one this toolkit "
                            "can save as this run's record.") from error
    return dict(record)
