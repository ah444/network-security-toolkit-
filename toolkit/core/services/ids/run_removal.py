"""Deleting a saved run (v5.3 step HR4a, RD4: IA7, IA8).

A run's files go in one order: its importance results first, then its bundle when the user
chose to remove it, then its record last, so a file is never left without the record that
owns it unless the user kept the bundle on purpose (then it is an unregistered bundle). If a
file can't be removed (on Windows: another program has it open), the delete stops there and
the record stays: the run is still listed, and the delete can be tried again. It all happens
under the one lock of the run files (``importance_files.PUBLISHING``), so an importance result
computed meanwhile finds no record and is not published. The run id is checked before any
path is made from it; the record only has to exist, so a damaged one can be deleted too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ...data.ids import importance_files
from ...data.ids.bundle_files import bundle_path
from ...data.ids.run_files import file_size, remove_file, runs_dir
from .comparison import NotARun, is_run_id

IMPORTANCE, BUNDLE, RECORD = "importance", "bundle", "record"  # the kinds, in the order they go

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunFile:
    """One of a run's files: its kind, where it is and its size in bytes."""
    kind: str
    path: Path
    size: int

    @property
    def name(self) -> str:
        """The file's name (what the user is shown: never its path)."""
        return self.path.name

    def to_dict(self) -> dict[str, str | int]:
        """``{"kind", "name", "size"}``."""
        return {"kind": self.kind, "name": self.name, "size": self.size}


@dataclass(frozen=True)
class Removal:
    """What a delete did: the files removed, in order; the bundle kept on purpose; the file
    it stopped at (then the record is still there)."""
    run_id: str
    removed: tuple[RunFile, ...]
    kept: RunFile | None
    failed: RunFile | None


def run_files(run_id: str, runs: str | Path | None = None,
              bundles: str | Path | None = None) -> list[RunFile]:
    """The files a delete of ``run_id`` removes, in the order they go: NotARun for a bad id,
    FileNotFoundError when the run has no record (its other files are the storage report's)."""
    if not is_run_id(run_id):
        raise NotARun("That is not a run id.")
    folder = Path(runs) if runs is not None else runs_dir()
    record = folder / f"{run_id}.json"
    record_size = file_size(record)
    if record_size is None:
        raise FileNotFoundError(run_id)
    found = [RunFile(IMPORTANCE, path, size)
             for path in importance_files.run_importances(folder, run_id)
             if (size := file_size(path)) is not None]
    bundle = bundle_path(run_id, bundles)
    if (size := file_size(bundle)) is not None:
        found.append(RunFile(BUNDLE, bundle, size))
    return [*found, RunFile(RECORD, record, record_size)]


def remove_run(run_id: str, with_bundle: bool, runs: str | Path | None = None,
               bundles: str | Path | None = None) -> Removal:
    """Delete ``run_id``'s files in order, its bundle only ``with_bundle``; the same errors as
    ``run_files``. A file that can't be removed stops the delete (``Removal.failed``)."""
    with importance_files.PUBLISHING:
        removed: list[RunFile] = []
        kept = None
        for item in run_files(run_id, runs, bundles):
            if item.kind == BUNDLE and not with_bundle:
                kept = item
                continue
            try:
                remove_file(item.path)
            except OSError:
                log.warning("Could not remove %s of run %s", item.name, run_id, exc_info=True)
                return Removal(run_id, tuple(removed), kept, item)
            removed.append(item)
        log.info("Deleted run %s: %d file(s), the bundle %s", run_id, len(removed),
                 "kept" if kept else "removed or none")
        return Removal(run_id, tuple(removed), kept, None)


def not_removed(removal: Removal) -> str:
    """What a delete that stopped tells the user (a file's name, never its path)."""
    assert removal.failed is not None
    return (f"{removal.failed.name} could not be removed: another program may have it open. "
            "The run is kept and still listed; close that program, then delete the run again.")
