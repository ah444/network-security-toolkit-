"""The storage report (v5.3 step HR4b, RD4: IA9): every file of the instance, by kind.

Each file is counted once, with its size: the saved runs' records, their importance results
and bundles, the scan reports, the uploads being read, the logs and the rest; then the total
and the free space of the disk. The leftovers are named: an importance result without its
record or for a model the record doesn't list, a bundle whose run was deleted (unregistered:
it can be registered again), an incomplete bundle, a damaged file (or one of a newer
version), a temporary file a stop left. It only reads: **nothing is ever deleted here**, and
a bundle is never unpickled (its state comes from its header). It warns (HR4c) past
``TOOLKIT_STORAGE_WARN_MB`` and under ``TOOLKIT_MIN_FREE_MB`` of free space, where new files
are refused (``NoSpace``); nothing is deleted on their account either.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..data import disk
from ..data.disk import MB, files_under
from ..data.disk import NoSpace as NoSpace  # what a refused new file raises, for the callers
from ..data.ids.bundle_files import bundles_dir
from ..data.ids.importance_files import INFIX
from ..data.ids.run_files import runs_dir
from ..data.publish import LEFTOVER
from ..data.recon.scan_files import scans_dir
from ..data.settings import instance_dir, min_free_mb, storage_warn_mb
from .ids.comparison import is_run_id
from .ids.exports import EXPORTED, INCOMPLETE, read_run, read_state
from .ids.registry import MODELS
from .ids.saved_importance import record_models

KINDS = (
    ("records", "Saved runs' records"),
    ("importance", "Importance results"),
    ("orphan_importance", "Importance results without their run (or for a model it doesn't have)"),
    ("bundles", "Bundles of saved runs"),
    ("unregistered_bundles", "Unregistered bundles: their run was deleted (Register again "
                             "brings it back)"),
    ("incomplete_bundles", "Incomplete bundles (an export was interrupted)"),
    ("damaged", "Damaged files, or from a newer version (not read)"),
    ("temporary", "Temporary files a stop left (cleared when the web server starts)"),
    ("scans", "Scan reports"),
    ("uploads", "Uploads being read"),
    ("logs", "Logs"),
    ("other", "Other files (the signing key, the jobs' journal)"),
)
LEFTOVERS = ("orphan_importance", "unregistered_bundles", "incomplete_bundles", "damaged",
             "temporary")


def storage_report() -> dict[str, Any]:
    """``{"kinds": [{"kind", "title", "count", "bytes", "files", "leftover"}],
    "total_bytes", "free_bytes", "warn_bytes", "floor_bytes", "warnings"}``; ``files`` names
    the leftovers (never a path)."""
    found: dict[str, list[tuple[str, int]]] = {kind: [] for kind, _title in KINDS}
    runs, bundles = runs_dir(), bundles_dir()
    models = _records(runs)
    sorts: dict[Path, Callable[[Path], str]] = {
        runs: lambda path: _in_runs(path, models),
        bundles: lambda path: _in_bundles(path, models),
        scans_dir(): lambda _path: "scans", instance_dir() / "uploads": lambda _path: "uploads",
        instance_dir() / "logs": lambda _path: "logs"}
    for path, size in files_under(instance_dir()):
        sort = sorts.get(path.parent)
        found[sort(path) if sort is not None else "other"].append((path.name, size))
    kinds = [{"kind": kind, "title": title, "count": len(found[kind]),
              "bytes": sum(size for _name, size in found[kind]),
              "files": sorted(name for name, _size in found[kind]) if kind in LEFTOVERS else [],
              "leftover": kind in LEFTOVERS} for kind, title in KINDS]
    total = sum(size for files in found.values() for _name, size in files)
    free = disk.free_bytes(instance_dir())
    warn, floor = storage_warn_mb() * MB, min_free_mb() * MB
    return {"kinds": kinds, "total_bytes": total, "free_bytes": free, "warn_bytes": warn,
            "floor_bytes": floor, "warnings": warnings(total, free, warn, floor)}


def warnings(total: int, free: int, warn: int, floor: int) -> list[str]:
    """What the report warns of (HR4c): the instance past its size, the disk under its floor."""
    found = []
    if total > warn:
        found.append(f"The toolkit's files take {_size(total)}, more than the {_size(warn)} set "
                     "in TOOLKIT_STORAGE_WARN_MB. Nothing is deleted on its own: delete the runs "
                     "you no longer need on the Saved Runs page (or with toolkit ids --delete).")
    if free < floor:
        found.append(f"Only {_size(free)} is free on this disk, under the {_size(floor)} a new "
                     "file must leave (TOOLKIT_MIN_FREE_MB): new runs, bundles, importance "
                     "results and scan reports are not saved until space is freed.")
    return found


def _size(size: int) -> str:
    """A size as the pages write it: "2.0 MB", "1.0 GB"."""
    for unit, bytes_ in (("GB", 2**30), ("MB", MB), ("KB", 2**10)):
        if size >= bytes_:
            return f"{size / bytes_:,.1f} {unit}"
    return f"{size} bytes"


def _records(runs: Path) -> dict[str, list[str] | None]:
    """Each record's run id and its models; None for a record that isn't readable."""
    found: dict[str, list[str] | None] = {}
    for path, _size in files_under(runs):
        if path.suffix == ".json" and INFIX not in path.name and is_run_id(path.stem):
            try:
                record = read_run(path)
                found[path.stem] = record_models(record) if record["run_id"] == path.stem \
                    else None
            except (OSError, ValueError, AttributeError):
                found[path.stem] = None
    return found


def _in_runs(path: Path, models: dict[str, list[str] | None]) -> str:
    """The kind of a file in ``instance/runs``."""
    name = path.name
    if LEFTOVER.fullmatch(name):
        return "temporary"
    if INFIX in name:
        run_id, _, model = name.removesuffix(".json").partition(INFIX)
        owned = name.endswith(".json") and model in MODELS and model in (models.get(run_id) or ())
        return "importance" if owned else "orphan_importance"
    return "records" if models.get(path.stem) is not None and path.suffix == ".json" \
        else "damaged"


def _in_bundles(path: Path, models: dict[str, list[str] | None]) -> str:
    """The kind of a file in ``instance/bundles`` (from its header: nothing unpickled)."""
    if LEFTOVER.fullmatch(path.name):
        return "temporary"
    if path.suffix != ".bundle":
        return "other"
    state, _manifest = read_state(path)
    if state == INCOMPLETE:
        return "incomplete_bundles"
    if state != EXPORTED or not is_run_id(path.stem):
        return "damaged"
    return "bundles" if path.stem in models else "unregistered_bundles"
