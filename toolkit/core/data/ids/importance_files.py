"""Permutation importance results on disk (v5.3 step HR2, RD4: IA2, IA5, IA7): names and
bytes, nothing more.

Each result is ``instance/runs/<run id>.importance.<model id>.json``, beside its run's
record, published once and never replaced (``toolkit.core.data.publish``). The caller checks
both ids before a name is made (``toolkit.core.services.ids.saved_importance``). A file is
read only up to ``MAX_BYTES``. ``PUBLISHING`` is the one lock of the process that orders
publishing and deleting the run files (IA7): the record checked and the result published
under it, so a run deleted meanwhile gets nothing (the delete is HR4).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..publish import publish_text

INFIX = ".importance."
MAX_BYTES = 2**20  # one result: about 100 bytes a feature
PUBLISHING = threading.Lock()


def importance_path(directory: Path, run_id: str, model: str) -> Path:
    """``<directory>/<run id>.importance.<model id>.json`` (both ids already checked)."""
    return directory / f"{run_id}{INFIX}{model}.json"


def is_importance_name(name: str) -> bool:
    """True for a result's file name, which is never a run's record."""
    return INFIX in name


def write_importance(directory: Path, run_id: str, model: str,
                     result: Mapping[str, Any]) -> Path:
    """Publish a result; FileExistsError rather than replace one."""
    text = json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    return publish_text(importance_path(directory, run_id, model), text)


def read_importance(path: Path) -> Any:
    """A result file as JSON; ValueError when it is larger than ``MAX_BYTES`` or not JSON."""
    with path.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{path.name} is larger than {MAX_BYTES} bytes")
    return json.loads(data.decode("utf-8"))


def run_importances(directory: Path, run_id: str) -> list[Path]:
    """The result files of a run (its id already checked), sorted by name."""
    return sorted(directory.glob(f"{run_id}{INFIX}*.json")) if directory.is_dir() else []
