"""The scan history on disk (F6a): one record per finished scan, as a dict.

Records live in ``instance/scan_history/<scan id>.json`` (gitignored; ``TOOLKIT_INSTANCE_DIR``
moves ``instance/``), apart from the reports the user saves in ``instance/scans/``. A record is
written once and never overwritten, and is published whole or not at all
(``toolkit.core.data.publish``, HR3a). What a record holds is
``toolkit.core.services.recon.history``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..publish import publish_text
from ..settings import instance_dir


def history_dir(environ: Mapping[str, str] | None = None) -> Path:
    """``instance/scan_history``: the scans' records."""
    return instance_dir(environ) / "scan_history"


def write_scan(scan_id: str, record: Mapping[str, Any],
               directory: str | Path | None = None) -> Path:
    """Write ``<scan id>.json``; raises FileExistsError rather than replace a record, NoSpace
    when the disk is (too) full (HR4c)."""
    directory = Path(directory) if directory is not None else history_dir()
    text = json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    return publish_text(directory / f"{scan_id}.json", text)


def read_scan(path: str | Path) -> Any:
    """A record as JSON (F6b); ``saved_scans`` checks what it is."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_scans(directory: str | Path | None = None) -> list[Path]:
    """The ``*.json`` files of ``directory`` (default ``history_dir``), sorted by name."""
    directory = Path(directory) if directory is not None else history_dir()
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def remove_file(path: Path) -> None:
    """Removes a scan's record (F6b); nothing when it is already gone. Any other OSError is
    raised (on Windows: a file another program has open)."""
    path.unlink(missing_ok=True)
