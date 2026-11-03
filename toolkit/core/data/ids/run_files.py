"""TrainingRun records on disk, as dicts (the ML core itself never writes files).

Runs live in ``instance/runs/<run id>.json`` (gitignored; ``TOOLKIT_INSTANCE_DIR`` moves
``instance/``). A record is written once and never overwritten, and is published whole or
not at all (``toolkit.core.data.publish``, HR3a). Which schema versions
are understood is ``toolkit.core.services.ids.exports.read_run``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..publish import publish_text
from ..settings import instance_dir


def runs_dir(environ: Mapping[str, str] | None = None) -> Path:
    """``instance/runs``: the TrainingRun records."""
    return instance_dir(environ) / "runs"


def write_record(run_id: str, record: Mapping[str, Any],
                 directory: str | Path | None = None) -> Path:
    """Write ``<run id>.json``; raises FileExistsError rather than replace a record."""
    directory = Path(directory) if directory is not None else runs_dir()
    text = json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    return publish_text(directory / f"{run_id}.json", text)


def read_record(path: str | Path) -> Any:
    """A record as JSON; ``read_run`` checks what it is."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_runs(directory: str | Path | None = None) -> list[Path]:
    """The ``*.json`` files of ``directory`` (default ``runs_dir``), sorted by name."""
    directory = Path(directory) if directory is not None else runs_dir()
    return sorted(directory.glob("*.json")) if directory.is_dir() else []
