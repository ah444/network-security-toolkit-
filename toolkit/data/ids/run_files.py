"""TrainingRun records on disk, as dicts (the ML core itself never writes files).

Runs live in ``instance/runs/<run id>.json`` (gitignored; ``TOOLKIT_INSTANCE_DIR`` moves
``instance/``). A record is written once and never overwritten. Which schema versions
are understood is ``toolkit.ids_core.exports.read_run``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..settings import instance_dir


def runs_dir(environ: Mapping[str, str] | None = None) -> Path:
    return instance_dir(environ) / "runs"


def write_record(run_id: str, record: Mapping[str, Any],
                 directory: str | Path | None = None) -> Path:
    """Write ``<run id>.json``; raises FileExistsError rather than replace a record."""
    directory = Path(directory) if directory is not None else runs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=1, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


def read_record(path: str | Path) -> Any:
    """A record as JSON; ``read_run`` checks what it is."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_runs(directory: str | Path | None = None) -> list[Path]:
    directory = Path(directory) if directory is not None else runs_dir()
    return sorted(directory.glob("*.json")) if directory.is_dir() else []
