"""Save and read TrainingRun records (edge code: the ML core itself never writes files).

Runs live in ``instance/runs/<run id>.json`` (gitignored; ``TOOLKIT_INSTANCE_DIR`` moves
``instance/``). A record is written once and never overwritten. Readers accept every
schema version up to the current one and refuse newer ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import instance_dir
from .ids_core.runs import SCHEMA_VERSION, TrainingRun


def runs_dir(environ=None) -> Path:
    return instance_dir(environ) / "runs"


def write_run(run: TrainingRun, directory: str | Path | None = None) -> Path:
    """Write ``<run id>.json``; raises FileExistsError rather than replace a record."""
    directory = Path(directory) if directory is not None else runs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.run_id}.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(run.to_dict(), handle, indent=1, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


def read_run(path: str | Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or "schema_version" not in record or "run_id" not in record:
        raise ValueError(f"{Path(path).name} is not a TrainingRun record")
    version = record["schema_version"]
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise ValueError(f"{Path(path).name} uses TrainingRun schema {version}, newer than this "
                         f"version of the toolkit understands ({SCHEMA_VERSION}); update it")
    return record


def list_runs(directory: str | Path | None = None) -> list[Path]:
    directory = Path(directory) if directory is not None else runs_dir()
    return sorted(directory.glob("*.json")) if directory.is_dir() else []
