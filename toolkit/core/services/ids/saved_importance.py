"""Saved permutation importance results (v5.3 step HR2, RD4: IA2, IA5-IA7).

A result is published once as its own file beside its run's record and never replaced: a
training after it, or another run, leaves it alone. Its name comes from checked ids only (a
run id of the toolkit's form, a model id of the registry) before any path is made. It is
published under the one lock of the run files, and only while the run's record exists and
lists that model, so a run deleted while its importance computed gets nothing. It is read
back as untrusted data: its size, its schema (a newer one refused), its ids matching its
name, its values; a file that fails is left out and counted, never an error. Until every
training has a record (HR5), only an exported run's results can be saved.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...data.disk import NoSpace
from ...data.ids import importance_files
from ...data.ids.run_files import runs_dir
from ...data.settings import ExplainConfig
from .comparison import is_run_id
from .exports import read_run
from .importance import Importance
from .registry import MODELS
from .runs import package_versions

SCHEMA_VERSION = 1
METHOD = "permutation_importance"
METHOD_VERSION = 1  # a change of method gets a new version, never a silent replacement
NO_RECORD = "Not saved: the run isn't saved yet. Export it to keep its results."
NOT_WRITTEN = "Not saved: it could not be written (details in the server log)."

log = logging.getLogger(__name__)


class NotAnImportance(ValueError):
    """A run id or a model id that no result file can be named after."""


def publish(run_id: str, model: str, found: Importance, config: ExplainConfig | None = None,
            directory: str | Path | None = None) -> str | None:
    """Publish ``found`` as the result of ``model`` of ``run_id``: None once it is saved (or
    one was already, which is kept), else why not, for the user."""
    folder = _folder(run_id, model, directory)
    result = to_result(run_id, model, found, config or ExplainConfig())
    with importance_files.PUBLISHING:
        try:
            record = read_run(folder / f"{run_id}.json")
        except (OSError, ValueError):
            return NO_RECORD
        if model not in record_models(record):
            return f"Not saved: the run's record has no model {model}."
        try:
            importance_files.write_importance(folder, run_id, model, result)
        except FileExistsError:
            pass  # published before: never replaced
        except NoSpace as error:  # HR4c: the disk is too full; it says so
            return str(error)
        except OSError:
            log.exception("Could not save the importance of %s of run %s", model, run_id)
            return NOT_WRITTEN
    return None


def to_result(run_id: str, model: str, found: Importance, config: ExplainConfig) -> dict[str, Any]:
    """What a result file holds (IA2)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "run_id": run_id,
        "model": model,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": {"max_rows": config.importance_max_rows,
                     "min_per_class": config.importance_min_per_class,
                     "repeats": found.repeats, "seed": found.seed, "metric": "macro-F1"},
        "rows": found.rows,
        "test_rows": found.test_rows,
        "baseline_macro_f1": found.baseline_macro_f1,
        "features": [{"feature": f.feature, "mean": f.mean, "std": f.std}
                     for f in found.features],
        "concentrated": found.concentrated,
        "versions": package_versions(),
    }


def saved_importance(run_id: str, model: str,
                     directory: str | Path | None = None) -> dict[str, Any] | None:
    """The saved result of ``model`` of ``run_id``, or None when there is none or it is not
    one this toolkit can show."""
    path = importance_files.importance_path(_folder(run_id, model, directory), run_id, model)
    try:
        return _checked(path, run_id, model)
    except (OSError, ValueError):
        return None


def saved_importances(run_id: str, directory: str | Path | None = None) -> dict[str, Any]:
    """A run's saved results, by model id: ``{"importances", "left_out"}``, ``left_out``
    counting the files under its name that are not results this toolkit can show."""
    folder = _folder(run_id, None, directory)
    prefix = f"{run_id}{importance_files.INFIX}"
    found, left_out = [], 0
    for path in importance_files.run_importances(folder, run_id):
        model = path.name[len(prefix):-len(".json")]
        try:
            if model not in MODELS:
                raise ValueError("not a model id")
            found.append(_checked(path, run_id, model))
        except (OSError, ValueError):
            left_out += 1
    return {"importances": found, "left_out": left_out}


def record_models(record: dict[str, Any]) -> list[str]:
    """The model ids a run's record lists."""
    entries = record.get("evaluation", {}).get("models", [])
    return [entry["name"] for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)]


def _folder(run_id: str, model: str | None, directory: str | Path | None) -> Path:
    """The results' folder, once ``run_id`` (and ``model``, when given) are ids a file can be
    named after; NotAnImportance otherwise."""
    if not is_run_id(run_id) or (model is not None and model not in MODELS):
        raise NotAnImportance("That is not a run id and a model id.")
    return Path(directory) if directory is not None else runs_dir()


def _checked(path: Path, run_id: str, model: str) -> dict[str, Any]:
    """The result at ``path`` if it is one of ``model`` of ``run_id`` this toolkit can show;
    ValueError otherwise (OSError when it can't be read)."""
    result = importance_files.read_importance(path)
    if not isinstance(result, dict):
        raise ValueError("not a result")
    version = result.get("schema_version")
    if not isinstance(version, int) or version > SCHEMA_VERSION or result.get("method") != METHOD:
        raise ValueError("a newer or another kind of result")
    if (result.get("run_id"), result.get("model")) != (run_id, model):
        raise ValueError("not the result its name says")
    if not _values_ok(result):
        raise ValueError("its values are not a result's")
    return result


def _values_ok(result: dict[str, Any]) -> bool:
    """True when the values the pages show have their types: each feature named, finite
    numbers, the row counts whole, the concentrated feature a name or none."""
    features = result.get("features")
    if not isinstance(features, list) or not all(
            isinstance(item, dict) and isinstance(item.get("feature"), str) for item in features):
        return False
    numbers = [result.get("baseline_macro_f1"),
               *(item.get(key) for item in features for key in ("mean", "std"))]
    return (all(isinstance(n, (int, float)) and math.isfinite(n) for n in numbers)
            and all(isinstance(result.get(key), int) for key in ("rows", "test_rows"))
            and isinstance(result.get("concentrated"), (str, type(None))))
