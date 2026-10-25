"""The plain-JSON form of a ``TrainingRun`` (plan D21; moved out of ``runs.py`` in v5.2
step 2.3h).

``TrainingRun.to_dict()`` builds the record from these: ``jsonable`` turns any value
into JSON types (dataclasses with their properties, NumPy values, NaN and infinity as
``null``), the ``*_dict`` helpers pick what the record keeps of the profile, the split,
the training and each fitted pipeline. Nothing here writes a file.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import TrainingOutcome
from .profiles import ProfileMatch
from .split import SplitResult


# C901 kept: one isinstance test per kind of value, in order; a dispatch table would only
# hide the order
def jsonable(value: Any) -> Any:  # noqa: C901
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        data = {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
        for name, attribute in vars(type(value)).items():  # e.g. CVSummary.macro_f1_mean
            if isinstance(attribute, property):
                data[name] = jsonable(getattr(value, name))
        return data
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, type):
        return value.__name__
    return repr(value)


def profile_dict(match: ProfileMatch | None) -> dict[str, Any] | None:
    if match is None:
        return None
    return {
        "name": match.profile.name, "title": match.profile.title, "variant": match.variant,
        "has_header": match.has_header, "columns": len(match.column_names),
        "label_column": match.label_column, "blocked_columns": list(match.blocked_columns),
        "split_strategy": match.split_strategy, "time_column": match.time_column,
        "group_column": match.group_column, "warnings": list(match.warnings),
    }


def split_dict(split: SplitResult) -> dict[str, Any]:
    data = {f.name: jsonable(getattr(split, f.name)) for f in fields(split)
            if f.name not in ("train", "test")}
    data["train_rows"], data["test_rows"] = len(split.train), len(split.test)
    return data


def training_dict(outcome: TrainingOutcome) -> dict[str, Any]:
    return {
        "train_rows": outcome.train_rows, "classes": list(outcome.classes),
        "class_counts": jsonable(outcome.class_counts),
        "numeric_features": list(outcome.numeric_features),
        "categorical_features": list(outcome.categorical_features),
        "failures": jsonable(outcome.failures), "skipped": jsonable(outcome.skipped),
        "notes": list(outcome.notes), "seed": outcome.seed, "workers": outcome.workers,
    }


def models_dict(outcome: TrainingOutcome) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name, fitted in outcome.models.items():
        steps = fitted.pipeline.named_steps
        numeric = "inf → missing, median imputation"
        if fitted.scaled:
            numeric += ", signed log, standard scaling"
        if not outcome.categorical_features:
            categorical = "none"
        elif fitted.encoding == "onehot":
            categorical = "one-hot (unknown categories ignored)"
        else:
            categorical = "ordinal (unknown → -1, missing → -2)"
        models[name] = {
            "steps": [step for step, _ in fitted.pipeline.steps],
            "numeric": numeric if outcome.numeric_features else "none",
            "categorical": categorical,
            "select": _step_dict(steps["select"]),
            "sample": _step_dict(steps["sample"]),
            "estimator": type(steps["model"]).__name__,
            "params": jsonable(steps["model"].get_params(deep=False)),
            "fit_seconds": fitted.fit_seconds,
            "notes": list(fitted.notes),
        }
    return models


def _step_dict(step: Any) -> dict[str, Any] | None:
    if step is None or step == "passthrough":
        return None
    return {"class": type(step).__name__, "params": jsonable(step.get_params(deep=False))}
