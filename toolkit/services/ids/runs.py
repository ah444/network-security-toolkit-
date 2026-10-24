"""TrainingRun: one immutable record of how a model was made (plan D21, step 0.4e).

``train(frame, …)`` runs the whole core (split → leakage guard → evaluate) and returns a
``TrainingRun``: the fitted models plus everything needed to explain or repeat them
(dataset SHA-256, profile, split and its reason codes, guard decisions and overrides,
features, preprocessing, balancing, model parameters, ``EvaluationResult``, package
versions, re-run tolerance). ``to_dict()`` is the plain-JSON form; the caller saves it
(``exports``, over ``toolkit.data.ids.run_files``), the core never writes files. ``rerun(record, frame)`` trains
again with the recorded settings and ``compare_metrics`` checks the result.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ...data.settings import (
    EvaluationConfig,
    LeakageConfig,
    PipelineConfig,
    SplitConfig,
    max_workers,
)
from ..ports import NullProgress, ProgressSink
from . import registry
from .datasets import LoadedDataset
from .evaluation import EvaluationResult, evaluate
from .leakage import LeakageReport, check_leakage
from .pipeline import TrainingOutcome
from .profiles import ProfileMatch
from .split import SplitResult, split_dataset

SCHEMA_VERSION = 1
VERSIONED_PACKAGES = ("scikit-learn", "numpy", "pandas", "scipy", "imbalanced-learn", "xgboost",
                      "lightgbm", "joblib", "threadpoolctl")
_CONFIGS = {"split": SplitConfig, "leakage": LeakageConfig, "pipeline": PipelineConfig,
            "evaluation": EvaluationConfig}


@dataclass(frozen=True)
class DatasetRef:
    """Which file a run used: the SHA-256 is what ties a run to its data."""

    name: str
    sha256: str
    rows: int
    columns: int

    @classmethod
    def from_loaded(cls, name: str, loaded: LoadedDataset) -> DatasetRef:
        return cls(name, loaded.report.sha256, loaded.report.rows, loaded.report.columns)


@dataclass(frozen=True)
class TrainSettings:
    """Everything the user chose for a run (defaults come from the profile and config)."""

    label: str | None = None  # default: the profile's label column
    strategy: str | None = None  # split strategy; default: the profile's
    time_column: str | None = None
    group_column: str | None = None
    features: tuple[str, ...] | None = None  # candidates; default: every other column
    allow: Mapping[str, str] = field(default_factory=dict)  # leakage overrides: column -> reason
    models: tuple[str, ...] | None = None  # default: registry.default_models()
    balancer: str = "none"  # or "recommended"
    selector: str = "none"
    k_features: int | None = None
    split: SplitConfig = field(default_factory=SplitConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    workers: int | None = None  # default: max_workers(); the record keeps the number used

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainSettings:
        values = dict(data)
        for key, config in _CONFIGS.items():
            values[key] = config(**values[key])
        for key in ("features", "models"):
            if values[key] is not None:
                values[key] = tuple(values[key])
        values["allow"] = dict(values["allow"])
        return cls(**values)


@dataclass(frozen=True, eq=False)
class TrainingRun:
    run_id: str
    created_at: str  # UTC, ISO 8601
    settings: TrainSettings  # as used (workers resolved)
    dataset: DatasetRef | None
    test_dataset: DatasetRef | None  # official split: the separate test file
    profile: ProfileMatch | None
    split: SplitResult
    leakage: LeakageReport
    evaluation: EvaluationResult  # includes the fitted models (``evaluation.training``)
    versions: dict[str, str | None]
    bundle_sha256: str | None = None  # set when the run is saved as a bundle (0.4f)
    schema_version: int = SCHEMA_VERSION

    @property
    def metric_tolerance(self) -> float:
        return 0.0 if self.settings.workers == 1 else self.settings.evaluation.parallel_metric_tolerance

    def to_dict(self) -> dict[str, Any]:
        """The record as plain JSON types (no fitted objects, no NaN)."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "dataset": _jsonable(self.dataset),
            "test_dataset": _jsonable(self.test_dataset),
            "profile": _profile_dict(self.profile),
            "settings": _jsonable(self.settings),
            "split": _split_dict(self.split),
            "leakage": _jsonable(self.leakage),
            "features": list(self.evaluation.training.features),
            "balancing": {
                "requested": self.settings.balancer,
                "method": self.evaluation.balancer,
                "recommendation": _jsonable(self.evaluation.balancer_recommendation),
            },
            "selection": {"selector": self.evaluation.selector,
                          "k_features": self.evaluation.k_features},
            "training": _training_dict(self.evaluation.training),
            "models": _models_dict(self.evaluation.training),
            "evaluation": {f.name: _jsonable(getattr(self.evaluation, f.name))
                           for f in fields(self.evaluation) if f.name != "training"},
            "versions": dict(self.versions),
            "reproducibility": {"workers": self.settings.workers,
                                "metric_tolerance": self.metric_tolerance},
            "bundle_sha256": self.bundle_sha256,
        }


# --- JSON form ---------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        data = {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
        for name, attribute in vars(type(value)).items():  # e.g. CVSummary.macro_f1_mean
            if isinstance(attribute, property):
                data[name] = _jsonable(getattr(value, name))
        return data
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, type):
        return value.__name__
    return repr(value)


def _profile_dict(match: ProfileMatch | None) -> dict[str, Any] | None:
    if match is None:
        return None
    return {
        "name": match.profile.name, "title": match.profile.title, "variant": match.variant,
        "has_header": match.has_header, "columns": len(match.column_names),
        "label_column": match.label_column, "blocked_columns": list(match.blocked_columns),
        "split_strategy": match.split_strategy, "time_column": match.time_column,
        "group_column": match.group_column, "warnings": list(match.warnings),
    }


def _split_dict(split: SplitResult) -> dict[str, Any]:
    data = {f.name: _jsonable(getattr(split, f.name)) for f in fields(split)
            if f.name not in ("train", "test")}
    data["train_rows"], data["test_rows"] = len(split.train), len(split.test)
    return data


def _training_dict(outcome: TrainingOutcome) -> dict[str, Any]:
    return {
        "train_rows": outcome.train_rows, "classes": list(outcome.classes),
        "class_counts": _jsonable(outcome.class_counts),
        "numeric_features": list(outcome.numeric_features),
        "categorical_features": list(outcome.categorical_features),
        "failures": _jsonable(outcome.failures), "skipped": _jsonable(outcome.skipped),
        "notes": list(outcome.notes), "seed": outcome.seed, "workers": outcome.workers,
    }


def _models_dict(outcome: TrainingOutcome) -> dict[str, Any]:
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
            "params": _jsonable(steps["model"].get_params(deep=False)),
            "fit_seconds": fitted.fit_seconds,
            "notes": list(fitted.notes),
        }
    return models


def _step_dict(step: Any) -> dict[str, Any] | None:
    if step is None or step == "passthrough":
        return None
    return {"class": type(step).__name__, "params": _jsonable(step.get_params(deep=False))}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version(),
                                       "platform": platform.platform()}
    for package in VERSIONED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


# --- train / re-run ------------------------------------------------------------------

def train(
    frame: pd.DataFrame,
    *,
    profile: ProfileMatch | None = None,
    settings: TrainSettings | None = None,
    test_frame: pd.DataFrame | None = None,
    source: DatasetRef | None = None,
    test_source: DatasetRef | None = None,
    sources: Sequence[str] | pd.Series | None = None,
    progress: ProgressSink | None = None,
) -> TrainingRun:
    """Split, guard, train and evaluate; returns the run (nothing is written).

    ``progress`` (v5.1 step 1.2d) gets ``start(total_steps(...))``: one step for the split,
    one for the leakage guard, then one per model per CV fold and per final fit; then
    ``finish()``, also after an error. An exception raised by ``advance()`` (a cancelled
    job) stops the training; nothing is returned or written."""
    settings = settings or TrainSettings()
    label = settings.label or (profile.label_column if profile else None)
    if not label:
        raise ValueError("No label column: set TrainSettings.label or pass a profile that has one")
    settings = dataclasses.replace(settings, workers=settings.workers or max_workers())
    now = datetime.now(timezone.utc)
    progress = progress or NullProgress()
    progress.start(total_steps(settings))
    try:
        return _train(frame, label, settings, now, profile, test_frame, source, test_source,
                      sources, progress)
    finally:
        progress.finish()


def total_steps(settings: TrainSettings) -> int:
    """Progress steps of ``train``: split + leakage guard + models x (CV folds + final fit)."""
    models = settings.models or registry.default_models()
    return 2 + len(models) * (settings.evaluation.cv_folds + 1)


def _train(frame: pd.DataFrame, label: str, settings: TrainSettings, now: datetime,
           profile: ProfileMatch | None, test_frame: pd.DataFrame | None,
           source: DatasetRef | None, test_source: DatasetRef | None,
           sources: Sequence[str] | pd.Series | None, progress: ProgressSink) -> TrainingRun:
    split = split_dataset(frame, label, profile=profile, strategy=settings.strategy,
                          time_column=settings.time_column, group_column=settings.group_column,
                          test_frame=test_frame, sources=sources, config=settings.split)
    progress.advance()
    guard = check_leakage(split.train, label, features=settings.features,
                          profile_blocked=profile.blocked_columns if profile else (),
                          allow=settings.allow, config=settings.leakage)
    progress.advance()
    evaluation = evaluate(split.train, split.test, label, features=guard.features,
                          models=settings.models, balancer=settings.balancer,
                          selector=settings.selector, k_features=settings.k_features,
                          config=settings.evaluation, pipeline_config=settings.pipeline,
                          workers=settings.workers, progress=progress)
    return TrainingRun(
        run_id=f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}",
        created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        settings=settings,
        dataset=source,
        test_dataset=test_source,
        profile=profile,
        split=split,
        leakage=guard,
        evaluation=evaluation,
        versions=package_versions(),
    )


def _check_source(recorded: Mapping[str, Any] | None, given: DatasetRef | None, what: str) -> None:
    if recorded is None:
        return
    if given is None:
        raise ValueError(f"The run used {recorded['name']}: pass its DatasetRef so the SHA-256 "
                         "can be checked")
    if given.sha256 != recorded["sha256"]:
        raise ValueError(f"The {what} SHA-256 differs from the one in the run "
                         f"({given.sha256[:12]}… vs {recorded['sha256'][:12]}…)")


def rerun(
    record: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    profile: ProfileMatch | None = None,
    test_frame: pd.DataFrame | None = None,
    source: DatasetRef | None = None,
    test_source: DatasetRef | None = None,
) -> TrainingRun:
    """Train again with a saved run's settings, on the same data (checked by SHA-256)."""
    _check_source(record.get("dataset"), source, "dataset")
    _check_source(record.get("test_dataset"), test_source, "test file")
    recorded_profile = record.get("profile")
    if recorded_profile and (profile is None or profile.profile.name != recorded_profile["name"]
                             or profile.variant != recorded_profile["variant"]):
        raise ValueError(f"The run used the {recorded_profile['name']} / "
                         f"{recorded_profile['variant']} profile: load the dataset again with "
                         "load_dataset and pass its profile")
    return train(frame, profile=profile, settings=TrainSettings.from_dict(record["settings"]),
                 test_frame=test_frame, source=source, test_source=test_source)


_TEST_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
_CV_METRICS = ("macro_f1", "balanced_accuracy", "accuracy")


def compare_metrics(first: Mapping[str, Any], second: Mapping[str, Any],
                    tolerance: float | None = None) -> list[str]:
    """Differences between two run records' results, larger than ``tolerance`` (default:
    the larger of the two recorded tolerances; 0 means exact, confusion matrices included)."""
    if tolerance is None:
        tolerance = max(first["reproducibility"]["metric_tolerance"],
                        second["reproducibility"]["metric_tolerance"])
    differences = []
    a, b = first["evaluation"], second["evaluation"]
    if a["selected"] != b["selected"]:
        differences.append(f"selected model: {a['selected']} vs {b['selected']}")
    other = {m["name"]: m for m in b["models"]}
    for model in a["models"]:
        name = model["name"]
        if name not in other:
            differences.append(f"{name}: missing in the second run")
            continue
        pairs = [(f"test.{k}", model["test"][k], other[name]["test"][k]) for k in _TEST_METRICS]
        pairs += [(f"cv.{k}[{i}]", x, y) for k in _CV_METRICS
                  for i, (x, y) in enumerate(zip(model["cv"][k], other[name]["cv"][k], strict=True))]
        pairs += [(f"test.recall[{x['label']}]", x["recall"], y["recall"])
                  for x, y in zip(model["test"]["per_class"], other[name]["test"]["per_class"],
                                  strict=True)]
        for key, x, y in pairs:
            if abs(x - y) > tolerance:
                differences.append(f"{name} {key}: {x} vs {y}")
        if tolerance == 0 and model["test"]["confusion"] != other[name]["test"]["confusion"]:
            differences.append(f"{name} test.confusion differs")
    return differences
