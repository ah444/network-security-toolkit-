"""The web trainer's training work: the settings checked, the job queued, the run kept in
the session's workspace (a cancel or a failure keeps nothing)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from ...data.settings import SplitConfig
from ..ids import registry
from ..ids.runs import TrainingRun, TrainSettings
from ..ids.runs import train as train_run
from .datasets import label_values
from .errors import BUSY, TRAINING_FAILED, UNKNOWN_COLUMN, ServiceError
from .jobs import Job, JobRunner
from .workspace import UploadedData, Workspace

log = logging.getLogger(__name__)


def start_job(ws: Workspace, data: UploadedData, settings: TrainSettings, jobs: JobRunner,
              max_seconds: float, answer: Callable[[TrainingRun], dict[str, Any]]) -> Job:
    """Queue a training of the session dataset; the job's result is ``answer(run)``.

    One training at a time per workspace (``training`` lock, else BUSY); the lock and a
    lease (D22) are held from now until the job ends, so the dataset can't change and the
    workspace isn't dropped while the job waits or runs.
    """
    if not ws.training.acquire(blocking=False):
        raise ServiceError("A training run is already in progress in this workspace. "
                           "Wait for it to finish or cancel it.", BUSY)
    ws.acquire_lease()

    def release() -> None:
        ws.release_lease()
        ws.training.release()

    job = Job(work=lambda job: answer(run_training(ws, data, settings, job)), on_end=release,
              max_seconds=max_seconds, models=len(settings.models or ()),
              folds=settings.evaluation.cv_folds)
    ws.split = None  # free the preview split: training makes its own
    ws.add_job(job)
    jobs.submit(job)
    return job


def training_settings(data: UploadedData, target: str, features: list[str] | None,
                      models: list[str] | None, balancing: str,
                      test_size: float) -> TrainSettings:
    """TrainSettings from the request's choices, checked here so errors read well (S8)."""
    models = models or list(registry.default_models())
    balancer = balancing or "none"
    if not target or target not in data.frame.columns:
        raise ServiceError("Choose the target column.", UNKNOWN_COLUMN)
    if not features:
        raise ServiceError("Choose at least one input feature.")
    missing = [c for c in features if c not in data.frame.columns]
    if missing:
        raise ServiceError(f"Unknown input feature(s): {', '.join(missing)}", UNKNOWN_COLUMN)
    if target in features:
        raise ServiceError("The target can't also be an input feature.")
    unknown = [m for m in models if m not in registry.available_models()]
    if unknown:
        raise ServiceError(f"Unknown or unavailable model(s): {', '.join(unknown)}")
    if balancer != "recommended" and balancer not in registry.BALANCERS:
        raise ServiceError(f"Unknown balancing method: {balancer}")
    if label_values(data.frame, target).nunique() < 2:
        raise ServiceError("The target needs at least two classes.")
    return TrainSettings(label=target, features=tuple(features), models=tuple(models),
                         balancer=balancer, split=SplitConfig(test_size=test_size))


def run_training(ws: Workspace, data: UploadedData, settings: TrainSettings,
                 job: Job) -> TrainingRun:
    """The job's work, in the worker thread. Without the workspace lock: dataset changes
    are refused (409) while the training lock is held. Only a finished training is
    stored; a cancel (JobCancelled from ``job.advance``) or a failure stores nothing."""
    try:
        run = train_run(data.frame, profile=data.profile, settings=settings,
                        test_frame=data.test_frame, source=data.source,
                        test_source=data.test_source, sources=data.sources, progress=job)
    except ValueError:
        log.exception("Training failed")
        raise ServiceError("Training could not run with these settings (for example: every chosen "
                       "feature was blocked by the leakage guard, or a class is too small for "
                       "the split). Details are in the server log.", TRAINING_FAILED) from None
    evaluation = run.evaluation
    if not evaluation.training.models or evaluation.selected is None:
        raise ServiceError("No model could be trained. Details are in the server log.",
                       TRAINING_FAILED,
                       details={"failures": [asdict(f) for f in evaluation.failures]})
    with ws.lock:
        ws.run, ws.export = run, None
    return run
