"""The permutation importance of one of the workspace's models, as a job (v5.3 step HR1, RD2).

It waits in the training's queue, so a training and an importance never run at once. It is
bound when it is asked for to its run and model: references to that fitted model and to that
run's test rows, its label and its evaluation seed, never "the current run" when it starts,
so a training after it changes nothing it computes (those references keep the old run in
memory until it ends). One job per run and model: asked again, it is the same job, unless
that one failed or was cancelled. Only the workspace's run can be calculated (RD3): a saved
run has no test split in memory. The result is the job's for now; HR2 makes it a file of its
own, F3d the route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ...core.data.settings import ExplainConfig
from ...core.services.ids.importance import Importance, permutation_importances
from ...core.services.ids.pipeline import FittedModel
from .errors import NOT_TRAINED, ServiceError
from .jobs import FAILED, HERE, IMPORTANCE, STOPPED, Job, JobRunner, Origin
from .workspace import Workspace


@dataclass(frozen=True)
class Bound:
    """What an importance job computes, taken when it is asked for (references, not copies)."""
    run_id: str
    model: str
    fitted: FittedModel
    test: pd.DataFrame
    label: str
    seed: int


def bind(ws: Workspace, model: str | None) -> Bound:
    """The workspace's run and its model ``model`` (the selected one by default); refused when
    there is no run or no such model in it."""
    run = ws.run
    if run is None:
        raise ServiceError("Train a model first, then calculate its importance.", NOT_TRAINED)
    name = model or run.evaluation.selected or ""
    fitted = run.evaluation.training.models.get(name)
    label = run.settings.label or (run.profile.label_column if run.profile else None)
    if fitted is None or not label:
        raise ServiceError(f"This run has no model {name!r}.")
    return Bound(run.run_id, name, fitted, run.split.test, label, run.evaluation.seed)


def start_importance(ws: Workspace, model: str | None, jobs: JobRunner, max_seconds: float,
                     origin: Origin = HERE, config: ExplainConfig | None = None) -> Job:
    """The importance job of ``model`` of the workspace's run, queued now, or the one already
    asked for (waiting, running or done). It holds a lease on the workspace until it ends
    (D22), not its training lock: a training can be asked for meanwhile, and waits behind it."""
    with ws.lock:
        bound = bind(ws, model)
        key = (bound.run_id, bound.model)
        for job in reversed(list(ws.jobs.values())):
            if job.kind == IMPORTANCE and job.key == key and job.state not in (FAILED, STOPPED):
                return job
        ws.acquire_lease()
        job = Job(kind=IMPORTANCE, work=lambda job: answer(bound, compute(bound, job, config)),
                  on_end=ws.release_lease, max_seconds=max_seconds, origin=origin, key=key,
                  steps=importance_steps)
        ws.add_job(job)
    jobs.submit(job)
    return job


def compute(bound: Bound, job: Job, config: ExplainConfig | None) -> Importance:
    """The job's work, in the worker thread: one step per feature, a cancel between two."""
    return permutation_importances(bound.fitted, bound.test, bound.label, seed=bound.seed,
                                   config=config, progress=job)


def importance_steps(done: int, total: int) -> str:
    """A running importance's phase text: the feature it shuffles now."""
    return f"Permutation importance, feature {min(done + 1, total)} of {total}"


def answer(bound: Bound, found: Importance) -> dict[str, Any]:
    """The job's result: the run and model it belongs to, and the importance."""
    return {
        "run_id": bound.run_id,
        "model": bound.model,
        "rows": found.rows,
        "test_rows": found.test_rows,
        "repeats": found.repeats,
        "seed": found.seed,
        "baseline_macro_f1": found.baseline_macro_f1,
        "features": [{"feature": f.feature, "mean": f.mean, "std": f.std}
                     for f in found.features],
        "concentrated": found.concentrated,
    }
