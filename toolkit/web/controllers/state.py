"""What one web app keeps between requests: ``app.extensions["toolkit"]`` (step 1.2a).

Nothing lives in module globals, so two apps (two tests, say) never share a workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from flask import current_app

from ...core.data.settings import WebConfig
from ...core.services.ports import ModelStore

if TYPE_CHECKING:
    from ..services.access import Access
    from ..services.jobs import JobKind, JobRunner
    from ..services.workspace import WorkspaceStore


@dataclass
class WebState:
    """What one app keeps between requests: its settings, the workspaces, the model store, the
    upload folder and a job runner for each kind of job."""
    config: WebConfig
    workspaces: WorkspaceStore
    model_store: ModelStore  # exports and loads: signed bundles in instance/bundles
    upload_dir: Path  # uploads are written here only while the loader reads them (S11)
    training_jobs: JobRunner  # one training at a time for the whole server, others queued
    scan_jobs: JobRunner  # one scan at a time for the whole server, next to a training
    access: Access | None = None  # the access token when opened to the network (--host)
    # the last server's unfinished jobs, by kind ("training", "scan"): a line each, shown once
    # in each session on that kind's page (HR3c)
    interrupted: dict[str, list[str]] = field(default_factory=dict)

    def runner(self, kind: JobKind) -> JobRunner:
        """The runner whose queue holds the jobs of ``kind``."""
        for runner in (self.training_jobs, self.scan_jobs):
            if runner.kind == kind:
                return runner
        raise LookupError(f"no runner for {kind.name} jobs")


def state() -> WebState:
    """The state of the app that handles the current request."""
    web_state: WebState = current_app.extensions["toolkit"]
    return web_state
