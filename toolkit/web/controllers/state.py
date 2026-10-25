"""What one web app keeps between requests: ``app.extensions["toolkit"]`` (step 1.2a).

Nothing lives in module globals, so two apps (two tests, say) never share a workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from flask import current_app

from ...core.data.settings import WebConfig
from ...core.services.ports import ModelStore

if TYPE_CHECKING:
    from ..services.jobs import JobRunner
    from ..services.workspace import WorkspaceStore


@dataclass
class WebState:
    """What one app keeps between requests: its settings, the workspaces, the model store, the
    upload folder and the job runner."""
    config: WebConfig
    workspaces: WorkspaceStore
    model_store: ModelStore  # exports and loads: signed bundles in instance/bundles
    upload_dir: Path  # uploads are written here only while the loader reads them (S11)
    jobs: JobRunner  # one training at a time for the whole server, others queued


def state() -> WebState:
    """The state of the app that handles the current request."""
    web_state: WebState = current_app.extensions["toolkit"]
    return web_state
