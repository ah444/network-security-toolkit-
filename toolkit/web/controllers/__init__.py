"""Web trainer: the Flask app (presentation layer, plan §2; moved here in v5.1 step 1.2a).

Since v5.0 (step 0.4g) the routes are a thin layer over ``toolkit.core.services.ids``: the loader
reads uploads, the split, leakage guard, training pipeline and evaluation run in
``train(...)``, and models are saved as signed bundles. Nothing here preprocesses data
itself, so the web trainer, a saved bundle and the CLI all use the same pipeline.

``create_app(config)`` builds an app with its own state (``app.extensions["toolkit"]``);
``server.run_server()`` serves one on this computer only. The page is ``toolkit.web.ui``. Library warnings are deliberately
NOT silenced (B19): they flag problems such as a model saved with a different
scikit-learn version.
"""

from __future__ import annotations

import ipaddress
import mimetypes
import secrets

from flask import Flask

from ...core.services.ids.exports import BundleStore
from ...core.data.settings import WebConfig, instance_dir, web_config
from . import access, bundles, datasets, jobs, pages, resume, runs, scan, scans, training, workspace
from .api import register_error_handlers
from ..services.datasets import prepare_uploads
from ..services.access import Access, is_this_computer
from ..services.jobs import IMPORTANCE, SCAN, TRAINING, JobRunner, instance_journal
from .security import register_security
from .state import WebState
from ..services.workspace import WorkspaceStore
from .. import ui as page

BLUEPRINTS = (pages.bp, access.bp, datasets.bp, training.bp, jobs.bp, scan.bp, bundles.bp,
              workspace.bp, resume.bp, runs.bp, scans.bp)


def network_banner(config: WebConfig) -> str:
    """The pages' banner when ``--host`` opened the toolkit to the network, else ''."""
    if is_this_computer(config.host):
        return ""
    where = (f"{config.host}, port {config.port}" if not ipaddress.ip_address(config.host)
             .is_unspecified else f"all of this computer's addresses, port {config.port}")
    return (f"This toolkit is open to your network ({where}): anyone with its access token can "
            "use it, port scans included. Stop it with Ctrl+C in its terminal when you are done.")


def create_app(config: WebConfig | None = None) -> Flask:
    """The web trainer with its own workspaces; settings from ``config`` (default: the
    ``TOOLKIT_*`` environment variables, see ``toolkit.core.data.settings.web_config``)."""
    config = config or web_config()
    # The page's scripts are ES modules, which browsers run only when served as JavaScript;
    # some Windows installs map ".js" to text/plain in the registry, which mimetypes reads.
    mimetypes.add_type("text/javascript", ".js")
    app = Flask(__name__, template_folder=page.TEMPLATES, static_folder=page.STATIC)
    # Upload size limit (S9). Large enough for the full CIC-IDS2017 file (~1 GB).
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_mb * 1024 * 1024
    app.secret_key = config.secret_key or secrets.token_hex(32)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        # Only behind HTTPS: browsers drop Secure cookies on plain http://127.0.0.1 (D23).
        SESSION_COOKIE_SECURE=config.https,
    )
    upload_dir = instance_dir() / "uploads"
    prepare_uploads(upload_dir)  # without the files a crash left (S11)
    # other computers type a token, made new at every start (v5.2 step 2.4f); a job they asked
    # for under an older unlock never starts (HS1)
    access = None if is_this_computer(config.host) else Access()
    refusal = access.refusal if access is not None else None
    journal = instance_journal()  # each job's start and end, read at the next start (HR3c)
    app.extensions["toolkit"] = WebState(
        config=config,
        workspaces=WorkspaceStore(config.workspace_idle_minutes * 60,
                                  max_workspaces=config.max_workspaces,
                                  max_data_bytes=config.max_data_mb * 2**20),
        model_store=BundleStore(),
        upload_dir=upload_dir,
        training_jobs=JobRunner(TRAINING, refusal, journal, also=(IMPORTANCE,)),  # RD2
        scan_jobs=JobRunner(SCAN, refusal, journal),
        access=access,
    )
    register_error_handlers(app)
    register_security(app, config.port)
    # every page says so while the toolkit is open to the network (v5.2 step 2.4f)
    app.jinja_env.globals["network_banner"] = network_banner(config)
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    return app
