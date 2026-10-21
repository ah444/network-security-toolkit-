"""Web trainer: the Flask app (presentation layer, plan §2; moved here in v5.1 step 1.2a).

Since v5.0 (step 0.4g) the routes are a thin layer over ``toolkit.ids_core``: the loader
reads uploads, the split, leakage guard, training pipeline and evaluation run in
``train(...)``, and models are saved as signed bundles. Nothing here preprocesses data
itself, so the web trainer, a saved bundle and the CLI all use the same pipeline.

``create_app(config)`` builds an app with its own state (``app.extensions["toolkit"]``);
``run_server()`` serves one on this computer only. Library warnings are deliberately
NOT silenced (B19): they flag problems such as a model saved with a different
scikit-learn version.
"""

from __future__ import annotations

import logging
import secrets

from flask import Flask

from ..bundle_files import BundleStore
from ..config import WebConfig, instance_dir, web_config
from ..logging_setup import setup_logging
from . import bundles, datasets, pages, training, workspace
from .api import register_error_handlers
from .security import register_security
from .state import WebState
from .workspace import WorkspaceStore

BLUEPRINTS = (pages.bp, datasets.bp, training.bp, bundles.bp, workspace.bp)


def create_app(config: WebConfig | None = None) -> Flask:
    """The web trainer with its own workspaces; settings from ``config`` (default: the
    ``TOOLKIT_*`` environment variables, see ``toolkit.config.web_config``)."""
    config = config or web_config()
    app = Flask(__name__)
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
    upload_dir.mkdir(parents=True, exist_ok=True)
    app.extensions["toolkit"] = WebState(
        config=config,
        workspaces=WorkspaceStore(config.workspace_idle_minutes * 60),
        model_store=BundleStore(),
        upload_dir=upload_dir,
    )
    register_error_handlers(app)
    register_security(app)
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    return app


def run_server(config: WebConfig | None = None) -> None:
    """Serve the app on this computer only (127.0.0.1), S6.

    - Default: waitress, a production-grade pure-Python server (works on every OS).
    - TOOLKIT_DEBUG=1: Flask's debug server, for development only. Never use it
      when other people can reach the machine: its debugger can run code.
    - TOOLKIT_PORT: port number (default 5000).
    """
    setup_logging("web", console_level=logging.INFO)
    config = config or web_config()
    app = create_app(config)
    host = "127.0.0.1"
    port = config.port
    print(f"IDS web app running at http://{host}:{port}/  (press Ctrl+C to stop)")
    if config.debug:
        print("WARNING: debug mode is ON (TOOLKIT_DEBUG=1). Development use only.")
        app.run(host=host, port=port, debug=True)
    else:
        from waitress import serve
        serve(app, host=host, port=port)
