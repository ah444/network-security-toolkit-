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

import argparse
import dataclasses
import logging
import mimetypes
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence

from flask import Flask

from ..bundle_files import BundleStore
from ..config import WebConfig, instance_dir, port_number, web_config
from ..logging_setup import setup_logging
from . import bundles, datasets, pages, resume, training, workspace
from .api import register_error_handlers
from .datasets import remove_stale_uploads
from .jobs import JobRunner
from .security import register_security
from .state import WebState
from .workspace import WorkspaceStore, start_idle_sweep

BLUEPRINTS = (pages.bp, datasets.bp, training.bp, bundles.bp, workspace.bp, resume.bp)


def create_app(config: WebConfig | None = None) -> Flask:
    """The web trainer with its own workspaces; settings from ``config`` (default: the
    ``TOOLKIT_*`` environment variables, see ``toolkit.config.web_config``)."""
    config = config or web_config()
    # The page's scripts are ES modules, which browsers run only when served as JavaScript;
    # some Windows installs map ".js" to text/plain in the registry, which mimetypes reads.
    mimetypes.add_type("text/javascript", ".js")
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
    remove_stale_uploads(upload_dir)  # left by a crash (S11)
    app.extensions["toolkit"] = WebState(
        config=config,
        workspaces=WorkspaceStore(config.workspace_idle_minutes * 60,
                                  max_workspaces=config.max_workspaces,
                                  max_data_bytes=config.max_data_mb * 2**20),
        model_store=BundleStore(),
        upload_dir=upload_dir,
        jobs=JobRunner(),
    )
    register_error_handlers(app)
    register_security(app)
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    return app


HOST = "127.0.0.1"  # this computer only (S6)
BROWSER_WAIT_SECONDS = 15  # the browser is opened once the port answers, if within this time
PORT_IN_USE = ("Port {port} is already in use on this computer: the web trainer may already be "
               "running (open {url}), or another program uses it.\nStart on another port: "
               "python -m toolkit web --port {other} (or TOOLKIT_PORT={other}).")


def bind_port(host: str, port: int) -> socket.socket:
    """A listening socket on ``host:port`` that no other program can share (step 1.4d).

    waitress sets SO_REUSEADDR, and on Windows that lets a second server bind a port that
    is already in use, without an error. So the port is bound here and the socket handed
    to waitress: exclusively on Windows (SO_EXCLUSIVEADDRUSE); elsewhere SO_REUSEADDR only
    allows a quick restart, never two listeners. OSError when the port is in use.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)  # Windows only
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1024)  # waitress's default backlog
    except OSError:
        sock.close()
        raise
    return sock


def open_browser_when_ready(url: str, host: str, port: int,
                            timeout: float = BROWSER_WAIT_SECONDS) -> threading.Thread:
    """Open ``url`` in the default browser from a thread, once the port answers; if no
    browser can be opened (no desktop, for example), say where to go instead."""
    def wait_and_open() -> None:
        end = time.monotonic() + timeout
        while True:
            try:
                with socket.create_connection((host, port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > end:
                    return
                time.sleep(0.2)
        try:
            opened = webbrowser.open(url)
        except webbrowser.Error:
            opened = False
        if not opened:
            print(f"Open {url} in your browser.")

    thread = threading.Thread(target=wait_and_open, name="open-browser", daemon=True)
    thread.start()
    return thread


def run_server(config: WebConfig | None = None) -> int:
    """Serve the app on this computer only (127.0.0.1), S6; the exit code.

    - Default: waitress, a production-grade pure-Python server (works on every OS).
    - TOOLKIT_DEBUG=1: Flask's debug server, for development only (one process, no
      reloader). Never use it when other people can reach the machine: its debugger can
      run code.
    - TOOLKIT_PORT: port number (default 5000); a port in use is a message, exit code 2.
    - The default browser opens the page once the port answers (TOOLKIT_NO_BROWSER=1: no).
    """
    setup_logging("web", console_level=logging.INFO)
    config = config or web_config()
    host, port = HOST, config.port
    url = f"http://{host}:{port}/"
    try:
        sock = bind_port(host, port)
    except OSError:
        print(PORT_IN_USE.format(port=port, url=url, other=port + 1 if port < 65535 else port - 1))
        return 2
    try:
        app = create_app(config)
        start_idle_sweep(app.extensions["toolkit"].workspaces)  # frees memory without requests
        print(f"IDS web app running at {url}  (press Ctrl+C to stop)")
        if config.open_browser:
            open_browser_when_ready(url, host, port)
        if config.debug:
            print("WARNING: debug mode is ON (TOOLKIT_DEBUG=1). Development use only.")
            sock.close()  # Flask's debug server binds the port itself (checked free above)
            app.run(host=host, port=port, debug=True, use_reloader=False)
        else:
            from waitress import serve
            serve(app, sockets=[sock])
    except KeyboardInterrupt:
        pass  # waitress stops by itself on Ctrl+C; Flask's debug server raises it
    finally:
        sock.close()
    print("Web trainer stopped.")
    return 0


def _port_argument(text: str) -> int:
    try:
        return port_number(text, "--port")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def run(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    """The command (``python -m toolkit web``, ``python -m toolkit.web``): its flags, then
    ``run_server``; exits with code 2 for a bad setting or a port in use."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="The web trainer, on this computer only (127.0.0.1). It opens in your "
                    "default browser. Other settings: TOOLKIT_DEBUG and the other TOOLKIT_* "
                    "environment variables in toolkit/config.py.")
    parser.add_argument("--port", type=_port_argument,
                        help="port number (default: TOOLKIT_PORT, else 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open the browser (also TOOLKIT_NO_BROWSER=1)")
    args = parser.parse_args(argv)
    try:
        config = web_config()
    except ValueError as error:  # a TOOLKIT_* setting that isn't a number, a bad port
        print(error)
        sys.exit(2)
    if args.port:
        config = dataclasses.replace(config, port=args.port)
    if args.no_browser:
        config = dataclasses.replace(config, open_browser=False)
    code = run_server(config)
    if code:
        sys.exit(code)
