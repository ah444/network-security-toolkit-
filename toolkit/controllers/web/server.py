"""Serving the web trainer on this computer only (127.0.0.1, S6): the port bound
exclusively, the browser opened once it answers, waitress (or Flask's debug server)."""

from __future__ import annotations

import logging
import threading

from ...data.logs import setup_logging
from ...data.settings import WebConfig, web_config
from ...data.web import net
from ...services.web.workspace import start_idle_sweep
from ...ui.cli import web as messages
from . import create_app

HOST = "127.0.0.1"  # this computer only (S6)
BROWSER_WAIT_SECONDS = 15  # the browser is opened once the port answers, if within this time


def open_browser_when_ready(url: str, host: str, port: int,
                            timeout: float = BROWSER_WAIT_SECONDS) -> threading.Thread:
    """Open ``url`` in the default browser once the port answers; if no browser can be
    opened (no desktop, for example), say where to go instead."""
    return net.open_browser_when_ready(url, host, port, timeout,
                                       lambda: messages.open_in_browser(url))


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
        sock = net.bind_port(host, port)
    except OSError:
        messages.port_in_use(port, url, port + 1 if port < 65535 else port - 1)
        return 2
    try:
        app = create_app(config)
        start_idle_sweep(app.extensions["toolkit"].workspaces)  # frees memory without requests
        messages.running(url)
        if config.open_browser:
            open_browser_when_ready(url, host, port)
        if config.debug:
            messages.debug_mode()
            sock.close()  # Flask's debug server binds the port itself (checked free above)
            app.run(host=host, port=port, debug=True, use_reloader=False)
        else:
            from waitress import serve
            serve(app, sockets=[sock])
    except KeyboardInterrupt:
        pass  # waitress stops by itself on Ctrl+C; Flask's debug server raises it
    finally:
        sock.close()
    messages.stopped()
    return 0
