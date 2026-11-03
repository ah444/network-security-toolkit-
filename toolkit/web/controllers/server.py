"""Serving the web trainer on this computer only (127.0.0.1, S6), or, with ``--host`` (v5.2
step 2.4f), on an address of the network too: one server per instance (its lock, HR3b), the
port bound exclusively, the browser opened once it answers, waitress (or Flask's debug
server, on this computer only)."""

from __future__ import annotations

import errno
import ipaddress
import logging
import threading
import time

from ...core.data.logs import setup_logging
from ...core.data.settings import WebConfig, web_config
from ..data import instance_lock, net
from ..services.access import is_this_computer
from ..services.workspace import start_idle_sweep
from ..ui import console as messages
from . import create_app

HOST = "127.0.0.1"  # this computer only (S6), unless --host opens it to the network (2.4f)
ALL_IPV4 = "0.0.0.0"
ADDRESS_NOT_HERE = {errno.EADDRNOTAVAIL, 10049}  # 10049: Windows' WSAEADDRNOTAVAIL
BROWSER_WAIT_SECONDS = 15  # the browser is opened once the port answers, if within this time


def open_browser_when_ready(url: str, host: str, port: int,
                            timeout: float = BROWSER_WAIT_SECONDS) -> threading.Thread:
    """Open ``url`` in the default browser once the port answers; if no browser can be
    opened (no desktop, for example), say where to go instead."""
    return net.open_browser_when_ready(url, host, port, timeout,
                                       lambda: messages.open_in_browser(url))


def url_of(host: str, port: int) -> str:
    """The page's address on ``host`` (an IPv6 address in brackets)."""
    return f"http://[{host}]:{port}/" if ":" in host else f"http://{host}:{port}/"


def network_urls(host: str, port: int) -> list[str]:
    """Where other computers open the toolkit: this computer's addresses for 0.0.0.0, else the
    address itself ([] for "::": the system lists no IPv6 addresses here)."""
    if host == ALL_IPV4:
        return [url_of(address, port) for address in net.local_addresses()]
    if ipaddress.ip_address(host).is_unspecified:
        return []
    return [url_of(host, port)]


def run_server(config: WebConfig | None = None) -> int:
    """Serve the app on this computer only (127.0.0.1), S6; the exit code.

    - Default: waitress, a production-grade pure-Python server (works on every OS).
    - TOOLKIT_DEBUG=1: Flask's debug server, for development only (one process, no
      reloader). Never use it when other people can reach the machine: its debugger can
      run code.
    - One web server per instance (HR3b): another one holding it is a message, exit code 2.
    - TOOLKIT_PORT: port number (default 5000); a port in use is a message, exit code 2.
    - The default browser opens the page once the port answers (TOOLKIT_NO_BROWSER=1: no).
    - --host ADDRESS (v5.2 step 2.4f): also on that address (0.0.0.0: all of this computer's
      IPv4 ones); 127.0.0.1 stays bound, so this computer's browser needs no access token;
      the warning and the token are printed. TOOLKIT_DEBUG=1 is refused then (exit code 2).
    """
    setup_logging("web", console_level=logging.INFO)
    config = config or web_config()
    port, local = config.port, is_this_computer(config.host)
    host = config.host if local else HOST  # where this computer's browser opens it
    url = url_of(host, port)
    if config.debug and not local:
        messages.debug_refused(config.host)
        return 2
    try:
        lock = instance_lock.take(url)
    except instance_lock.InstanceBusy as busy:
        holder = busy.holder
        messages.instance_in_use(busy.folder, holder.pid if holder else None,
                                 holder.url if holder else "")
        return 2
    interrupted: dict[str, list[str]] = {}
    for job in lock.interrupted:  # HR3c: said in the terminal now, on its page once
        line = messages.interrupted_line(job.kind, time.strftime("%Y-%m-%d %H:%M",
                                                                 time.localtime(job.asked)),
                                         job.source, job.ran)
        messages.interrupted(line)
        interrupted.setdefault(job.kind, []).append(line)
    try:
        return _serve(config, host, url, interrupted)
    finally:
        lock.release()


def _serve(config: WebConfig, host: str, url: str, interrupted: dict[str, list[str]]) -> int:
    """``run_server`` once the instance is held: the port bound, the app served until it
    stops; the exit code."""
    port, local = config.port, is_this_computer(config.host)
    hosts = [config.host] if local or config.host == ALL_IPV4 else [config.host, HOST]
    try:
        sockets = net.bind_ports(hosts, port)
    except OSError as error:
        if error.errno in ADDRESS_NOT_HERE or getattr(error, "winerror", None) == 10049:
            messages.not_this_computer(config.host)
        else:
            messages.port_in_use(port, url, port + 1 if port < 65535 else port - 1)
        return 2
    try:
        app = create_app(config)
        app.extensions["toolkit"].interrupted = interrupted
        start_idle_sweep(app.extensions["toolkit"].workspaces)  # frees memory without requests
        messages.running(url)
        access = app.extensions["toolkit"].access
        if access is not None:
            messages.open_to_network(network_urls(config.host, port), access.token)
        if config.open_browser:
            open_browser_when_ready(url, host, port)
        if config.debug:  # this computer only (refused above otherwise)
            messages.debug_mode()
            for sock in sockets:
                sock.close()  # Flask's debug server binds the port itself (checked free above)
            app.run(host=host, port=port, debug=True, use_reloader=False)
        else:
            from waitress import serve
            serve(app, sockets=sockets)
    except KeyboardInterrupt:
        pass  # waitress stops by itself on Ctrl+C; Flask's debug server raises it
    finally:
        for sock in sockets:
            sock.close()
    messages.stopped()
    return 0
