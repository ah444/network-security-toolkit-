"""The web trainer's listening sockets, this computer's addresses, and opening the page in the
default browser."""

import ipaddress
import socket
import threading
import time
import webbrowser
from collections.abc import Callable


def bind_port(host: str, port: int) -> socket.socket:
    """A listening socket on ``host:port`` that no other program can share (step 1.4d).

    waitress sets SO_REUSEADDR, and on Windows that lets a second server bind a port that
    is already in use, without an error. So the port is bound here and the socket handed
    to waitress: exclusively on Windows (SO_EXCLUSIVEADDRUSE); elsewhere SO_REUSEADDR only
    allows a quick restart, never two listeners. OSError when the port is in use, or when
    ``host`` (an IPv4 or IPv6 address) isn't one of this computer's.
    """
    family = socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
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


def bind_ports(hosts: list[str], port: int) -> list[socket.socket]:
    """``bind_port`` on each of ``hosts``; OSError (the others closed) when one fails."""
    sockets: list[socket.socket] = []
    try:
        for host in hosts:
            sockets.append(bind_port(host, port))
    except OSError:
        for sock in sockets:
            sock.close()
        raise
    return sockets


def local_addresses() -> list[str]:
    """This computer's IPv4 addresses other computers may reach (loopback left out), as the
    system's resolver gives them for its own name; [] if it gives none."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return []
    found = dict.fromkeys(str(info[4][0]) for info in infos)
    return [address for address in found if not ipaddress.ip_address(address).is_loopback]


def open_browser_when_ready(url: str, host: str, port: int, timeout: float,
                            no_browser: Callable[[], None]) -> threading.Thread:
    """Open ``url`` in the default browser from a thread, once the port answers;
    ``no_browser`` is called when none can be opened (no desktop, for example)."""
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
            no_browser()

    thread = threading.Thread(target=wait_and_open, name="open-browser", daemon=True)
    thread.start()
    return thread
