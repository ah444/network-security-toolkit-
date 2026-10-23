"""Port scanning: TCP and UDP checks of one host (R1, R2, B8, B9).

One connection per TCP port, always closed; its banner is read over that connection
(see ``probes``). A UDP port that doesn't answer is ``open|filtered``. Progress goes to
a ``ProgressSink`` (none by default).
"""

import logging
import os
import socket
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..ports import NullProgress, ProgressSink
from .probes import banner_text, read_banner, udp_payload

log = logging.getLogger(__name__)

SCAN_TIMEOUT = 1  # seconds per port


@dataclass
class PortResult:
    port: int
    protocol: str  # "tcp" or "udp"
    state: str  # "open", "closed" (TCP: closed or filtered) or (UDP) "open|filtered"
    service: str = ""
    banner: str = ""


def service_name(port: int, protocol: str = "tcp") -> str:
    try:
        return socket.getservbyport(port, protocol)
    except OSError:
        return ""


def scan_tcp_port(host: str, port: int, timeout: float = SCAN_TIMEOUT) -> PortResult:
    """One connection per port, always closed, also when reading the banner fails (R1)."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError:  # refused or no answer (Windows retries refused connections, so a
        return PortResult(port, "tcp", "closed")  # timeout can't tell "filtered" apart)
    with sock:
        try:
            banner = read_banner(sock, host, port)
        except OSError as error:  # includes TLS errors: the port is still open
            log.debug("Banner of %s:%s failed: %s", host, port, error)
            banner = ""
    return PortResult(port, "tcp", "open", service_name(port), banner)


def scan_udp_port(host: str, port: int, timeout: float = SCAN_TIMEOUT) -> PortResult:
    """A reply means open; no reply is "open|filtered", not "closed" (B8), because many UDP
    services stay silent. "closed" only when the host reports the port unreachable."""
    family, _, _, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(address)
            sock.send(udp_payload(port))
            data = sock.recv(1024)
        except TimeoutError:
            return PortResult(port, "udp", "open|filtered", service_name(port, "udp"))
        except (ConnectionRefusedError, ConnectionResetError):
            return PortResult(port, "udp", "closed")
        except OSError as error:
            log.debug("UDP %s:%s failed: %s", host, port, error)
            return PortResult(port, "udp", "closed")
    return PortResult(port, "udp", "open", service_name(port, "udp"), banner_text(data))


Scan = Callable[[str, int], PortResult]


def scan_ports(host: str, ports: Iterable[int], protocols: Sequence[str],
               progress: ProgressSink | None = None) -> list[PortResult]:
    """Every port of ``host`` for each protocol, sorted by port; each finished check is
    reported to ``progress`` (a ProgressSink, default none)."""
    sink = progress or NullProgress()
    scanners: dict[str, Scan] = {"tcp": scan_tcp_port, "udp": scan_udp_port}
    tasks = [(scanners[protocol], port) for port in ports for protocol in protocols]
    if not tasks:
        return []

    def run(scan: Scan, port: int) -> PortResult:
        try:
            return scan(host, port)
        finally:
            sink.advance()

    results: list[PortResult] = []
    workers = min(100, len(tasks), (os.cpu_count() or 1) * 4)
    sink.start(len(tasks))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run, scan, port) for scan, port in tasks]
            try:
                for future in as_completed(futures):
                    results.append(future.result())
            except BaseException:  # Ctrl+C: only the checks already running finish
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    finally:
        sink.finish()
    return sorted(results, key=lambda r: (r.port, r.protocol))
