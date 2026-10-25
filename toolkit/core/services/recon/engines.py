"""The scan engines (step N1): what checks a host's ports. ``ENGINES`` names each; the scan
picks one by name (``ScanRequest.engine``). The toolkit's own scanner is the default and
needs nothing installed; nmap is an engine of its own when it is installed (N2-N4).
"""

import threading
from collections.abc import Callable, Iterable, Sequence

from ...data.recon.nmap import NmapMissing, NmapOptions, find_nmap, has_privileges, nmap_command
from ...data.recon.nmap_run import run_nmap
from ...data.recon.nmap_xml import parse_nmap
from ...data.recon.scanner import PortResult, Progress, scan_ports
from ...data.settings import scan_limits
from ..ports import NullProgress, ScanEngine

NMAP_GRACE = 120  # seconds nmap may run past its --host-timeout before it is stopped
PORT_TASKS = {"tcp": ("SYN Stealth Scan", "Connect Scan"), "udp": ("UDP Scan",)}
LATER_TASKS = (("Service scan",), ("NSE",))  # -sV, then its version scripts and --scripts


class BuiltinEngine:
    """The toolkit's own scanner (``toolkit.core.data.recon.scanner``): a TCP connection or a
    UDP probe per port and the banner it sends; works on every OS with nothing installed."""

    name = "builtin"

    def available(self) -> str | None:
        """Always None: it can always run."""
        return None

    def scan(self, host: str, ports: Iterable[int], protocols: Sequence[str],
             progress: Progress | None = None,
             stop: threading.Event | None = None) -> list[PortResult]:
        """``scan_ports`` of ``host``."""
        return scan_ports(host, ports, protocols, progress, stop)


class NmapProgress:
    """nmap's tasks as one bar: each phase it runs (the port scans, the service scan, the
    scripts) is 100 units, a task's percent fills its phase, and the bar never goes back.
    nmap reports a phase only if it lasts a second or more, so a phase can be skipped."""

    def __init__(self, sink: Progress, protocols: Sequence[str]) -> None:
        self.sink = sink
        self.phases = [PORT_TASKS[p] for p in ("tcp", "udp") if p in protocols] + list(
            LATER_TASKS)
        self.total = 100 * len(self.phases)
        self.done = 0
        self.lock = threading.Lock()

    def _reach(self, value: int) -> None:
        """Moves the bar to ``value`` units if that is further."""
        with self.lock:
            value = min(value, self.total)
            if value > self.done:
                self.sink.advance(value - self.done)
                self.done = value

    def report(self, task: str, percent: float) -> None:
        """nmap's ``task`` is ``percent`` done; a task of no phase is left out."""
        for index, names in enumerate(self.phases):
            if task in names:
                self._reach(index * 100 + min(int(percent), 100))

    def complete(self) -> None:
        """nmap finished: the bar is full."""
        self._reach(self.total)


class NmapEngine:
    """nmap (``toolkit.core.data.recon.nmap``), when it is installed: its port scan, service
    detection (with nmap's own version scripts), and if chosen OS detection and the allowed
    scripts (``NmapOptions``). The ports come back as the built-in scanner's states, with
    ``product``, ``version``, ``cpe``, ``reason`` and ``scripts``; the host's OS matches and
    scripts are kept on the engine (``os_matches``, ``host_scripts``) after a scan."""

    name = "nmap"

    def __init__(self, options: NmapOptions | None = None) -> None:
        self.options = options or NmapOptions()
        self.os_matches: tuple[tuple[str, int], ...] = ()
        self.host_scripts: tuple[tuple[str, str], ...] = ()

    def available(self) -> str | None:
        """None when nmap is found and new enough, else why not and how to install it."""
        try:
            find_nmap()
        except NmapMissing as error:
            return str(error)
        return None

    def scan(self, host: str, ports: Iterable[int], protocols: Sequence[str],
             progress: Progress | None = None,
             stop: threading.Event | None = None) -> list[PortResult]:
        """nmap's scan of ``host``, with raw packets when it can send them (``has_privileges``).

        NmapMissing, ValueError (UDP or OS detection without privileges, a bad address),
        NmapFailed, or ScanStopped once ``stop`` is set; the progress is finished in every
        case."""
        port_list = list(ports)
        minutes = scan_limits().max_minutes
        command = nmap_command(find_nmap().path, host, port_list, protocols, self.options,
                               privileged=has_privileges(), max_minutes=minutes)
        sink = progress or NullProgress()
        bar = NmapProgress(sink, protocols)
        sink.start(bar.total)
        try:
            xml = run_nmap(command, bar.report, stop, time_limit=minutes * 60 + NMAP_GRACE)
            bar.complete()
        finally:
            sink.finish()
        found = parse_nmap(xml, host, port_list, protocols)
        self.os_matches, self.host_scripts = found.os_matches, found.scripts
        return found.ports


ENGINES: dict[str, Callable[[], ScanEngine]] = {"builtin": BuiltinEngine, "nmap": NmapEngine}
DEFAULT_ENGINE = "builtin"


def engine_named(name: str) -> ScanEngine:
    """The engine called ``name``; ValueError with the known names for any other."""
    if name not in ENGINES:
        raise ValueError(f"Unknown scan engine {name!r}; use one of {', '.join(ENGINES)}.")
    return ENGINES[name]()
