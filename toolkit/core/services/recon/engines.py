"""The scan engines (step N1): what checks a host's ports. ``ENGINES`` names each; the scan
picks one by name (``ScanRequest.engine``). The toolkit's own scanner is the default and
needs nothing installed; nmap comes as an engine of its own when it is installed (N2-N4).
"""

import threading
from collections.abc import Callable, Iterable, Sequence

from ...data.recon.scanner import PortResult, Progress, scan_ports
from ..ports import ScanEngine


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


ENGINES: dict[str, Callable[[], ScanEngine]] = {"builtin": BuiltinEngine}
DEFAULT_ENGINE = "builtin"


def engine_named(name: str) -> ScanEngine:
    """The engine called ``name``; ValueError with the known names for any other."""
    if name not in ENGINES:
        raise ValueError(f"Unknown scan engine {name!r}; use one of {', '.join(ENGINES)}.")
    return ENGINES[name]()
