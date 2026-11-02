"""The record of every scan (HS3, RD1): one log line when a scan starts and one when it ends,
written by ``run_scan`` for the CLI's scans and the web's alike, in the tool's own log
(``instance/logs/scanner.log`` or ``web.log``, which rotates at 1 MB with three older files).

The values are JSON in ASCII, so a typed target with a line break or quotes stays inside its
own line and can't forge another. The source is where the request came from: this computer,
or another computer's address; an address is not a person (RD1). No key or token is written:
the lines hold what was asked, not how. The lines are INFO: a ``TOOLKIT_LOG_LEVEL`` above
INFO leaves them out.
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Mapping, Sequence
from typing import Any

from ...data.recon.nmap import port_ranges

log = logging.getLogger(__name__)

THIS_COMPUTER = "this computer"  # the source of a request made on this computer


def _line(event: str, fields: Mapping[str, Any]) -> None:
    """One line of the record: ``Scan <event> {the fields as JSON}``."""
    log.info("Scan %s %s", event, json.dumps(fields, ensure_ascii=True))


def scan_started(source: str, target: str, addresses: Sequence[str], ports: Sequence[int],
                 protocols: Sequence[str], engine: str, online: Mapping[str, str]) -> str:
    """Writes the line of a scan that starts: where its request came from, the target as typed,
    the addresses it was resolved to, the ports as ranges, the protocols, the engine and what
    may go online; the scan's id, which its end line repeats."""
    scan_id = secrets.token_urlsafe(6)
    _line("started", {"scan_id": scan_id, "source": source, "target": target,
                      "addresses": list(addresses), "ports": port_ranges(ports),
                      "protocols": list(protocols), "engine": engine, "online": dict(online)})
    return scan_id


def scan_ended(scan_id: str, outcome: str, open_ports: int, engine_failed: int,
               seconds: float) -> None:
    """Writes the line of a scan that ended: ``done``, ``stopped`` (a cancel, the time limit or
    Ctrl+C) or ``failed`` (an error nobody expected), the open ports found, how many targets the
    engine couldn't scan and how long it took."""
    _line("ended", {"scan_id": scan_id, "outcome": outcome, "open_ports": open_ports,
                    "engine_failed": engine_failed, "seconds": round(seconds, 1)})
