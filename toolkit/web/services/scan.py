"""The web port scan (v5.2 step 2.4d): the command line's scanner
(``toolkit.core.services.recon.scan``) run as a job of the kind ``scan``, one at a time for the
whole server (its own queue, next to a training) and one at a time in a session.

The scan contract holds as on the command line, and is stricter in one place: nothing at all
is sent, not even a DNS query for a host name, before the request says the user is
authorised to scan the target (``authorized``); then the choices are checked (no network),
then the engine (nmap's own ``--version``, on this computer), then the target is resolved
and the limits checked; a refusal is a 400 at once. AbuseIPDB and the AI explanation are
opt-in and refused without their key; a private address never goes to AbuseIPDB (the core
scan's rule); the keys never leave the server. Nothing is asked during the scan: the form
answered everything. The last finished scan stays in the session (never saved, 4a); a
cancelled one keeps nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ...core.data.keys import KEY_VARIABLES, load_api_keys
from ...core.data.logs import add_secrets
from ...core.data.recon.nmap import NEEDS_PRIVILEGES, VERSION_SCRIPTS_NOTE
from ...core.data.settings import ScanConfig
from ...core.services.recon.engines import (
    DEFAULT_ENGINE,
    ENGINES,
    INTENSITIES,
    SCRIPTS,
    TIMINGS,
    NmapEngine,
    NmapOptions,
    engine_problem,
    has_privileges,
)
from ...core.services.recon.scan import (
    PROTOCOLS,
    ReportEntry,
    ScanRequest,
    ScanStopped,
    check_scan,
    report_text,
    resolve_targets,
    run_scan,
)
from ...core.services.recon.targets import DEFAULT_PORTS, parse_ports
from .errors import (
    BAD_REQUEST,
    BUSY,
    CANCELLED,
    ENGINE_UNAVAILABLE,
    NO_SCAN,
    NOT_AUTHORIZED,
    SCAN_LIMIT,
    TIME_LIMIT,
    ServiceError,
)
from .jobs import SCAN, Job, JobRunner
from .scan_view import WebScan, nmap_missing
from .workspace import Workspace

log = logging.getLogger(__name__)

LIMIT_SETTINGS = ("TOOLKIT_SCAN_MAX_HOSTS", "TOOLKIT_SCAN_MAX_PROBES")  # named by a refusal
MIME_TYPES = {"txt": "text/plain", "json": "application/json", "csv": "text/csv"}


@dataclass(frozen=True)
class ScanChoices:
    """``POST /api/v1/jobs/scan``: what the page's form chose (the request's schema too). The
    nmap-only choices are empty unless the engine is nmap."""

    target: str = ""
    ports: str = DEFAULT_PORTS
    protocol: str = "tcp"
    engine: str = DEFAULT_ENGINE
    service_detection: str = ""  # "" is nmap's default level
    timing: str = ""  # "" is nmap's default timing
    scripts: bool = False  # the toolkit's allowed scripts (nmap's "safe" set of the CLI)
    os_detect: bool = False
    abuseipdb: bool = False  # public addresses only, with ABUSEIPDB_API_KEY
    ai: bool = False  # the software names only, with OPENROUTER_API_KEY
    scan_unresponsive: bool = False
    authorized: bool = False  # the user confirmed being authorised to scan the target


@dataclass(frozen=True)
class FinishedScan:
    """A session's last finished scan, in memory only: its entries for the download and its
    answer for the page."""

    entries: list[ReportEntry]
    answer: dict[str, Any]
    stamp: str  # when it ended (UTC), in the downloaded file's name


# --- What the page can choose --------------------------------------------------------------

def scan_options(keys: Mapping[str, str], limits: ScanConfig) -> dict[str, Any]:
    """The engines (nmap: whether it can run here, and with raw packets), each choice, the
    limits, and whether each API key is set (never a key itself)."""
    nmap_problem = NmapEngine().available()
    raw_packets = nmap_problem is None and has_privileges()
    if nmap_problem is not None:
        log.info("nmap can't be offered: %s", nmap_problem)
    defaults = NmapOptions()
    return {
        "engines": [
            {"id": "builtin", "available": True, "reason": None},
            {"id": "nmap", "available": nmap_problem is None,
             "reason": None if nmap_problem is None else nmap_missing(),
             "raw_packets": raw_packets},
        ],
        "default_engine": DEFAULT_ENGINE,
        "default_ports": DEFAULT_PORTS,
        "protocols": list(PROTOCOLS),
        "nmap": {"service_detection": list(INTENSITIES), "timing": list(TIMINGS),
                 "default_service_detection": defaults.service_detection,
                 "default_timing": defaults.timing, "scripts": list(SCRIPTS),
                 "udp_and_os_detection": raw_packets, "raw_packets_note": NEEDS_PRIVILEGES,
                 "version_scripts_note": VERSION_SCRIPTS_NOTE},
        "limits": {"max_hosts": limits.max_hosts, "max_checks": limits.max_probes,
                   "max_minutes": limits.max_minutes},
        "keys": {name: bool(keys.get(name)) for name in KEY_VARIABLES},
    }


# --- Starting a scan -----------------------------------------------------------------------

def _refuse(message: str, code: str = BAD_REQUEST) -> ServiceError:
    """A refusal (400 unless its code says otherwise) with a message for the user."""
    return ServiceError(message, code)


def _checked(choices: ScanChoices, keys: Mapping[str, str]) -> tuple[list[int], list[str],
                                                                     NmapOptions]:
    """The ports, the protocols and nmap's options from the choices, checked without any
    network; a ServiceError (400) for a bad or a refused one."""
    if choices.protocol not in PROTOCOLS:
        raise _refuse(f"'protocol' must be one of {', '.join(PROTOCOLS)}.")
    try:
        ports = parse_ports(choices.ports or DEFAULT_PORTS)
    except ValueError as error:
        raise _refuse(str(error)) from None
    if choices.engine not in ENGINES:
        raise _refuse(f"'engine' must be one of {', '.join(ENGINES)}.")
    nmap_choices = choices.service_detection or choices.timing or choices.scripts or \
        choices.os_detect
    if choices.engine != "nmap" and nmap_choices:
        raise _refuse("Service detection, timing, scripts and OS detection are nmap's: "
                      "choose the nmap engine for them.")
    if choices.service_detection and choices.service_detection not in INTENSITIES:
        raise _refuse(f"'service_detection' must be one of {', '.join(INTENSITIES)}.")
    if choices.timing and choices.timing not in TIMINGS:
        raise _refuse(f"'timing' must be one of {', '.join(TIMINGS)}.")
    for wanted, name, what in ((choices.ai, "openrouter", "an AI explanation"),
                               (choices.abuseipdb, "abuseipdb", "a reputation check")):
        if wanted and not keys.get(name):
            raise _refuse(f"No {KEY_VARIABLES[name]} is set on the server, so no {what}.")
    defaults = NmapOptions()
    options = NmapOptions(choices.service_detection or defaults.service_detection,
                          choices.timing or defaults.timing,
                          "safe" if choices.scripts else "", choices.os_detect)
    return ports, PROTOCOLS[choices.protocol], options


def _targets(text: str, ports: Sequence[int], protocols: Sequence[str], limits: ScanConfig,
             view: WebScan) -> list[str]:
    """The addresses to scan (a host name is resolved here, after the authorization), within
    the limits; SCAN_LIMIT above them, BAD_REQUEST for a bad target."""
    try:
        targets = resolve_targets(text, limits, view)
        check_scan(targets, ports, protocols, limits)
    except ValueError as error:
        message = str(error)
        raise _refuse(message, SCAN_LIMIT if any(name in message for name in LIMIT_SETTINGS) else BAD_REQUEST) \
            from None
    return targets


def require_authorized(choices: ScanChoices) -> None:
    """NOT_AUTHORIZED unless the user confirmed being authorised to scan the target: the first
    check of every scan, before anything else is done."""
    if not choices.authorized:
        raise _refuse("Confirm that you are authorised to scan this target: only scan systems "
                      "you own or have written permission to test.", NOT_AUTHORIZED)


def start_scan(ws: Workspace, choices: ScanChoices, runner: JobRunner,
               limits: ScanConfig) -> Job:
    """Check a scan and queue it as a job of the session ``ws``; the job's result is its answer
    and the session keeps it (``ws.scan``). The order is the contract: the authorization first,
    nothing sent before it (module docstring). One scan at a time per session (else BUSY); its
    lock and a lease are held until the job ends."""
    require_authorized(choices)
    keys = load_api_keys()
    ports, protocols, options = _checked(choices, keys)
    if not ws.scanning.acquire(blocking=False):
        raise _refuse("A scan is already running in this session. Wait for it to finish or "
                      "cancel it.", BUSY)
    try:
        problem = engine_problem(choices.engine, options, protocols)
        if problem:
            log.info("The %s engine can't run a scan: %s", choices.engine, problem)
            raise _refuse(f"The {choices.engine} engine can't run this scan: " + (
                nmap_missing() if NmapEngine().available() else problem), ENGINE_UNAVAILABLE)
        job = Job(kind=SCAN, work=lambda job: _run(ws, request, keys, job, view),
                  on_end=lambda: _release(ws), max_seconds=limits.max_minutes * 60,
                  steps=lambda done, total: view.phase())
        view = WebScan(job)
        targets = _targets(choices.target.strip(), ports, protocols, limits, view)
        request = ScanRequest(
            choices.target.strip(), targets, ports, protocols, verbose=True, use_ai=choices.ai,
            abuseipdb=choices.abuseipdb, scan_unresponsive=choices.scan_unresponsive,
            engine=choices.engine, nmap_options=options)
        view.hosts = len(targets)
    except BaseException:
        ws.scanning.release()
        raise
    ws.acquire_lease()
    ws.add_job(job)
    runner.submit(job)
    return job


def _release(ws: Workspace) -> None:
    """Gives back the session's scan lock and lease when the job ends."""
    ws.release_lease()
    ws.scanning.release()


def _run(ws: Workspace, request: ScanRequest, keys: Mapping[str, str], job: Job,
         view: WebScan) -> dict[str, Any]:
    """The job's work, in the scan queue's thread: the core scan with the job's cancel as its
    stop, and a timer that stops it after ``max_seconds`` (TIME_LIMIT). Only a finished scan is
    kept (``ws.scan``); one line in the log for each scan."""
    add_secrets(*keys.values())  # the keys never reach the log
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        job.cancel_requested.set()

    timer = threading.Timer(job.max_seconds, expire)
    timer.daemon = True
    timer.start()
    began, outcome = time.time(), "failed"
    try:
        code = run_scan(request, keys, began, view, job.cancel_requested)
        outcome = "done"
    except ScanStopped:
        outcome = "stopped"
        raise job.stopped(TIME_LIMIT if timed_out.is_set() else CANCELLED) from None
    finally:
        timer.cancel()
        log.info("Scan %s: %r, %d host(s), %d port(s) x %s, engine %s, %d open port(s), %.0f s",
                 outcome, request.target_text, len(request.targets), len(request.ports),
                 "+".join(request.protocols), request.engine, len(view.entries),
                 time.time() - began)
    answer = view.answer(request, code)
    with ws.lock:
        ws.scan = FinishedScan(list(view.entries), answer,
                               datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    return answer


# --- The report download -------------------------------------------------------------------

def scan_report(ws: Workspace, fmt: str) -> tuple[str, str, str]:
    """The session's last finished scan as a file: (its text, its media type, its name);
    NO_SCAN before one, BAD_REQUEST for another format."""
    if fmt not in MIME_TYPES:
        raise _refuse(f"'format' must be one of {', '.join(MIME_TYPES)}.")
    if ws.scan is None:
        raise _refuse("No finished scan in this session yet.", NO_SCAN)
    return report_text(ws.scan.entries, fmt), MIME_TYPES[fmt], f"scan-{ws.scan.stamp}.{fmt}"
