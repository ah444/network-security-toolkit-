"""A scan, start to end (B11, S16, step 1.4e): each target pinged, its ports checked, the
open ones looked up online (each product once, in parallel), the report entries made and,
if asked, saved. What the user sees and answers on the way goes through a ``ScanView``
(``toolkit.cli.ui.scan.ConsoleScan``), so this module never prints or asks.

A scan can be stopped (v5.2 step 2.4c, the web scan's Cancel): once its ``stop`` event is
set, nothing more is sent, to the targets or online, and ``run_scan`` raises ScanStopped
without a summary or a file.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ...data.keys import KEY_VARIABLES
from ...data.recon.local import is_this_computer, this_os
from ...data.recon.lookup import INTEL_WORKERS
from ...data.recon.nmap import NmapMissing, NmapOptions
from ...data.recon.nmap_run import NmapFailed
from ...data.recon.nvd import NVD_INTERVAL
from ...data.recon.ping import ping_host
from ...data.recon.software import nmap_software, product_version
from ...data.recon.scan_files import EXPORT_FORMATS as EXPORT_FORMATS
from ...data.recon.scan_files import ReportEntry as ReportEntry  # what the UI shows
from ...data.recon.scan_files import export_results
from ...data.recon.scan_files import report_text as report_text  # a download's text
from ...data.recon.scanner import PortResult as PortResult
from ...data.recon.scanner import ScanStopped as ScanStopped  # raised once a scan is stopped
from ...data.recon.scanner import Progress
from ...data.recon.text import one_line as one_line  # network text, shown on one line
from ...data.settings import ScanConfig
from .engines import DEFAULT_ENGINE, describe, engine_named
from .intel import Intel, LookupResult
from .os_guess import OsHint as OsHint  # what the UI shows
from .os_guess import guess_os, os_hint
from .report import make_entry
from .targets import check_limits, is_ip_literal, is_public_ip, parse_target

PROTOCOLS = {"tcp": ["tcp"], "udp": ["udp"], "both": ["tcp", "udp"]}


@dataclass(frozen=True)
class ScanRequest:
    """What to scan and what may leave this computer: from the flags (``scan TARGET``),
    or from the questions (``interactive``: AbuseIPDB, hosts that don't answer ping and
    the export are then asked during the scan instead)."""

    target_text: str
    targets: list[str]
    ports: list[int]
    protocols: list[str]
    verbose: bool = False
    use_ai: bool = False
    abuseipdb: bool = False
    scan_unresponsive: bool = False
    export: str | None = None
    interactive: bool = False
    engine: str = DEFAULT_ENGINE  # a name in engines.ENGINES (step N1)
    nmap_options: NmapOptions = NmapOptions()  # nmap's choices, for --engine nmap (step N4)


@dataclass(frozen=True)
class OnlinePlan:
    """What a scan may send to each online service, or why it sends nothing (step A5)."""

    nvd: str
    ai: str
    abuseipdb: str


def _ai_plan(request: ScanRequest, keys: Mapping[str, str]) -> str:
    """The AI explanation: on, or why not (no key, not asked for, the answer no)."""
    if request.use_ai:
        return "on, the software name and version only"
    if not keys.get("openrouter"):
        return f"off: no {KEY_VARIABLES['openrouter']}"
    return "off: you said no" if request.interactive else "off: not asked for (--ai)"


def _abuse_plan(request: ScanRequest, keys: Mapping[str, str]) -> str:
    """The AbuseIPDB check: for the public addresses only, or why not."""
    if not any(is_public_ip(target) for target in request.targets):
        return "off: no public address (a private one is never sent)"
    if not keys.get("abuseipdb"):
        return f"off: no {KEY_VARIABLES['abuseipdb']}"
    if request.interactive:
        return "asked before each public address"
    return "on, the public addresses only" if request.abuseipdb else \
        "off: not asked for (--abuseipdb)"


def online_plan(request: ScanRequest, keys: Mapping[str, str]) -> OnlinePlan:
    """What this scan may send online and why not, from the request (the flags or the answers)
    and the keys that are set."""
    pace = "" if keys.get("nvd") else (
        f"; no {KEY_VARIABLES['nvd']}, so one request every {NVD_INTERVAL[False]:g} s")
    return OnlinePlan(f"on, the software name and version only{pace}",
                      _ai_plan(request, keys), _abuse_plan(request, keys))


class ScanView(Protocol):
    """What a scan shows and asks on its way (the console: ``toolkit.cli.ui.scan``)."""

    def resolved(self, text: str, address: str) -> None:
        """Once, when the target was a host name: it resolved to ``address``. The DNS query has
        already gone out (see ``resolve_targets``)."""

    def abuse_skipped(self, target: str) -> None:
        """A public ``target`` in an interactive scan without an AbuseIPDB key: nothing is sent."""

    def ask_abuse(self, target: str) -> bool:
        """Interactive scans, a public ``target`` and a key: True sends it to AbuseIPDB. Asked
        before the ping, so nothing has reached the host yet."""

    def abuse_report(self, target: str, data: Mapping[str, Any] | None,
                     error: str | None) -> None:
        """AbuseIPDB's answer for ``target``: ``data`` (its fields, text from the network: show it
        cleaned), or None and ``error`` saying why the lookup failed."""

    def host_down(self, target: str) -> None:
        """``target`` did not answer the ping; ``ask_scan_anyway`` (interactive) or
        ``unresponsive_skipped`` (no ``--scan-unresponsive``) may follow."""

    def ask_scan_anyway(self) -> bool:
        """Interactive scans, after ``host_down``: True scans the host anyway."""

    def unresponsive_skipped(self) -> None:
        """A host that did not answer was skipped (no ``--scan-unresponsive``)."""

    def scanning(self, target: str, ttl: int | None, guess: tuple[str, int] | None) -> None:
        """The port checks of ``target`` start; ``guess`` is the OS guess from the ping's ``ttl`` (a
        name and the hops away), None without a TTL."""

    def progress(self) -> Progress:
        """Verbose scans only: a new ProgressSink for this target's checks; it is advanced from the
        scanner's worker threads, so it must be thread-safe."""

    def silent_udp(self, ports: Sequence[int]) -> None:
        """UDP ports that sent no reply (``open|filtered``), sorted; there may be many."""

    def no_open_ports(self, target: str) -> None:
        """No port of ``target`` is open: it adds no entry to the report."""

    def open_ports(self, results: Sequence[PortResult]) -> None:
        """The open ports, before their lookups; ``service`` and ``banner`` come from the network:
        show them cleaned (``one_line``)."""

    def os_hint(self, target: str, hint: OsHint) -> None:
        """After the port checks (step A3): what the TTL, the banners and the open ports say about
        the OS of ``target``, or that it is this computer; its text is the scanner's own words."""

    def looking_up(self, products: int, workers: int) -> None:
        """``products`` distinct products (at least one) are about to be looked up online, up to
        ``workers`` at a time."""

    def entry(self, entry: ReportEntry, first: bool) -> None:
        """One open port's report entry, in port order; ``first`` for the target's first entry."""

    def engine_notice(self, lines: Sequence[str]) -> None:
        """Once, before ``lookups_notice``: the scan engine and its choices (``engines.describe``;
        for nmap, its own version scripts too), the scanner's own words."""

    def lookups_notice(self, online: OnlinePlan) -> None:
        """Once, before the first target: what this scan may send to each online service, or why
        it sends nothing (step A5)."""

    def scan_failed(self, target: str, message: str) -> None:
        """The engine could not scan ``target`` (nmap missing or failed, ``message`` saying why):
        it adds no entry, and the scan goes on with the next target."""

    def target_done(self) -> None:
        """After each target, scanned or skipped."""

    def summary(self, targets: int, entries: int, failed: Sequence[str], seconds: float,
                sent: Mapping[str, int]) -> None:
        """After the last target: the targets, the entries, the failed lookups (as ``port/protocol
        error``), the seconds since the command started and the requests sent to each online
        service (``Intel.sent``)."""

    def nothing_to_export(self) -> None:
        """No entries, and an export was asked for (interactive or ``--export``)."""

    def ask_export(self) -> str | None:
        """Interactive scans with entries: a format of ``EXPORT_FORMATS``, or None for no file; any
        other answer is refused through ``unknown_format``."""

    def unknown_format(self) -> None:
        """The answer to ``ask_export`` is not a known format: nothing is saved."""

    def exported(self, path: Path) -> None:
        """The report was saved as ``path``, a new file (never overwriting one)."""


def resolve_targets(text: str, limits: ScanConfig, view: ScanView) -> list[str]:
    """The addresses to scan from what the user typed: an address, a network, or a host name, which
    is resolved here (its first address, shown through ``view.resolved``). ValueError with a message
    for a bad target or a network above the hosts limit.

    This runs before the authorization question, so for a host name the DNS query (the name, to the
    system's resolver) is sent before the user confirms; nothing reaches the target itself before
    then."""
    targets = parse_target(text, limits.max_hosts)
    if "/" not in text and not is_ip_literal(text):
        view.resolved(text, targets[0])
    return targets


def check_scan(targets: Sequence[str], ports: Sequence[int], protocols: Sequence[str],
               limits: ScanConfig) -> None:
    """ValueError with the message when the scan is above the limits (the hosts, or the host x port
    x protocol checks). Called before the authorization question, so a scan that is too big is
    refused without asking it."""
    problem = check_limits(len(targets), len(targets) * len(ports) * len(protocols), limits)
    if problem:
        raise ValueError(problem)


def _reputation(target: str, request: ScanRequest, intel: Intel, view: ScanView) -> None:
    """AbuseIPDB, for a public address only (a private one is never sent): with a key, when
    the user says yes (asked) or passed ``--abuseipdb``; asked without a key, the skip is
    shown. Runs before the ping, so nothing reaches the host before this decision."""
    if not is_public_ip(target):
        return
    has_key = bool(intel.keys.get("abuseipdb"))
    if request.interactive and not has_key:
        view.abuse_skipped(target)
    elif has_key and (view.ask_abuse(target) if request.interactive else request.abuseipdb):
        data, error = intel.abuse_report(target)
        view.abuse_report(target, data, error)


def _ping_decision(target: str, request: ScanRequest, view: ScanView) -> tuple[bool, int | None]:
    """Ping the host: whether to scan it, and the reply's TTL. A host that doesn't answer is
    scanned only if the user says so (asked) or passed ``--scan-unresponsive``."""
    alive, ttl = ping_host(target)
    if alive:
        return True, ttl
    view.host_down(target)
    if request.interactive:
        return view.ask_scan_anyway(), ttl
    if not request.scan_unresponsive:
        view.unresponsive_skipped()
        return False, ttl
    return True, ttl


def _software(result: PortResult) -> str | None:
    """What is looked up for a port: nmap's product and version when it gave them (the A1
    names, ``nmap_software``), else the product and version in the banner."""
    if result.product:
        return nmap_software(result.product, result.version)
    return product_version(result.banner)


def _entries(target: str, open_ports: list[PortResult], intel: Intel, view: ScanView,
             os_text: str, engine: str) -> list[ReportEntry]:
    """The report entry of each open port, with its product's lookup (each product once, in
    parallel, step 1.4e; a port without a product gets an empty one), the host's OS hint
    (``os_text``) and the engine's name; each shown as made."""
    products = [_software(result) for result in open_ports]
    distinct = [software for software in dict.fromkeys(products) if software]
    if distinct:
        view.looking_up(len(distinct), INTEL_WORKERS)
    looked_up = intel.lookup_many(distinct)
    entries: list[ReportEntry] = []
    for result, software in zip(open_ports, products, strict=True):
        found: LookupResult = (
            looked_up[software] if software else
            {"cves": [], "cve_source": "", "max_cvss": None, "severity": "UNKNOWN",
             "ai_explanation": "", "errors": []})
        entry = make_entry(target, result, software, found, os_text, engine)
        view.entry(entry, first=not entries)
        entries.append(entry)
    return entries


def check_stop(stop: threading.Event | None) -> None:
    """ScanStopped once ``stop`` is set (None: a scan nobody stops, the command line's)."""
    if stop is not None and stop.is_set():
        raise ScanStopped("the scan was stopped")


def scan_target(target: str, request: ScanRequest, intel: Intel, view: ScanView,
                stop: threading.Event | None = None) -> list[ReportEntry]:
    """Scan one host and look up its open ports; returns the report entries. NmapMissing or
    NmapFailed when nmap can't scan it (``run_scan`` shows it and goes on); ScanStopped once
    ``stop`` is set, before AbuseIPDB, the ping and the port checks, during the checks (the
    engine) and the lookups (``Intel``)."""
    check_stop(stop)
    _reputation(target, request, intel, view)
    check_stop(stop)
    scan, ttl = _ping_decision(target, request, view)
    if not scan:
        return []
    check_stop(stop)
    view.scanning(target, ttl, guess_os(ttl))

    engine = engine_named(request.engine, request.nmap_options)
    results = engine.scan(target, request.ports, request.protocols,
                          view.progress() if request.verbose else None, stop)
    open_ports = [r for r in results if r.state == "open"]
    silent_udp = [r.port for r in results if r.state == "open|filtered"]
    if silent_udp:
        view.silent_udp(silent_udp)
    if open_ports:
        view.open_ports(open_ports)
    else:
        view.no_open_ports(target)
    hint = os_hint(ttl, open_ports, is_this_computer(target), this_os(),
                   nmap_os=engine.os_matches)
    if hint:
        view.os_hint(target, hint)
    return _entries(target, open_ports, intel, view, hint.summary() if hint else "",
                    engine.name)


def run_scan(request: ScanRequest, keys: Mapping[str, str], start_time: float,
             view: ScanView, stop: threading.Event | None = None) -> int:
    """Scan every target, show the summary, export (asked, or ``--export``); the exit code:
    1 when the engine could not scan a target (the others are still scanned), else 0.
    ScanStopped once ``stop`` is set (``scan_target``), with no summary and no file."""
    intel = Intel(keys, request.use_ai, stop=stop)
    view.engine_notice(describe(request.engine, request.nmap_options))
    view.lookups_notice(online_plan(request, keys))

    entries: list[ReportEntry] = []
    code = 0
    for target in request.targets:
        try:
            entries.extend(scan_target(target, request, intel, view, stop))
        except (NmapMissing, NmapFailed) as error:
            view.scan_failed(target, str(error))
            code = 1
        view.target_done()
    check_stop(stop)

    failed =[f"{e['port']}/{e['protocol']} {error}" for e in entries
              for error in e["lookup_errors"]]
    view.summary(len(request.targets), len(entries), failed, time.time() - start_time,
                 intel.sent())

    if not entries:
        if request.interactive or request.export:
            view.nothing_to_export()
        return code
    fmt = request.export
    if request.interactive:
        fmt = view.ask_export()
        if fmt is not None and fmt not in EXPORT_FORMATS:
            view.unknown_format()
            return code
    if fmt:
        view.exported(export_results(entries, fmt))
    return code
