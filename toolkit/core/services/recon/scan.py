"""A scan, start to end (B11, S16, step 1.4e): each target pinged, its ports checked, the
open ones looked up online (each product once, in parallel), the report entries made and,
if asked, saved. What the user sees and answers on the way goes through a ``ScanView``
(``toolkit.cli.ui.scan.ConsoleScan``), so this module never prints or asks.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ...data.recon.lookup import INTEL_WORKERS
from ...data.recon.ping import ping_host
from ...data.recon.probes import product_version
from ...data.recon.scan_files import EXPORT_FORMATS as EXPORT_FORMATS
from ...data.recon.scan_files import ReportEntry as ReportEntry  # what the UI shows
from ...data.recon.scan_files import export_results
from ...data.recon.scanner import PortResult as PortResult
from ...data.recon.scanner import Progress, scan_ports
from ...data.recon.text import one_line as one_line  # network text, shown on one line
from ...data.settings import ScanConfig
from .intel import Intel, LookupResult
from .os_guess import guess_os
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

    def looking_up(self, products: int, workers: int) -> None:
        """``products`` distinct products (at least one) are about to be looked up online, up to
        ``workers`` at a time."""

    def entry(self, entry: ReportEntry, first: bool) -> None:
        """One open port's report entry, in port order; ``first`` for the target's first entry."""

    def lookups_notice(self) -> None:
        """Once, before the first target: what the CVE lookups send online."""

    def target_done(self) -> None:
        """After each target, scanned or skipped."""

    def summary(self, targets: int, entries: int, failed: Sequence[str],
                seconds: float) -> None:
        """After the last target: the targets, the entries, the failed lookups (as ``port/protocol
        error``) and the seconds since the command started."""

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


def _entries(target: str, open_ports: list[PortResult], intel: Intel,
             view: ScanView) -> list[ReportEntry]:
    """The report entry of each open port, with its product's lookup (each product once, in
    parallel, step 1.4e; a port without a product gets an empty one); each shown as made."""
    view.open_ports(open_ports)
    products = [product_version(result.banner) for result in open_ports]
    distinct = [software for software in dict.fromkeys(products) if software]
    if distinct:
        view.looking_up(len(distinct), INTEL_WORKERS)
    looked_up = intel.lookup_many(distinct)
    entries: list[ReportEntry] = []
    for result, software in zip(open_ports, products, strict=True):
        found: LookupResult = (
            looked_up[software] if software else
            {"cves": [], "max_cvss": None, "severity": "UNKNOWN", "ai_explanation": "",
             "errors": []})
        entry = make_entry(target, result, software, found)
        view.entry(entry, first=not entries)
        entries.append(entry)
    return entries


def scan_target(target: str, request: ScanRequest, intel: Intel,
                view: ScanView) -> list[ReportEntry]:
    """Scan one host and look up its open ports; returns the report entries."""
    _reputation(target, request, intel, view)
    scan, ttl = _ping_decision(target, request, view)
    if not scan:
        return []
    view.scanning(target, ttl, guess_os(ttl))

    results = scan_ports(target, request.ports, request.protocols,
                         view.progress() if request.verbose else None)
    open_ports = [r for r in results if r.state == "open"]
    silent_udp = [r.port for r in results if r.state == "open|filtered"]
    if silent_udp:
        view.silent_udp(silent_udp)
    if not open_ports:
        view.no_open_ports(target)
        return []
    return _entries(target, open_ports, intel, view)


def run_scan(request: ScanRequest, keys: Mapping[str, str], start_time: float,
             view: ScanView) -> int:
    """Scan every target, show the summary, export (asked, or ``--export``); the exit code."""
    intel = Intel(keys, request.use_ai)
    view.lookups_notice()

    entries: list[ReportEntry] = []
    for target in request.targets:
        entries.extend(scan_target(target, request, intel, view))
        view.target_done()

    failed = [f"{e['port']}/{e['protocol']} {error}" for e in entries
              for error in e["lookup_errors"]]
    view.summary(len(request.targets), len(entries), failed, time.time() - start_time)

    if not entries:
        if request.interactive or request.export:
            view.nothing_to_export()
        return 0
    fmt = request.export
    if request.interactive:
        fmt = view.ask_export()
        if fmt is not None and fmt not in EXPORT_FORMATS:
            view.unknown_format()
            return 0
    if fmt:
        view.exported(export_results(entries, fmt))
    return 0
