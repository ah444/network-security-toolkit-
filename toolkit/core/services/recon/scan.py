"""A scan, start to end (B11, S16, step 1.4e): each target pinged, its ports checked, the
open ones looked up online (each product once, in parallel), the report entries made and,
if asked, saved. What the user sees and answers on the way goes through a ``ScanView``
(``toolkit.ui.cli.scan.ConsoleScan``), so this module never prints or asks.
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
    """What a scan shows and asks on its way (the console: ``toolkit.ui.cli.scan``)."""

    def resolved(self, text: str, address: str) -> None: ...

    def abuse_skipped(self, target: str) -> None: ...

    def ask_abuse(self, target: str) -> bool: ...

    def abuse_report(self, target: str, data: Mapping[str, Any] | None,
                     error: str | None) -> None: ...

    def host_down(self, target: str) -> None: ...

    def ask_scan_anyway(self) -> bool: ...

    def unresponsive_skipped(self) -> None: ...

    def scanning(self, target: str, ttl: int | None, guess: tuple[str, int] | None) -> None: ...

    def progress(self) -> Progress: ...

    def silent_udp(self, ports: Sequence[int]) -> None: ...

    def no_open_ports(self, target: str) -> None: ...

    def open_ports(self, results: Sequence[PortResult]) -> None: ...

    def looking_up(self, products: int, workers: int) -> None: ...

    def entry(self, entry: ReportEntry, first: bool) -> None: ...

    def lookups_notice(self) -> None: ...

    def target_done(self) -> None: ...

    def summary(self, targets: int, entries: int, failed: Sequence[str],
                seconds: float) -> None: ...

    def nothing_to_export(self) -> None: ...

    def ask_export(self) -> str | None: ...

    def unknown_format(self) -> None: ...

    def exported(self, path: Path) -> None: ...


def resolve_targets(text: str, limits: ScanConfig, view: ScanView) -> list[str]:
    targets = parse_target(text, limits.max_hosts)
    if "/" not in text and not is_ip_literal(text):
        view.resolved(text, targets[0])
    return targets


def check_scan(targets: Sequence[str], ports: Sequence[int], protocols: Sequence[str],
               limits: ScanConfig) -> None:
    problem = check_limits(len(targets), len(targets) * len(ports) * len(protocols), limits)
    if problem:
        raise ValueError(problem)


def scan_target(target: str, request: ScanRequest, intel: Intel,
                view: ScanView) -> list[ReportEntry]:
    """Scan one host and look up its open ports; returns the report entries."""
    if is_public_ip(target):
        has_key = bool(intel.keys.get("abuseipdb"))
        if request.interactive and not has_key:
            view.abuse_skipped(target)
        elif has_key and (view.ask_abuse(target) if request.interactive else request.abuseipdb):
            data, error = intel.abuse_report(target)
            view.abuse_report(target, data, error)

    alive, ttl = ping_host(target)
    if not alive:
        view.host_down(target)
        if request.interactive:
            if not view.ask_scan_anyway():
                return []
        elif not request.scan_unresponsive:
            view.unresponsive_skipped()
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

    view.open_ports(open_ports)
    products = [product_version(result.banner) for result in open_ports]
    distinct = [software for software in dict.fromkeys(products) if software]
    if distinct:
        view.looking_up(len(distinct), INTEL_WORKERS)
    looked_up = intel.lookup_many(distinct)  # each product once, in parallel (step 1.4e)
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
