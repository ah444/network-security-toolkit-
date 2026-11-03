"""The scan history (F6a): every finished scan, the CLI's and the web's, saved by ``run_scan`` as
one record, under the id of its lines in the log (HS3, ``record``).

A record holds what was asked (the target as typed, its addresses, the ports, the protocols,
the engine and nmap's choices, the switches, where the request came from), what may have gone
online and why not (``online_plan``), what each host gave (scanned, skipped because it didn't
answer the ping, or failed, with why), AbuseIPDB's answers, the report entries and the totals.
Never a key or a token: they are not in the request. Network text is kept as the entries have
it (cleaned, ``report``); a page shows it as text. A stopped or failed scan saves nothing, as a cancelled
training keeps nothing. A record that can't be saved (a full disk, HR4c) leaves the scan as it
was: the view says why. Nothing here deletes a record.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...data.recon.nmap import port_ranges
from ...data.recon.scan_files import ReportEntry
from ...data.recon.scan_records import write_scan
from ...data.recon.scanner import PortResult, Progress
from ...data.recon.text import one_line

if TYPE_CHECKING:
    from .online import OnlinePlan
    from .os_guess import OsHint
    from .scan import ScanRequest, ScanView

log = logging.getLogger(__name__)

SCHEMA = 1  # a record's version: a newer one is read only by a newer toolkit
SCAN_ID = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")  # a training run's id has the same form


def new_scan_id() -> str:
    """A new scan's id: when it started (UTC) and 8 random hex digits."""
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def is_scan_id(text: str) -> bool:
    """Whether ``text`` is a scan id (and so a safe file name)."""
    return SCAN_ID.fullmatch(text) is not None


class ScanNotes:
    """The ``ScanView`` ``run_scan`` reports to: it passes every call to the user's view and
    notes what the record needs, how each host ended, its OS hint, AbuseIPDB's answers, the
    entries and the totals."""

    def __init__(self, view: ScanView, targets: Sequence[str]) -> None:
        self.view = view
        self.hosts: dict[str, dict[str, str]] = {
            target: {"target": target, "state": "skipped", "os": "", "message": ""}
            for target in targets}  # a host neither scanned nor failed was skipped (the ping)
        self.reputation: dict[str, dict[str, str]] = {}
        self.entries: list[ReportEntry] = []
        self.totals: dict[str, Any] = {}

    def _host(self, target: str) -> dict[str, str]:
        """The host's entry (a target the request didn't name gets one too)."""
        return self.hosts.setdefault(
            target, {"target": target, "state": "skipped", "os": "", "message": ""})

    def resolved(self, text: str, address: str) -> None:
        """Passed on."""
        self.view.resolved(text, address)

    def abuse_skipped(self, target: str) -> None:
        """Passed on."""
        self.view.abuse_skipped(target)

    def ask_abuse(self, target: str) -> bool:
        """The view's answer."""
        return self.view.ask_abuse(target)

    def abuse_report(self, target: str, data: Mapping[str, Any] | None,
                     error: str | None) -> None:
        """AbuseIPDB's answer, each field cleaned to one line, or why it failed."""
        self.reputation[target] = {"error": one_line(error or "failed", 300)} if data is None \
            else {str(name): one_line(value, 200) for name, value in data.items()}
        self.view.abuse_report(target, data, error)

    def host_down(self, target: str) -> None:
        """Passed on (the host stays skipped unless it is scanned after all)."""
        self.view.host_down(target)

    def ask_scan_anyway(self) -> bool:
        """The view's answer."""
        return self.view.ask_scan_anyway()

    def unresponsive_skipped(self) -> None:
        """Passed on."""
        self.view.unresponsive_skipped()

    def scanning(self, target: str, ttl: int | None, guess: tuple[str, int] | None) -> None:
        """The host is scanned."""
        self._host(target)["state"] = "scanned"
        self.view.scanning(target, ttl, guess)

    def progress(self) -> Progress:
        """The view's progress."""
        return self.view.progress()

    def silent_udp(self, ports: Sequence[int]) -> None:
        """Passed on."""
        self.view.silent_udp(ports)

    def no_open_ports(self, target: str) -> None:
        """Passed on."""
        self.view.no_open_ports(target)

    def open_ports(self, results: Sequence[PortResult]) -> None:
        """Passed on."""
        self.view.open_ports(results)

    def os_hint(self, target: str, hint: OsHint) -> None:
        """The host's OS hint, as its entries have it."""
        self._host(target)["os"] = hint.summary()
        self.view.os_hint(target, hint)

    def looking_up(self, products: int, workers: int) -> None:
        """Passed on."""
        self.view.looking_up(products, workers)

    def entry(self, entry: ReportEntry, first: bool) -> None:
        """The entry, kept for the record."""
        self.entries.append(entry)
        self.view.entry(entry, first)

    def engine_notice(self, lines: Sequence[str]) -> None:
        """Passed on (the record names the engine and nmap's choices instead)."""
        self.view.engine_notice(lines)

    def lookups_notice(self, online: OnlinePlan) -> None:
        """Passed on (the record has the plan from ``run_scan``)."""
        self.view.lookups_notice(online)

    def scan_failed(self, target: str, message: str) -> None:
        """The host failed, with why."""
        self._host(target).update(state="failed", message=one_line(message, 300))
        self.view.scan_failed(target, message)

    def target_done(self) -> None:
        """Passed on."""
        self.view.target_done()

    def summary(self, targets: int, entries: int, failed: Sequence[str], seconds: float,
                sent: Mapping[str, int], unavailable: Mapping[str, str]) -> None:
        """The totals, kept for the record."""
        self.totals = {"targets": targets, "open_ports": entries, "failed_lookups": list(failed),
                       "sent": dict(sent), "unavailable": dict(unavailable)}
        self.view.summary(targets, entries, failed, seconds, sent, unavailable)

    def saved_to_history(self, scan_id: str) -> None:
        """Passed on."""
        self.view.saved_to_history(scan_id)

    def not_saved_to_history(self, reason: str) -> None:
        """Passed on."""
        self.view.not_saved_to_history(reason)

    def nothing_to_export(self) -> None:
        """Passed on."""
        self.view.nothing_to_export()

    def ask_export(self) -> str | None:
        """The view's answer."""
        return self.view.ask_export()

    def unknown_format(self) -> None:
        """Passed on."""
        self.view.unknown_format()

    def exported(self, path: Path) -> None:
        """Passed on."""
        self.view.exported(path)


def scan_record(scan_id: str, request: ScanRequest, plan: OnlinePlan, notes: ScanNotes,
                started: float, seconds: float, code: int) -> dict[str, Any]:
    """A finished scan's record: ``started`` is when it began (``time.time()``), ``seconds``
    how long its scanning took, ``code`` its exit code (1: the engine couldn't scan a host)."""
    return {
        "kind": "scan", "schema": SCHEMA, "scan_id": scan_id,
        "started": datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(seconds, 1), "source": request.source,
        "target": request.target_text, "addresses": list(request.targets),
        "ports": port_ranges(request.ports), "protocols": list(request.protocols),
        "engine": request.engine,
        "nmap": asdict(request.nmap_options) if request.engine == "nmap" else None,
        "choices": {"scan_unresponsive": request.scan_unresponsive, "nvd": request.nvd,
                    "ai": request.use_ai, "abuseipdb": request.abuseipdb},  # to scan again (F6b)
        "online": {"nvd": plan.nvd, "ai": plan.ai, "abuseipdb": plan.abuseipdb},
        "hosts": [notes.hosts[target] for target in request.targets],
        "reputation": notes.reputation, "entries": notes.entries, "summary": notes.totals,
        "exit_code": code,
    }


def save_scan(scan_id: str, request: ScanRequest, plan: OnlinePlan, notes: ScanNotes,
              started: float, seconds: float, code: int) -> Path | None:
    """Saves the finished scan's record (``scan_record``) and tells the view: where, or why it
    could not (a full disk, a folder it can't write); the record's path, or None."""
    try:
        path = write_scan(scan_id, scan_record(scan_id, request, plan, notes, started, seconds,
                                               code))
    except OSError as error:  # NoSpace (HR4c) too: the scan itself is kept as it was
        log.warning("Scan %s not saved in the history: %s", scan_id, error)
        notes.not_saved_to_history(str(error))
        return None
    notes.saved_to_history(scan_id)
    return path
