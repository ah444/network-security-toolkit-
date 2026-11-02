"""What the web page sees of a running scan (v5.2 step 2.4d): ``WebScan``, the page's
``ScanView``. It asks nothing (the form answered everything), writes each step to the job's log
on one line (network text cleaned: the page shows it as text only), counts the job's progress
host by host and gathers the scan's answer. No path of this computer reaches the page.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.data.recon.nmap import how_to_install
from ...core.services.recon.engines import NmapEngine
from ...core.services.recon.scan import (
    OnlinePlan,
    OsHint,
    PortResult,
    ReportEntry,
    ScanRequest,
    one_line,
    severity_text,
)
from .jobs import Job

HOST_UNITS = 1000  # each host's share of the job's progress
REPUTATION_FIELDS = ("abuseConfidenceScore", "totalReports", "countryCode", "isp", "domain",
                     "usageType", "isTor", "lastReportedAt")  # AbuseIPDB's, as the page shows


def nmap_missing() -> str:
    """Why nmap isn't offered, without this computer's paths (the details go to the log)."""
    return f"nmap can't be used on this computer. {how_to_install()}"




class HostProgress:
    """One host's checks as its share of the job's progress (thread-safe: the scanner's worker
    threads and nmap's reader advance it)."""

    def __init__(self, view: WebScan, index: int) -> None:
        self.view, self.index = view, index
        self.total = self.done = 0
        self.lock = threading.Lock()

    def start(self, total: int) -> None:
        """The host's check count; nothing done yet."""
        with self.lock:
            self.total, self.done = total, 0

    def advance(self, done: int = 1) -> None:
        """Counts finished checks and moves the job's progress."""
        with self.lock:
            self.done += done
            share = self.done / self.total if self.total else 0
        self.view.tally(self.index * HOST_UNITS + min(HOST_UNITS, int(share * HOST_UNITS)))

    def finish(self) -> None:
        """Nothing: the next host starts its own share."""
        pass


class WebScan:
    """The ``ScanView`` of the web page: asks nothing (the form answered), writes each step to
    the job's log on one line (network text cleaned: the page shows it as text), counts the
    progress host by host, and gathers what the answer needs."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.hosts = 1
        self.index = 0  # the host being scanned, from 0
        self.step = "Starting"
        self.entries: list[ReportEntry] = []
        self.host_states: list[dict[str, str]] = []
        self.reputation: dict[str, dict[str, Any]] = {}
        self.engine_lines: list[str] = []
        self.online: OnlinePlan | None = None
        self.failed: list[str] = []
        self.totals: dict[str, Any] = {}
        self.down = ""  # the last host that didn't answer the ping

    def note(self, line: str) -> None:
        """A line in the job's log."""
        self.job.note(line)

    def tally(self, units: int) -> None:
        """The job's progress: ``units`` of all the hosts' HOST_UNITS each."""
        self.job.tally(units, self.hosts * HOST_UNITS)

    def phase(self) -> str:
        """The job's phase text while it runs."""
        if self.hosts == 1:
            return self.step
        return f"Host {min(self.index + 1, self.hosts)} of {self.hosts}: {self.step}"

    def _host(self, target: str, state: str) -> None:
        """Records how a host ended: scanned, skipped or failed."""
        self.host_states.append({"target": target, "state": state, "os": ""})

    # --- ScanView ---------------------------------------------------------------------------
    def resolved(self, text: str, address: str) -> None:
        """The host name and the address it resolved to."""
        self.note(f"{one_line(text, 200)} resolved to {address}")

    def abuse_skipped(self, target: str) -> None:
        """Never called: AbuseIPDB without a key is refused before the scan."""
        pass  # never: AbuseIPDB without a key is refused before the scan

    def ask_abuse(self, target: str) -> bool:
        """Never asked (not interactive): no."""
        return False  # never asked: not interactive

    def abuse_report(self, target: str, data: Mapping[str, Any] | None,
                     error: str | None) -> None:
        """AbuseIPDB's answer for a public address, its fields cleaned, or why it failed."""
        if data is None:
            self.reputation[target] = {"error": error or "failed"}
            self.note(f"AbuseIPDB for {target}: the lookup failed ({error})")
            return
        fields = {name: one_line(data.get(name, "N/A"), 200) for name in REPUTATION_FIELDS}
        self.reputation[target] = fields
        self.note(f"AbuseIPDB for {target}: confidence of abuse "
                  f"{fields['abuseConfidenceScore']}%, {fields['totalReports']} report(s)")

    def host_down(self, target: str) -> None:
        """The host didn't answer the ping."""
        self.down = target
        self.note(f"Host {target} did not answer the ping.")

    def ask_scan_anyway(self) -> bool:
        """Never asked (not interactive): no."""
        return False  # never asked: not interactive

    def unresponsive_skipped(self) -> None:
        """The host that didn't answer was skipped."""
        self._host(self.down, "skipped")
        self.note("Skipped (\"Scan hosts that don't answer ping\" scans it anyway).")

    def scanning(self, target: str, ttl: int | None, guess: tuple[str, int] | None) -> None:
        """The host's port checks start, with the OS guess from the ping's TTL."""
        self.step = f"checking the ports of {target}"
        self._host(target, "scanned")
        self.note(f"Scanning {target}")
        if guess:
            self.note(f"OS guess from TTL {ttl}: {guess[0]} (about {guess[1]} hops away; only "
                      "a guess)")

    def progress(self) -> HostProgress:
        """The host's share of the job's progress."""
        return HostProgress(self, self.index)

    def silent_udp(self, ports: Sequence[int]) -> None:
        """How many UDP ports sent no reply, the first 20 listed."""
        listed = ", ".join(map(str, ports[:20])) + (" ..." if len(ports) > 20 else "")
        self.note(f"UDP ports with no reply (open|filtered): {len(ports)} ({listed})")

    def no_open_ports(self, target: str) -> None:
        """The host has no open port."""
        self.note(f"No open ports on {target}.")

    def open_ports(self, results: Sequence[PortResult]) -> None:
        """The open ports and their services, on one line."""
        self.note("Open: " + ", ".join(f"{r.port}/{r.protocol} {one_line(r.service, 30)}"
                                       for r in results))

    def os_hint(self, target: str, hint: OsHint) -> None:
        """The OS guess from all the evidence, kept for the host."""
        if self.host_states:
            self.host_states[-1]["os"] = hint.summary()
        self.note(f"OS: {hint.verdict}" if hint.this_computer else
                  f"OS guess: {hint.verdict} (only a guess)")

    def looking_up(self, products: int, workers: int) -> None:
        """The products about to be looked up online."""
        self.step = f"looking up {products} product(s) online"
        self.note(f"Looking up {products} product(s) online...")

    def entry(self, entry: ReportEntry, first: bool) -> None:
        """One open port's entry, kept for the answer and the download."""
        self.entries.append(entry)
        found = f"{len(entry['cves'])} CVE(s), {severity_text(entry)}" if entry["software"] else \
            "software not recognised (no CVE lookup)"
        self.note(f"{entry['port']}/{entry['protocol']} {one_line(entry['service'], 30)}: "
                  f"{one_line(entry['software'], 60) or '-'}: {found}")

    def engine_notice(self, lines: Sequence[str]) -> None:
        """The engine and its choices, without this computer's paths."""
        self.engine_lines = [_without_paths(line) for line in lines]
        self.note("Scan engine: " + self.engine_lines[0])

    def lookups_notice(self, online: OnlinePlan) -> None:
        """What this scan may send online, kept for the answer."""
        self.online = online
        self.note(f"Online: NVD {online.nvd}; AI {online.ai}; AbuseIPDB {online.abuseipdb}")

    def scan_failed(self, target: str, message: str) -> None:
        """The engine could not scan a host; nmap's folders never shown."""
        self._host(target, "failed")
        self.failed.append(target)
        if NmapEngine().available() is not None:  # gone since the scan was checked
            message = nmap_missing()  # its own message names this computer's folders
        self.note(f"Could not scan {target}: {one_line(message, 300)}")

    def target_done(self) -> None:
        """The next host starts: its share of the progress begins."""
        self.index += 1
        self.step = "starting the next host" if self.index < self.hosts else "finishing"
        self.tally(self.index * HOST_UNITS)

    def summary(self, targets: int, entries: int, failed: Sequence[str], seconds: float,
                sent: Mapping[str, int], unavailable: Mapping[str, str]) -> None:
        """The scan's totals, kept for the answer; ``unavailable``: the services that failed
        and were not asked again, with why (HS4)."""
        self.totals = {"targets": targets, "open_ports": entries, "failed_lookups": list(failed),
                       "seconds": round(seconds, 1), "sent": dict(sent),
                       "unavailable": dict(unavailable)}
        self.note(f"Done: {targets} host(s), {entries} open port(s), {len(failed)} failed "
                  f"lookup(s), {seconds:.1f} s")

    def nothing_to_export(self) -> None:
        """Never called: nothing is saved (the download gives the files)."""
        pass  # never: no export (the download gives the files)

    def ask_export(self) -> str | None:
        """Never asked (not interactive): no file."""
        return None  # never asked: not interactive

    def unknown_format(self) -> None:
        """Never called: no format is asked."""
        pass

    def exported(self, path: Path) -> None:
        """Never called: nothing is saved."""
        pass  # never: nothing is saved

    # --- The job's result -------------------------------------------------------------------
    def answer(self, request: ScanRequest, code: int) -> dict[str, Any]:
        """A finished scan's answer: the job's result, and the page's after a reload."""
        online = self.online
        return {
            "message": f"Scanned {len(request.targets)} host(s): {len(self.entries)} open "
                       f"port(s)." + (f" Could not scan {len(self.failed)}." if code else ""),
            "target": request.target_text,
            "ports": len(request.ports),
            "protocols": list(request.protocols),
            "engine": request.engine,
            "engine_notice": self.engine_lines,
            "online": {"nvd": online.nvd, "ai": online.ai, "abuseipdb": online.abuseipdb}
            if online else None,
            "hosts": self.host_states,
            "reputation": self.reputation,
            "entries": self.entries,
            "summary": self.totals,
            "failed_targets": self.failed,
        }


def _without_paths(line: str) -> str:
    """An engine notice line without this computer's paths: nmap's version without where it
    is ("nmap 7.92 (C:\\...)"), or why it can't run without the folders it looked in."""
    if line.startswith("nmap: "):
        return f"nmap: {nmap_missing()}"
    if line.startswith("nmap ") and line.endswith(")") and " (" in line:
        return line[:line.index(" (")]
    return line
