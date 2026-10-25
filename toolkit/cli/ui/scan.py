"""What the port scanner shows and asks (S14, S16): the questions, the authorization,
the progress line, the tables and the report; ``ConsoleScan`` is the ``ScanView`` a scan
reports to (``toolkit.core.services.recon.scan``). Text from the network is shown on one line,
cleaned (``toolkit.core.data.recon.text``)."""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.services.recon.scan import PROTOCOLS, PortResult, ReportEntry, one_line

AUTHORIZATION = ("\nOnly scan systems you own or have written permission to test. Scanning "
                 "other systems can be illegal.\nType 'yes' if you are authorised to scan "
                 "{target}: ")


def confirm_authorized(target: str, flag: bool = False) -> bool:
    """True if the user confirms (or passed --i-am-authorized)."""
    if flag:
        return True
    return input(AUTHORIZATION.format(target=target)).strip().lower() == "yes"


class ConsoleProgress:
    """ProgressSink for verbose mode: prints the progress line every ``interval`` seconds
    from its own thread (a slow port can't freeze it), then the final 100 % line."""

    def __init__(self, interval: float = 2) -> None:
        self.interval = interval
        self._total = 0
        self._done = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, total: int) -> None:
        self._total, self._done = total, 0
        self._stop.clear()
        if total > 0:  # nothing to show (and no division by zero) for 0 tasks
            self._write()
            self._thread = threading.Thread(target=self._repeat, daemon=True)
            self._thread.start()

    def advance(self, done: int = 1) -> None:
        with self._lock:
            self._done += done

    def finish(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._write()
        print("\nProgress: 100.00% complete" if self._done >= self._total else "")  # else cancelled

    def _repeat(self) -> None:
        while not self._stop.wait(self.interval):
            self._write()

    def _write(self) -> None:
        with self._lock:
            done = self._done
        sys.stdout.write(f"\rProgress: {done / self._total * 100:.2f}% complete | "
                         f"{self._total - done} tasks remaining")
        sys.stdout.flush()


ABUSE_FIELDS = (
    "ipAddress", "isPublic", "countryCode", "domain", "hostnames", "abuseConfidenceScore",
    "totalReports", "numDistinctUsers", "isTor", "isWhitelisted", "usageType", "isp",
    "lastReportedAt",
)


def print_abuse_report(data: Mapping[str, Any]) -> None:
    for field in ABUSE_FIELDS:
        value = data.get(field, "N/A")
        if isinstance(value, list):
            value = ", ".join(map(str, value)) or "N/A"
        print(f"  {field}: {one_line(value, 200)}")


def print_table(results: Sequence[PortResult], title: str) -> None:
    print(f"\n{title}")
    header = f"| {'PORT':^9} | {'STATE':^7} | {'SERVICE':^10} | {'BANNER':^40} |"
    separator = "+" + "-" * (len(header) - 2) + "+"
    print(separator)
    print(header)
    print(separator)
    for r in results:
        print(f"| {f'{r.port}/{r.protocol}':^9} | {r.state:^7} | {one_line(r.service, 10):^10} "
              f"| {one_line(r.banner, 40):^40} |")
    print(separator)


def print_entry(entry: ReportEntry) -> None:
    print(f"\nPort: {entry['port']}/{entry['protocol']}")
    print(f"Service: {entry['service']}")
    print(f"Banner: {one_line(entry['banner'], 200)}")
    if not entry["software"]:
        print("Software: not recognised in the banner (no CVE lookup)")
        return
    print(f"Software: {entry['software']}")
    cves = entry["cves"]
    shown = ", ".join(f"{c['id']} (CVSS {c['cvss'] if c['cvss'] is not None else '?'}, "
                      f"{c['severity']})" for c in cves[:10])
    more = f" ... +{len(cves) - 10} more" if len(cves) > 10 else ""
    print(f"CVEs (NVD keyword search): {shown + more if cves else 'none found'}")
    print(f"Severity (from CVSS): {entry['severity']}")
    if entry["ai_explanation"]:
        print("AI explanation (may be wrong):")
        print(entry["ai_explanation"])
    for error in entry["lookup_errors"]:
        print(f"Lookup failed: {error}")


def ask_yes_no(question: str) -> bool:
    return input(question).strip().lower() in ("y", "yes")


def ask_protocols() -> list[str]:
    while True:
        choice = input("Select protocol for scanning (TCP/UDP/both): ").strip().lower()
        if choice in PROTOCOLS:
            return PROTOCOLS[choice]
        print("Please type TCP, UDP or both.")


def ask_target() -> str:
    return input("Enter target IP (IPv4 or IPv6), network (e.g. 192.168.1.0/24) "
                 "or domain: ").strip()


def ask_ports() -> tuple[str, str]:
    return input("Enter start port: "), input("Enter end port: ")


def ask_verbose() -> bool:
    return ask_yes_no("Enable verbose mode? (Y/N): ")


def ask_ai(model: str) -> bool:
    return ask_yes_no(
        f"Send the detected software names (e.g. 'OpenSSH 8.9', never the raw banner) "
        f"to OpenRouter ({model}) for an AI explanation? (Y/N): ")


def flag_notice(flag: str, variable: str | None = None, model: str = "") -> None:
    """What ``--ai`` / ``--abuseipdb`` will do: nothing without its key (``variable``)."""
    if flag == "--ai":
        print(f"--ai: no {variable} set, so no AI explanation." if variable else
              f"--ai: the detected software names (e.g. 'OpenSSH 8.9', never the raw banner) "
              f"go to OpenRouter ({model}) for an AI explanation.")
    else:
        print(f"--abuseipdb: no {variable} set, so no reputation check." if variable else
              "--abuseipdb: public target IPs go to AbuseIPDB for a reputation check.")


def say(message: object) -> None:
    """A message, a refusal or a notice, as it is."""
    print(message)


def cancelled() -> None:
    print("\nScan cancelled.")


class ConsoleScan:
    """The ``ScanView`` of the terminal: every line a scan shows, every question it asks."""

    def resolved(self, text: str, address: str) -> None:
        print(f"Domain {text} resolved to IP: {address}")

    def abuse_skipped(self, target: str) -> None:
        print(f"{target} is a public IP; AbuseIPDB check skipped (no key).")

    def ask_abuse(self, target: str) -> bool:
        return ask_yes_no(f"Send {target} to AbuseIPDB to check its reputation? (Y/N): ")

    def abuse_report(self, target: str, data: Mapping[str, Any] | None,
                     error: str | None) -> None:
        print(f"\nAbuseIPDB reputation info for {target}:")
        if data is None:  # the lookup failed: error says why
            print(f"  Lookup failed: {error}")
        else:
            print_abuse_report(data)

    def host_down(self, target: str) -> None:
        print(f"Host {target} did not respond to ping.")

    def ask_scan_anyway(self) -> bool:
        return ask_yes_no("Do you want to scan this host anyway? (Y/N): ")

    def unresponsive_skipped(self) -> None:
        print("Skipped (--scan-unresponsive scans it anyway).")

    def scanning(self, target: str, ttl: int | None, guess: tuple[str, int] | None) -> None:
        print(f"\nScanning target: {target}")
        if guess:
            print(f"OS guess from TTL {ttl}: {guess[0]} (about {guess[1]} hops away; "
                  "TTLs can be changed, so this is only a guess)")
        else:
            print("OS guess: not possible (no TTL in the ping reply)")

    def progress(self) -> ConsoleProgress:
        return ConsoleProgress()

    def silent_udp(self, ports: Sequence[int]) -> None:
        listed = ", ".join(map(str, ports[:20])) + (" ..." if len(ports) > 20 else "")
        print(f"UDP ports with no reply (open|filtered): {len(ports)} ({listed})")

    def no_open_ports(self, target: str) -> None:
        print(f"No open ports found on {target}")

    def open_ports(self, results: Sequence[PortResult]) -> None:
        print_table(results, "OPEN PORTS")

    def looking_up(self, products: int, workers: int) -> None:
        if products > 1:
            print(f"\nLooking up {products} products online (up to {workers} at a "
                  "time)...")
        else:
            print("\nLooking up 1 product online...")

    def entry(self, entry: ReportEntry, first: bool) -> None:
        if first:
            print("\nVulnerability details per open port:")
        print_entry(entry)

    def lookups_notice(self) -> None:
        print("CVE lookups send only the detected software name and version to the NVD API "
              "(services.nvd.nist.gov).")

    def target_done(self) -> None:
        print("=" * 60)

    def summary(self, targets: int, entries: int, failed: Sequence[str],
                seconds: float) -> None:
        print("\nScan Summary Report")
        print("-------------------")
        print(f"Total targets scanned: {targets}")
        print(f"Total open ports found: {entries}")
        print(f"Failed lookups: {len(failed)}" + (f" ({'; '.join(failed)})" if failed else ""))
        print(f"Elapsed time: {seconds:.2f} seconds")

    def nothing_to_export(self) -> None:
        print("No open port results to export.")

    def ask_export(self) -> str | None:
        """The format the user wants, or None for no export."""
        if not ask_yes_no("Do you want to export the results to a file? (Y/N): "):
            return None
        return input("Enter desired file extension (txt, json, csv): ").strip().lower()

    def unknown_format(self) -> None:
        print("Unknown extension. No export performed.")

    def exported(self, path: Path) -> None:
        print(f"Results exported to {path}")


def utf8_output() -> None:
    """UTF-8 output even when redirected (Windows would use cp1252 and crash on banners)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and \
                (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
