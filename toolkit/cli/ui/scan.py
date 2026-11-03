"""What the port scanner shows and asks (S14, S16): the questions, the authorization,
the progress line, the tables and the report; ``ConsoleScan`` is the ``ScanView`` a scan
reports to (``toolkit.core.services.recon.scan``). Text from the network is shown on one line,
cleaned (``toolkit.core.data.recon.text``)."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.services.recon.scan import (
    PROTOCOLS,
    OnlinePlan,
    OsHint,
    PortResult,
    ReportEntry,
    match_label,
    one_line,
    severity_text,
)
from .progress import ConsoleProgress as ConsoleProgress

AUTHORIZATION = ("\nOnly scan systems you own or have written permission to test. Scanning "
                 "other systems can be illegal.\nType 'yes' if you are authorised to scan "
                 "{target}: ")


def confirm_authorized(target: str, flag: bool = False) -> bool:
    """True if the user confirms (or passed --i-am-authorized)."""
    if flag:
        return True
    return input(AUTHORIZATION.format(target=target)).strip().lower() == "yes"


ABUSE_FIELDS = (
    "ipAddress", "isPublic", "countryCode", "domain", "hostnames", "abuseConfidenceScore",
    "totalReports", "numDistinctUsers", "isTor", "isWhitelisted", "usageType", "isp",
    "lastReportedAt",
)


def print_abuse_report(data: Mapping[str, Any]) -> None:
    """The AbuseIPDB fields of ``ABUSE_FIELDS``, one per line, each on one line, cleaned and cut
    (the reply is text from the network); a missing one as N/A."""
    for field in ABUSE_FIELDS:
        value = data.get(field, "N/A")
        if isinstance(value, list):
            value = ", ".join(map(str, value)) or "N/A"
        print(f"  {field}: {one_line(value, 200)}")


def print_table(results: Sequence[PortResult], title: str) -> None:
    """The ports as a table, 79 characters wide; the service and the banner cleaned and cut to
    their column."""
    print(f"\n{title}")
    header = f"| {'PORT':^9} | {'STATE':^7} | {'SERVICE':^14} | {'BANNER':^36} |"
    separator = "+" + "-" * (len(header) - 2) + "+"
    print(separator)
    print(header)
    print(separator)
    for r in results:
        print(f"| {f'{r.port}/{r.protocol}':^9} | {r.state:^7} | {one_line(r.service, 14):^14} "
              f"| {one_line(r.banner, 36):^36} |")
    print(separator)


def print_entry(entry: ReportEntry) -> None:
    """One open port's details: the software, up to 10 CVEs (and how many more), the severity, the
    AI explanation labelled as possibly wrong, and the failed lookups."""
    print(f"\nPort: {entry['port']}/{entry['protocol']}")
    print(f"Service: {entry['service']}")
    print(f"Banner: {one_line(entry['banner'], 200)}")
    if entry["cpe"]:
        print(f"CPE (nmap): {' '.join(entry['cpe'])}")
    if not entry["software"]:
        print("Software: not recognised in the banner (no CVE lookup)" if entry["engine"] !=
              "nmap" else "Software: not recognised by nmap or in the banner (no CVE lookup)")
        return
    print(f"Software: {entry['software']}")
    cves = entry["cves"]
    shown = ", ".join(f"{c['id']} (CVSS {c['cvss'] if c['cvss'] is not None else '?'}, "
                      f"{c['severity']}, {match_label(c)})" for c in cves[:10])
    more = f" ... +{len(cves) - 10} more" if len(cves) > 10 else ""
    none = "not checked" if entry["severity"] == "NOT CHECKED" else "none found"  # HS4
    print(f"CVEs (NVD): {shown + more if cves else none}")
    if entry["cve_source"]:
        print(f"CVE source: {entry['cve_source']}")
    print(f"Severity (from CVSS): {severity_text(entry)}")
    if entry["severity_unconfirmed"]:
        print("Unconfirmed: found by the product's name only; the version was not checked")
    if entry["ai_explanation"]:
        print("AI explanation (may be wrong):")
        print(entry["ai_explanation"])
    for error in entry["lookup_errors"]:
        print(f"Lookup failed: {error}")


def ask_yes_no(question: str, default: bool = False) -> bool:
    """True for y or yes (any case); an empty answer is ``default``; any other answer is no."""
    answer = input(question).strip().lower()
    return default if not answer else answer in ("y", "yes")


def ask_protocols() -> list[str]:
    """Asks until the answer is TCP, UDP or both; the protocols to scan."""
    while True:
        choice = input("Select protocol for scanning (TCP/UDP/both): ").strip().lower()
        if choice in PROTOCOLS:
            return PROTOCOLS[choice]
        print("Please type TCP, UDP or both.")


def ask_target() -> str:
    """The target as typed; checked by ``resolve_targets``."""
    return input("Enter target IP (IPv4 or IPv6), network (e.g. 192.168.1.0/24) "
                 "or domain: ").strip()


def ask_ports() -> tuple[str, str]:
    """The start and end port as typed; checked by ``parse_port_range``."""
    return input("Enter start port: "), input("Enter end port: ")


def ask_choice(question: str, choices: Sequence[str], default: str) -> str:
    """Asks until the answer is one of ``choices`` (any case); Enter is ``default``."""
    while True:
        answer = input(f"{question} ({'/'.join(choices)}, Enter for {default}): ").strip().lower()
        if not answer:
            return default
        if answer in choices:
            return answer
        print(f"Please type {', '.join(choices[:-1])} or {choices[-1]}.")


def ask_engine(engines: Sequence[str], default: str) -> str:
    """The scan engine; asked only when nmap is installed."""
    return ask_choice("Scan engine: the toolkit's own scanner or nmap?", engines, default)


def ask_service_detection(levels: Sequence[str], default: str) -> str:
    """nmap's service detection level."""
    return ask_choice("nmap service detection: how hard to try?", levels, default)


def ask_scripts(scripts: Sequence[str]) -> bool:
    """Whether nmap also runs the toolkit's allowed scripts (``scripts``)."""
    return ask_yes_no(f"Also run the toolkit's {len(scripts)} nmap scripts "
                      f"({', '.join(scripts)})? (Y/N): ")


def ask_os_detect() -> bool:
    """Whether nmap detects the OS; asked only when it can send raw packets."""
    return ask_yes_no("Let nmap detect the OS (-O, raw packets)? (Y/N): ")


def ask_verbose() -> bool:
    """Whether to show the progress line."""
    return ask_yes_no("Enable verbose mode? (Y/N): ")


def ask_nvd() -> bool:
    """Whether this scan looks up the known CVEs in NVD (HS4: yes unless the answer is no)."""
    return ask_yes_no("Look up the known CVEs in NVD (sends the detected software name and "
                      "version, e.g. 'OpenSSH 8.9')? (Y/n): ", default=True)


def ask_ai(model: str) -> bool:
    """Whether this scan sends the detected software names to OpenRouter (``model``); asked only
    when a key is set (HS4: yes unless the answer is no)."""
    return ask_yes_no(
        f"Send the detected software names (e.g. 'OpenSSH 8.9', never the raw banner) "
        f"to OpenRouter ({model}) for an AI explanation? (Y/n): ", default=True)


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
    """Ctrl+C, or the end of input at a question (the command then exits with 130)."""
    print("\nScan cancelled.")


class ConsoleScan:
    """The ``ScanView`` of the terminal: every line a scan shows, every question it asks."""

    def resolved(self, text: str, address: str) -> None:
        """Shows the address the host name resolved to."""
        print(f"Domain {text} resolved to IP: {address}")

    def abuse_skipped(self, target: str) -> None:
        """Says the reputation check is skipped for want of a key."""
        print(f"{target} is a public IP; AbuseIPDB check skipped (no key).")

    def ask_abuse(self, target: str) -> bool:
        """Asks whether ``target`` may be sent to AbuseIPDB."""
        return ask_yes_no(f"Send {target} to AbuseIPDB to check its reputation? (Y/n): ",
                          default=True)

    def abuse_report(self, target: str, data: Mapping[str, Any] | None,
                     error: str | None) -> None:
        """Shows the reputation fields, or why the lookup failed."""
        print(f"\nAbuseIPDB reputation info for {target}:")
        if data is None:  # the lookup failed: error says why
            print(f"  Lookup failed: {error}")
        else:
            print_abuse_report(data)

    def host_down(self, target: str) -> None:
        """Says the host did not answer the ping."""
        print(f"Host {target} did not respond to ping.")

    def ask_scan_anyway(self) -> bool:
        """Asks whether to scan the host anyway."""
        return ask_yes_no("Do you want to scan this host anyway? (Y/N): ")

    def unresponsive_skipped(self) -> None:
        """Says the host was skipped, and the flag that scans it."""
        print("Skipped (--scan-unresponsive scans it anyway).")

    def scanning(self, target: str, ttl: int | None, guess: tuple[str, int] | None) -> None:
        """Shows the target and the OS guess, said to be only a guess."""
        print(f"\nScanning target: {target}")
        if guess:
            print(f"OS guess from TTL {ttl}: {guess[0]} (about {guess[1]} hops away; "
                  "TTLs can be changed, so this is only a guess)")
        else:
            print("OS guess: not possible (no TTL in the ping reply)")

    def progress(self) -> ConsoleProgress:
        """A ``ConsoleProgress`` for the target's checks."""
        return ConsoleProgress()

    def silent_udp(self, ports: Sequence[int]) -> None:
        """Counts the silent UDP ports and lists the first 20."""
        listed = ", ".join(map(str, ports[:20])) + (" ..." if len(ports) > 20 else "")
        print(f"UDP ports with no reply (open|filtered): {len(ports)} ({listed})")

    def no_open_ports(self, target: str) -> None:
        """Says the target has no open port."""
        print(f"No open ports found on {target}")

    def open_ports(self, results: Sequence[PortResult]) -> None:
        """Shows the open ports as a table."""
        print_table(results, "OPEN PORTS")

    def os_hint(self, target: str, hint: OsHint) -> None:
        """Shows the OS verdict (or that the target is this computer), each piece of evidence
        and why some of it may disagree; a guess is said to be one."""
        if hint.this_computer:
            print(f"\nOS: {hint.verdict}, as {target} is one of its own addresses")
        else:
            print(f"\nOS guess from all the evidence: {hint.verdict}")
        for evidence in hint.evidence:
            print(f"  - {evidence.says}: {evidence.seen}")
        if hint.note:
            print(f"  {hint.note}")
        if not hint.this_computer:
            print("  (only a guess: TTLs can be set and banners can lie)")

    def looking_up(self, products: int, workers: int) -> None:
        """Says how many products will be looked up online."""
        if products > 1:
            print(f"\nLooking up {products} products online (up to {workers} at a "
                  "time)...")
        else:
            print("\nLooking up 1 product online...")

    def entry(self, entry: ReportEntry, first: bool) -> None:
        """Shows one port's details, under a heading before the first."""
        if first:
            print("\nVulnerability details per open port:")
        print_entry(entry)

    def engine_notice(self, lines: Sequence[str]) -> None:
        """Shows the scan engine and its choices, one per line."""
        print("Scan engine:")
        for line in lines:
            print(f"  {line}")

    def scan_failed(self, target: str, message: str) -> None:
        """Says the engine could not scan ``target`` and why."""
        print(f"Could not scan {target}: {message}")

    def lookups_notice(self, online: OnlinePlan) -> None:
        """Says what this scan may send to each online service, or why it sends nothing."""
        print("Online in this scan:")
        print(f"  NVD (CVEs, services.nvd.nist.gov): {online.nvd}")
        print(f"  AI explanation (OpenRouter): {online.ai}")
        print(f"  IP reputation (AbuseIPDB): {online.abuseipdb}")

    def target_done(self) -> None:
        """A line under the target."""
        print("=" * 60)

    def summary(self, targets: int, entries: int, failed: Sequence[str], seconds: float,
                sent: Mapping[str, int], unavailable: Mapping[str, str]) -> None:
        """Shows the summary report, with the requests sent online and the services that failed
        and were not asked again (HS4)."""
        print("\nScan Summary Report")
        print("-------------------")
        print(f"Total targets scanned: {targets}")
        print(f"Total open ports found: {entries}")
        print(f"Failed lookups: {len(failed)}" + (f" ({'; '.join(failed)})" if failed else ""))
        print("Requests sent online: " + ", ".join(f"{name} {n}" for name, n in sent.items()))
        if unavailable:
            print("Unavailable: " + "; ".join(f"{name} ({why})"
                                              for name, why in unavailable.items()))
        print(f"Elapsed time: {seconds:.2f} seconds")

    def nothing_to_export(self) -> None:
        """Says there is nothing to export."""
        print("No open port results to export.")

    def ask_export(self) -> str | None:
        """The format the user wants, or None for no export."""
        if not ask_yes_no("Do you want to export the results to a file? (Y/N): "):
            return None
        return input("Enter desired file extension (txt, json, csv): ").strip().lower()

    def unknown_format(self) -> None:
        """Says the extension is unknown and nothing was saved."""
        print("Unknown extension. No export performed.")

    def exported(self, path: Path) -> None:
        """Shows where the report was saved."""
        print(f"Results exported to {path}")


def utf8_output() -> None:
    """UTF-8 output even when redirected (Windows would use cp1252 and crash on banners)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and \
                (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
