"""Port scanner: open TCP/UDP ports, service banners, an OS guess and known CVEs.

Only scan systems you own or have written permission to test. Every scan asks for that
confirmation first (or ``--i-am-authorized`` in scripts) and has host/port limits (S16).

    python -m toolkit scan                      interactive: asks its questions
    python -m toolkit scan 192.168.1.10 -p 1-1024 --protocol both --export csv
                                               no questions but the authorization one
    python -m toolkit scan ... --i-am-authorized   not even that one (scripts)

(from the project folder; ``python -m toolkit.recon.cli`` also works; moved from
tools/Port_scan/Port_Scan.py in v5.1 step 1.4a, flags in step 1.4c). With a TARGET nothing
is sent online but the CVE lookups unless a flag says so (``--ai``, ``--abuseipdb``).
Ctrl+C (or the end of input at a question) cancels: "Scan cancelled.", exit code 130.

What leaves this computer (step 0.6, plan D13-D15):
- CVE lookups send only the detected product and version (for example "OpenSSH 8.9") to
  the public NVD CVE API 2.0. CVE IDs, CVSS scores and severity come only from NVD (B10).
- The AI explanation (OpenRouter) is off unless you agree for this scan, and it receives
  the same product/version only, never the raw banner (S12). Its reply is shown as plain
  text labelled "may be wrong" and never adds a CVE or changes a severity (S13).
- AbuseIPDB receives a public target IP only after you agree.

API keys (all optional): environment variables OPENROUTER_API_KEY, ABUSEIPDB_API_KEY,
NVD_API_KEY (and OPENROUTER_MODEL), or the same lines in a .env file in the project
folder (S15). Results are saved as UTF-8 in instance/scans/, never overwriting a file
(B12, S14).

The scanning logic is in ``toolkit.recon`` (targets, ping and OS guess, probes registry,
scanner, online lookups, export); this module only talks to the user (v5.1 step 1.1d).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..data.settings import ScanConfig, scan_limits
from ..data.logs import add_secrets, setup_logging
from ..data.recon.scan_files import EXPORT_FORMATS, ReportEntry, export_results
from ..services.recon.report import make_entry
from ..services.recon.intel import Intel, LookupResult
from ..data.keys import KEY_VARIABLES, ai_model, legacy_key_file_notice, load_api_keys
from ..data.recon.lookup import INTEL_WORKERS
from ..services.recon.os_guess import guess_os
from ..data.recon.ping import ping_host
from ..data.recon.probes import product_version
from ..data.recon.scanner import PortResult, scan_ports
from ..services.recon.targets import (
    check_limits,
    is_ip_literal,
    is_public_ip,
    parse_port_range,
    parse_ports,
    parse_target,
)
from ..data.recon.text import one_line

# --- Authorization (S16) ----------------------------------------------------------------------

AUTHORIZATION = ("\nOnly scan systems you own or have written permission to test. Scanning "
                 "other systems can be illegal.\nType 'yes' if you are authorised to scan "
                 "{target}: ")


def confirm_authorized(target: str, flag: bool = False) -> bool:
    """True if the user confirms (or passed --i-am-authorized)."""
    if flag:
        return True
    return input(AUTHORIZATION.format(target=target)).strip().lower() == "yes"


# --- Progress in verbose mode ------------------------------------------------------------------

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


# --- Printing (S14) ----------------------------------------------------------------------------

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


# --- What to scan ------------------------------------------------------------------------------

DEFAULT_PORTS = "1-1024"  # the well-known ports
PROTOCOLS = {"tcp": ["tcp"], "udp": ["udp"], "both": ["tcp", "udp"]}
NOT_AUTHORIZED = "Scan cancelled: only scan systems you are authorised to scan."
# The flags that only make sense with a TARGET (without one, the scanner asks instead)
SCAN_FLAGS = ("ports", "protocol", "verbose", "ai", "abuseipdb", "scan_unresponsive", "export")


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


def ask_yes_no(question: str) -> bool:
    return input(question).strip().lower() in ("y", "yes")


def ask_protocols() -> list[str]:
    while True:
        choice = input("Select protocol for scanning (TCP/UDP/both): ").strip().lower()
        if choice in PROTOCOLS:
            return PROTOCOLS[choice]
        print("Please type TCP, UDP or both.")


def resolve_targets(text: str, limits: ScanConfig) -> list[str]:
    targets = parse_target(text, limits.max_hosts)
    if "/" not in text and not is_ip_literal(text):
        print(f"Domain {text} resolved to IP: {targets[0]}")
    return targets


def check_scan(targets: Sequence[str], ports: Sequence[int], protocols: Sequence[str],
               limits: ScanConfig) -> None:
    problem = check_limits(len(targets), len(targets) * len(ports) * len(protocols), limits)
    if problem:
        raise ValueError(problem)


def ask_request(keys: Mapping[str, str], limits: ScanConfig, authorized: bool) -> ScanRequest:
    """No TARGET: the questions, in the order they have always been asked."""
    target_text = input("Enter target IP (IPv4 or IPv6), network (e.g. 192.168.1.0/24) "
                        "or domain: ").strip()
    targets = resolve_targets(target_text, limits)
    ports = list(parse_port_range(input("Enter start port: "), input("Enter end port: ")))
    protocols = ask_protocols()
    check_scan(targets, ports, protocols, limits)
    if not confirm_authorized(target_text, authorized):
        raise ValueError(NOT_AUTHORIZED)
    verbose = ask_yes_no("Enable verbose mode? (Y/N): ")
    use_ai = False
    if keys["openrouter"]:
        use_ai = ask_yes_no(
            f"Send the detected software names (e.g. 'OpenSSH 8.9', never the raw banner) "
            f"to OpenRouter ({ai_model()}) for an AI explanation? (Y/N): ")
    return ScanRequest(target_text, targets, ports, protocols, verbose, use_ai,
                       interactive=True)


def request_from_flags(args: argparse.Namespace, keys: Mapping[str, str],
                       limits: ScanConfig) -> ScanRequest:
    """``scan TARGET [flags]``: nothing asked but the authorization (unless the flag)."""
    targets = resolve_targets(args.target, limits)
    ports = parse_ports(args.ports or DEFAULT_PORTS)
    protocols = PROTOCOLS[args.protocol or "tcp"]
    check_scan(targets, ports, protocols, limits)
    if not confirm_authorized(args.target, args.i_am_authorized):
        raise ValueError(NOT_AUTHORIZED)
    use_ai = args.ai and bool(keys["openrouter"])
    if args.ai and not use_ai:
        print(f"--ai: no {KEY_VARIABLES['openrouter']} set, so no AI explanation.")
    elif use_ai:
        print(f"--ai: the detected software names (e.g. 'OpenSSH 8.9', never the raw banner) "
              f"go to OpenRouter ({ai_model()}) for an AI explanation.")
    abuseipdb = args.abuseipdb and bool(keys["abuseipdb"])
    if args.abuseipdb and not abuseipdb:
        print(f"--abuseipdb: no {KEY_VARIABLES['abuseipdb']} set, so no reputation check.")
    elif abuseipdb:
        print("--abuseipdb: public target IPs go to AbuseIPDB for a reputation check.")
    return ScanRequest(args.target, targets, ports, protocols, args.verbose, use_ai,
                       abuseipdb, args.scan_unresponsive, args.export)


# --- The scan ----------------------------------------------------------------------------------

def scan_target(target: str, request: ScanRequest, intel: Intel) -> list[ReportEntry]:
    """Scan one host and look up its open ports; returns the report entries."""
    if is_public_ip(target):
        has_key = bool(intel.keys.get("abuseipdb"))
        if request.interactive and not has_key:
            print(f"{target} is a public IP; AbuseIPDB check skipped (no key).")
        elif has_key and (ask_yes_no(f"Send {target} to AbuseIPDB to check its reputation? "
                                     "(Y/N): ") if request.interactive else request.abuseipdb):
            data, error = intel.abuse_report(target)
            print(f"\nAbuseIPDB reputation info for {target}:")
            if data is None:  # the lookup failed: error says why
                print(f"  Lookup failed: {error}")
            else:
                print_abuse_report(data)

    alive, ttl = ping_host(target)
    if not alive:
        print(f"Host {target} did not respond to ping.")
        if request.interactive:
            if not ask_yes_no("Do you want to scan this host anyway? (Y/N): "):
                return []
        elif not request.scan_unresponsive:
            print("Skipped (--scan-unresponsive scans it anyway).")
            return []
    print(f"\nScanning target: {target}")
    guess = guess_os(ttl)
    if guess:
        print(f"OS guess from TTL {ttl}: {guess[0]} (about {guess[1]} hops away; "
              "TTLs can be changed, so this is only a guess)")
    else:
        print("OS guess: not possible (no TTL in the ping reply)")

    results = scan_ports(target, request.ports, request.protocols,
                         ConsoleProgress() if request.verbose else None)
    open_ports = [r for r in results if r.state == "open"]
    silent_udp = [r.port for r in results if r.state == "open|filtered"]
    if silent_udp:
        listed = ", ".join(map(str, silent_udp[:20])) + (" ..." if len(silent_udp) > 20 else "")
        print(f"UDP ports with no reply (open|filtered): {len(silent_udp)} ({listed})")
    if not open_ports:
        print(f"No open ports found on {target}")
        return []

    print_table(open_ports, "OPEN PORTS")
    products = [product_version(result.banner) for result in open_ports]
    distinct = [software for software in dict.fromkeys(products) if software]
    if len(distinct) > 1:
        print(f"\nLooking up {len(distinct)} products online (up to {INTEL_WORKERS} at a "
              "time)...")
    elif distinct:
        print("\nLooking up 1 product online...")
    looked_up = intel.lookup_many(distinct)  # each product once, in parallel (step 1.4e)
    print("\nVulnerability details per open port:")
    entries: list[ReportEntry] = []
    for result, software in zip(open_ports, products, strict=True):
        found: LookupResult = (
            looked_up[software] if software else
            {"cves": [], "max_cvss": None, "severity": "UNKNOWN", "ai_explanation": "",
             "errors": []})
        entry = make_entry(target, result, software, found)
        print_entry(entry)
        entries.append(entry)
    return entries


def run_scan(request: ScanRequest, keys: Mapping[str, str], start_time: float) -> int:
    """Scan every target, print the summary, export (asked, or ``--export``)."""
    intel = Intel(keys, request.use_ai)
    print("CVE lookups send only the detected software name and version to the NVD API "
          "(services.nvd.nist.gov).")

    entries: list[ReportEntry] = []
    for target in request.targets:
        entries.extend(scan_target(target, request, intel))
        print("=" * 60)

    failed = [f"{e['port']}/{e['protocol']} {error}" for e in entries
              for error in e["lookup_errors"]]
    print("\nScan Summary Report")
    print("-------------------")
    print(f"Total targets scanned: {len(request.targets)}")
    print(f"Total open ports found: {len(entries)}")
    print(f"Failed lookups: {len(failed)}" + (f" ({'; '.join(failed)})" if failed else ""))
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    if not entries:
        if request.interactive or request.export:
            print("No open port results to export.")
        return 0
    fmt = request.export
    if request.interactive and ask_yes_no("Do you want to export the results to a file? "
                                          "(Y/N): "):
        fmt = input("Enter desired file extension (txt, json, csv): ").strip().lower()
        if fmt not in EXPORT_FORMATS:
            print("Unknown extension. No export performed.")
            return 0
    if fmt:
        print(f"Results exported to {export_results(entries, fmt)}")
    return 0


# --- The command -------------------------------------------------------------------------------

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Port scanner. Only scan systems you are authorised to scan.\n\n"
                    "Without a TARGET it asks its questions. With one it asks nothing but the\n"
                    "authorization question, and sends nothing online but the CVE lookups\n"
                    "unless --ai or --abuseipdb says so.",
        epilog="example:\n  %(prog)s 192.168.1.10 -p 1-1024 --protocol both --export csv")
    parser.add_argument("target", nargs="?", metavar="TARGET",
                        help="IPv4 or IPv6 address, network (e.g. 192.168.1.0/24) or domain")
    parser.add_argument("-p", "--ports", metavar="SPEC",
                        help=f"80, 1-1024 or a list such as 22,80,8000-8100 "
                             f"(default {DEFAULT_PORTS})")
    parser.add_argument("--protocol", type=str.lower, choices=list(PROTOCOLS),
                        help="default tcp")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the progress while scanning")
    parser.add_argument("--ai", action="store_true",
                        help="send the detected software names (never the raw banner) to "
                             f"OpenRouter for an AI explanation (needs "
                             f"{KEY_VARIABLES['openrouter']})")
    parser.add_argument("--abuseipdb", action="store_true",
                        help="send public target IPs to AbuseIPDB for a reputation check "
                             f"(needs {KEY_VARIABLES['abuseipdb']})")
    parser.add_argument("--scan-unresponsive", action="store_true",
                        help="also scan hosts that don't answer ping (default: skip them)")
    parser.add_argument("--export", type=str.lower, choices=list(EXPORT_FORMATS),
                        help="save the report in instance/scans/ (never overwrites a file)")
    parser.add_argument("--i-am-authorized", action="store_true",
                        help="confirm you are authorised to scan the target (for scripts)")
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    used = ["--" + name.replace("_", "-") for name in SCAN_FLAGS if getattr(args, name)]
    if args.target is None and used:
        parser.error(f"{', '.join(used)}: only with a TARGET (without one the scanner asks "
                     "its questions)")
    try:
        limits = scan_limits()
    except ValueError as error:
        print(error)
        return 2
    notice = legacy_key_file_notice()
    if notice:
        print(notice)
    keys = load_api_keys()
    add_secrets(*keys.values())  # keys from the environment / .env never reach the log
    start_time = time.time()
    try:
        try:
            request = (ask_request(keys, limits, args.i_am_authorized) if args.target is None
                       else request_from_flags(args, keys, limits))
        except ValueError as error:  # bad target or ports, a limit, no authorization
            print(error)
            return 2
        return run_scan(request, keys, start_time)
    except (EOFError, KeyboardInterrupt):
        print("\nScan cancelled.")
        return 130


def utf8_output() -> None:
    """UTF-8 output even when redirected (Windows would use cp1252 and crash on banners)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and \
                (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")


def run(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    """The command: UTF-8 output, the tool's log, then ``main``; exits with its code.
    ``prog`` is the command its help shows (``python -m toolkit scan``)."""
    utf8_output()
    setup_logging("scanner")  # instance/logs/scanner.log; warnings also on the console
    sys.exit(main(argv, prog))


if __name__ == "__main__":
    run()
