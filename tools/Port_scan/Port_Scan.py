"""Port scanner: open TCP/UDP ports, service banners, an OS guess and known CVEs.

Only scan systems you own or have written permission to test. Every scan asks for that
confirmation first (or ``--i-am-authorized`` in scripts) and has host/port limits (S16).

    python Port_Scan.py                     interactive
    python Port_Scan.py --i-am-authorized   skip the authorization question (scripts)

What leaves this computer (step 0.6, plan D13-D15):
- CVE lookups send only the detected product and version (for example "OpenSSH 8.9") to
  the public NVD CVE API 2.0. CVE IDs, CVSS scores and severity come only from NVD (B10).
- The AI explanation (OpenRouter) is off unless you agree for this scan, and it receives
  the same product/version only, never the raw banner (S12). Its reply is shown as plain
  text labelled "may be wrong" and never adds a CVE or changes a severity (S13).
- AbuseIPDB receives a public target IP only after you agree.

API keys (all optional): environment variables OPENROUTER_API_KEY, ABUSEIPDB_API_KEY,
NVD_API_KEY (and OPENROUTER_MODEL), or the same lines in a .env file in the project
folder, or api_keys.txt next to this script (see api_keys.example.txt) (S15).
Results are saved as UTF-8 in instance/scans/, never overwriting a file (B12, S14).
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.config import scan_limits  # noqa: E402
from toolkit.logging_setup import add_secrets, setup_logging  # noqa: E402
from toolkit.recon.export import EXPORT_FORMATS, export_results, make_entry  # noqa: E402
from toolkit.recon.intel import Intel  # noqa: E402
from toolkit.recon.intel.keys import ai_model, load_api_keys  # noqa: E402
from toolkit.recon.os_detect import guess_os, ping_host  # noqa: E402
from toolkit.recon.probes import product_version  # noqa: E402
from toolkit.recon.scanner import scan_ports  # noqa: E402
from toolkit.recon.targets import (  # noqa: E402
    check_limits,
    is_ip_literal,
    is_public_ip,
    parse_port_range,
    parse_target,
)
from toolkit.recon.text import one_line  # noqa: E402

log = logging.getLogger(__name__)

# --- Authorization (S16) ----------------------------------------------------------------------

AUTHORIZATION = ("\nOnly scan systems you own or have written permission to test. Scanning "
                 "other systems can be illegal.\nType 'yes' if you are authorised to scan "
                 "{target}: ")


def confirm_authorized(target, flag=False):
    """True if the user confirms (or passed --i-am-authorized)."""
    if flag:
        return True
    return input(AUTHORIZATION.format(target=target)).strip().lower() == "yes"


# --- Progress in verbose mode ------------------------------------------------------------------

class ConsoleProgress:
    """ProgressSink for verbose mode: prints the progress line every ``interval`` seconds
    from its own thread (a slow port can't freeze it), then the final 100 % line."""

    def __init__(self, interval=2):
        self.interval = interval
        self._total = 0
        self._done = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self, total):
        self._total, self._done = total, 0
        self._stop.clear()
        if total > 0:  # nothing to show (and no division by zero) for 0 tasks
            self._write()
            self._thread = threading.Thread(target=self._repeat, daemon=True)
            self._thread.start()

    def advance(self, done=1):
        with self._lock:
            self._done += done

    def finish(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._write()
        print("\nProgress: 100.00% complete")

    def _repeat(self):
        while not self._stop.wait(self.interval):
            self._write()

    def _write(self):
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


def print_abuse_report(data):
    for field in ABUSE_FIELDS:
        value = data.get(field, "N/A")
        if isinstance(value, list):
            value = ", ".join(map(str, value)) or "N/A"
        print(f"  {field}: {one_line(value, 200)}")


# --- Printing (S14) ----------------------------------------------------------------------------

def print_table(results, title):
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


def print_entry(entry):
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


# --- Interactive scan -------------------------------------------------------------------------

def ask_yes_no(question):
    return input(question).strip().lower() in ("y", "yes")


def ask_protocols():
    while True:
        choice = input("Select protocol for scanning (TCP/UDP/both): ").strip().lower()
        if choice in ("tcp", "udp"):
            return [choice]
        if choice == "both":
            return ["tcp", "udp"]
        print("Please type TCP, UDP or both.")


def scan_target(target, ports, protocols, intel, verbose):
    """Scan one host and look up its open ports; returns the report entries."""
    if is_public_ip(target):
        if not intel.keys.get("abuseipdb"):
            print(f"{target} is a public IP; AbuseIPDB check skipped (no key).")
        elif ask_yes_no(f"Send {target} to AbuseIPDB to check its reputation? (Y/N): "):
            data, error = intel.abuse_report(target)
            print(f"\nAbuseIPDB reputation info for {target}:")
            if error:
                print(f"  Lookup failed: {error}")
            else:
                print_abuse_report(data)

    alive, ttl = ping_host(target)
    if not alive:
        print(f"Host {target} did not respond to ping.")
        if not ask_yes_no("Do you want to scan this host anyway? (Y/N): "):
            return []
    print(f"\nScanning target: {target}")
    guess = guess_os(ttl)
    if guess:
        print(f"OS guess from TTL {ttl}: {guess[0]} (about {guess[1]} hops away; "
              "TTLs can be changed, so this is only a guess)")
    else:
        print("OS guess: not possible (no TTL in the ping reply)")

    results = scan_ports(target, ports, protocols, ConsoleProgress() if verbose else None)
    open_ports = [r for r in results if r.state == "open"]
    silent_udp = [r.port for r in results if r.state == "open|filtered"]
    if silent_udp:
        listed = ", ".join(map(str, silent_udp[:20])) + (" ..." if len(silent_udp) > 20 else "")
        print(f"UDP ports with no reply (open|filtered): {len(silent_udp)} ({listed})")
    if not open_ports:
        print(f"No open ports found on {target}")
        return []

    print_table(open_ports, "OPEN PORTS")
    print("\nVulnerability details per open port:")
    entries = []
    for result in open_ports:
        software = product_version(result.banner)
        found = (intel.lookup(software) if software else
                 {"cves": [], "max_cvss": None, "severity": "UNKNOWN", "ai_explanation": "",
                  "errors": []})
        entry = make_entry(target, result, software, found)
        print_entry(entry)
        entries.append(entry)
    return entries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Port scanner. Only scan systems you are authorised to scan.")
    parser.add_argument("--i-am-authorized", action="store_true",
                        help="confirm you are authorised to scan the target (for scripts)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        limits = scan_limits()
    except ValueError as error:
        print(error)
        return 2
    keys = load_api_keys()
    add_secrets(*keys.values())  # keys from .env / api_keys.txt never reach the log
    start_time = time.time()

    target_text = input("Enter target IP (IPv4 or IPv6), network (e.g. 192.168.1.0/24) "
                        "or domain: ").strip()
    try:
        targets = parse_target(target_text, limits.max_hosts)
    except ValueError as error:
        print(error)
        return 2
    if "/" not in target_text and not is_ip_literal(target_text):
        print(f"Domain {target_text} resolved to IP: {targets[0]}")
    try:
        ports = parse_port_range(input("Enter start port: "), input("Enter end port: "))
    except ValueError as error:
        print(error)
        return 2
    protocols = ask_protocols()
    problem = check_limits(len(targets), len(targets) * len(ports) * len(protocols), limits)
    if problem:
        print(problem)
        return 2
    if not confirm_authorized(target_text, args.i_am_authorized):
        print("Scan cancelled: only scan systems you are authorised to scan.")
        return 2

    verbose = ask_yes_no("Enable verbose mode? (Y/N): ")
    use_ai = False
    if keys["openrouter"]:
        model = ai_model()
        use_ai = ask_yes_no(
            f"Send the detected software names (e.g. 'OpenSSH 8.9', never the raw banner) "
            f"to OpenRouter ({model}) for an AI explanation? (Y/N): ")
    intel = Intel(keys, use_ai)
    print("CVE lookups send only the detected software name and version to the NVD API "
          "(services.nvd.nist.gov).")

    entries = []
    for target in targets:
        entries.extend(scan_target(target, ports, protocols, intel, verbose))
        print("=" * 60)

    failed = [f"{e['port']}/{e['protocol']} {error}" for e in entries
              for error in e["lookup_errors"]]
    print("\nScan Summary Report")
    print("-------------------")
    print(f"Total targets scanned: {len(targets)}")
    print(f"Total open ports found: {len(entries)}")
    print(f"Failed lookups: {len(failed)}" + (f" ({'; '.join(failed)})" if failed else ""))
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    if not entries:
        print("No open port results to export.")
        return 0
    if ask_yes_no("Do you want to export the results to a file? (Y/N): "):
        fmt = input("Enter desired file extension (txt, json, csv): ").strip().lower()
        if fmt in EXPORT_FORMATS:
            print(f"Results exported to {export_results(entries, fmt)}")
        else:
            print("Unknown extension. No export performed.")
    return 0


def utf8_output():
    """UTF-8 output even when redirected (Windows would use cp1252 and crash on banners)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and \
                (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    utf8_output()
    setup_logging("scanner")  # instance/logs/scanner.log; warnings also on the console
    sys.exit(main())
