"""Port scanner: open TCP/UDP ports, service banners, an OS guess and known CVEs.

Only scan systems you own or have written permission to test. Every scan asks for that
confirmation first (or ``--i-am-authorized`` in scripts) and has host/port limits (S16).

    python -m toolkit scan                      interactive: asks its questions
    python -m toolkit scan 192.168.1.10 -p 1-1024 --protocol both --export csv
                                               no questions but the authorization one
    python -m toolkit scan ... --i-am-authorized   not even that one (scripts)

(from the project folder; ``toolkit scan`` once installed; moved from
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

This module is the command: its flags or questions make a ``ScanRequest``; the scan is
``toolkit.core.services.recon.scan``, what the user sees ``toolkit.ui.cli.scan`` (v5.1 step
1.5d).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence

from ...core.data.keys import KEY_VARIABLES, ai_model, legacy_key_file_notice, load_api_keys
from ...core.data.logs import add_secrets, setup_logging
from ...core.data.settings import ScanConfig, scan_limits
from ...core.services.recon.scan import (
    EXPORT_FORMATS,
    PROTOCOLS,
    ScanRequest,
    check_scan,
    resolve_targets,
    run_scan,
)
from ...core.services.recon.targets import parse_port_range, parse_ports
from ...ui.cli import scan as screen

DEFAULT_PORTS = "1-1024"  # the well-known ports
NOT_AUTHORIZED = "Scan cancelled: only scan systems you are authorised to scan."
# The flags that only make sense with a TARGET (without one, the scanner asks instead)
SCAN_FLAGS = ("ports", "protocol", "verbose", "ai", "abuseipdb", "scan_unresponsive", "export")


def ask_request(keys: Mapping[str, str], limits: ScanConfig, authorized: bool,
                view: screen.ConsoleScan) -> ScanRequest:
    """No TARGET: the questions, in the order they have always been asked."""
    target_text = screen.ask_target()
    targets = resolve_targets(target_text, limits, view)
    ports = list(parse_port_range(*screen.ask_ports()))
    protocols = screen.ask_protocols()
    check_scan(targets, ports, protocols, limits)
    if not screen.confirm_authorized(target_text, authorized):
        raise ValueError(NOT_AUTHORIZED)
    verbose = screen.ask_verbose()
    use_ai = screen.ask_ai(ai_model()) if keys["openrouter"] else False
    return ScanRequest(target_text, targets, ports, protocols, verbose, use_ai,
                       interactive=True)


def request_from_flags(args: argparse.Namespace, keys: Mapping[str, str],
                       limits: ScanConfig, view: screen.ConsoleScan) -> ScanRequest:
    """``scan TARGET [flags]``: nothing asked but the authorization (unless the flag)."""
    targets = resolve_targets(args.target, limits, view)
    ports = parse_ports(args.ports or DEFAULT_PORTS)
    protocols = PROTOCOLS[args.protocol or "tcp"]
    check_scan(targets, ports, protocols, limits)
    if not screen.confirm_authorized(args.target, args.i_am_authorized):
        raise ValueError(NOT_AUTHORIZED)
    use_ai = args.ai and bool(keys["openrouter"])
    if args.ai:
        screen.flag_notice("--ai", None if use_ai else KEY_VARIABLES["openrouter"], ai_model())
    abuseipdb = args.abuseipdb and bool(keys["abuseipdb"])
    if args.abuseipdb:
        screen.flag_notice("--abuseipdb", None if abuseipdb else KEY_VARIABLES["abuseipdb"])
    return ScanRequest(args.target, targets, ports, protocols, args.verbose, use_ai,
                       abuseipdb, args.scan_unresponsive, args.export)


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
        screen.say(error)
        return 2
    notice = legacy_key_file_notice()
    if notice:
        screen.say(notice)
    keys = load_api_keys()
    add_secrets(*keys.values())  # keys from the environment / .env never reach the log
    start_time = time.time()
    view = screen.ConsoleScan()
    try:
        try:
            request = (ask_request(keys, limits, args.i_am_authorized, view)
                       if args.target is None else request_from_flags(args, keys, limits, view))
        except ValueError as error:  # bad target or ports, a limit, no authorization
            screen.say(error)
            return 2
        return run_scan(request, keys, start_time, view)
    except (EOFError, KeyboardInterrupt):
        screen.cancelled()
        return 130


def run(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    """The command: UTF-8 output, the tool's log, then ``main``; exits with its code.
    ``prog`` is the command its help shows (``python -m toolkit scan``)."""
    screen.utf8_output()
    setup_logging("scanner")  # instance/logs/scanner.log; warnings also on the console
    sys.exit(main(argv, prog))


if __name__ == "__main__":
    run()
