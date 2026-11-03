"""Port scanner: open TCP/UDP ports, service banners, an OS guess and known CVEs.

Only scan systems you own or have written permission to test. Every scan asks for that
confirmation first (or ``--i-am-authorized`` in scripts) and has host/port limits (S16).

    python -m toolkit scan                      interactive: asks its questions
    python -m toolkit scan 192.168.1.10 -p 1-1024 --protocol both --export csv
                                               no questions but the authorization one
    python -m toolkit scan ... --i-am-authorized   not even that one (scripts)
    python -m toolkit scan --history            the saved scans (F6d; also --show,
                                               --compare and --delete): nothing is scanned

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
(B12, S14). Every finished scan is also saved in the scan history, instance/scan_history/
(F6a).

This module is the command: its flags or questions make a ``ScanRequest``; the scan is
``toolkit.core.services.recon.scan``, what the user sees ``toolkit.cli.ui.scan`` (v5.1 step
1.5d).
"""

from __future__ import annotations

import argparse
import errno
import sys
import time
from collections.abc import Mapping, Sequence

from ...core.data.keys import KEY_VARIABLES, ai_model, legacy_key_file_notice, load_api_keys
from ...core.data.logs import add_secrets, setup_logging
from ...core.data.settings import ScanConfig, scan_limits
from ...core.services.recon.engines import (
    DEFAULT_ENGINE,
    ENGINES,
    INTENSITIES,
    SCRIPT_SETS,
    SCRIPTS,
    TIMINGS,
    NmapEngine,
    NmapOptions,
    engine_problem,
    has_privileges,
)
from ...core.services.recon.scan import (
    EXPORT_FORMATS,
    PROTOCOLS,
    ScanRequest,
    check_scan,
    resolve_targets,
    run_scan,
)
from ...core.services.recon.targets import DEFAULT_PORTS, parse_port_range, parse_ports
from ..ui import scan as screen
from .scan_history import HISTORY_FLAGS, history

NOT_AUTHORIZED = "Scan cancelled: only scan systems you are authorised to scan."
# The flags that only make sense with a TARGET (without one, the scanner asks instead)
NMAP_FLAGS = ("service_detection", "timing", "scripts", "os_detect")  # only with --engine nmap
SCAN_FLAGS = ("ports", "protocol", "verbose", "no_nvd", "no_ai", "ai", "no_abuseipdb", "abuseipdb",
              "scan_unresponsive", "export", "engine", *NMAP_FLAGS)


def check_engine(engine: str, options: NmapOptions, protocols: list[str]) -> None:
    """ValueError with the reason when the engine can't do this scan here (nmap not installed,
    or UDP or OS detection without raw packets): asked before the authorization, and never
    replaced by the built-in engine without the user choosing it."""
    problem = engine_problem(engine, options, protocols)
    if problem:
        raise ValueError(f"The {engine} engine can't run this scan: {problem}")


def ask_engine() -> tuple[str, NmapOptions]:
    """No TARGET: the engine, asked only when nmap is installed, and nmap's choices (OS
    detection only when it can send raw packets; the timing stays normal)."""
    if NmapEngine().available() is not None or \
            screen.ask_engine(list(ENGINES), DEFAULT_ENGINE) != "nmap":
        return DEFAULT_ENGINE, NmapOptions()
    defaults = NmapOptions()
    level = screen.ask_service_detection(list(INTENSITIES), defaults.service_detection)
    scripts = "safe" if screen.ask_scripts(SCRIPTS) else ""
    os_detect = screen.ask_os_detect() if has_privileges() else False
    return "nmap", NmapOptions(level, defaults.timing, scripts, os_detect)


def ask_request(keys: Mapping[str, str], limits: ScanConfig, authorized: bool,
                view: screen.ConsoleScan) -> ScanRequest:
    """No TARGET: the questions, in the order they have always been asked (the engine's after
    the protocol, step N4)."""
    target_text = screen.ask_target()
    targets = resolve_targets(target_text, limits, view)
    ports = list(parse_port_range(*screen.ask_ports()))
    protocols = screen.ask_protocols()
    engine, options = ask_engine()
    check_scan(targets, ports, protocols, limits)
    check_engine(engine, options, protocols)
    if not screen.confirm_authorized(target_text, authorized):
        raise ValueError(NOT_AUTHORIZED)
    verbose = screen.ask_verbose()
    nvd = screen.ask_nvd()
    use_ai = screen.ask_ai(ai_model()) if keys["openrouter"] else False
    return ScanRequest(target_text, targets, ports, protocols, verbose, use_ai,
                       interactive=True, engine=engine, nmap_options=options, nvd=nvd)


def request_from_flags(args: argparse.Namespace, keys: Mapping[str, str],
                       limits: ScanConfig, view: screen.ConsoleScan) -> ScanRequest:
    """``scan TARGET [flags]``: nothing asked but the authorization (unless the flag)."""
    targets = resolve_targets(args.target, limits, view)
    ports = parse_ports(args.ports or DEFAULT_PORTS)
    protocols = PROTOCOLS[args.protocol or "tcp"]
    engine = args.engine or DEFAULT_ENGINE
    defaults = NmapOptions()
    options = NmapOptions(args.service_detection or defaults.service_detection,
                          args.timing or defaults.timing, args.scripts or "", args.os_detect)
    check_scan(targets, ports, protocols, limits)
    check_engine(engine, options, protocols)
    if not screen.confirm_authorized(args.target, args.i_am_authorized):
        raise ValueError(NOT_AUTHORIZED)
    # HS4: each service is on when its key is set, unless its --no- flag turns it off
    use_ai = not args.no_ai and bool(keys["openrouter"])
    if args.ai:
        screen.flag_notice("--ai", None if use_ai else KEY_VARIABLES["openrouter"], ai_model())
    abuseipdb = not args.no_abuseipdb and bool(keys["abuseipdb"])
    if args.abuseipdb:
        screen.flag_notice("--abuseipdb", None if abuseipdb else KEY_VARIABLES["abuseipdb"])
    return ScanRequest(args.target, targets, ports, protocols, args.verbose, use_ai,
                       abuseipdb, args.scan_unresponsive, args.export, engine=engine,
                       nmap_options=options, nvd=not args.no_nvd)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """The ``scan`` command's flags; ``prog`` is the command its help shows."""
    parser = argparse.ArgumentParser(
        prog=prog, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Port scanner. Only scan systems you are authorised to scan.\n\n"
                    "Without a TARGET it asks its questions. With one it asks nothing but the\n"
                    "authorization question. NVD, the AI and AbuseIPDB are used when their key\n"
                    "is set (NVD needs none) unless --no-nvd, --no-ai or --no-abuseipdb turns\n"
                    "them off.",
        epilog="examples:\n  %(prog)s 192.168.1.10 -p 1-1024 --protocol both --export csv\n"
               "  %(prog)s --history")
    parser.add_argument("target", nargs="?", metavar="TARGET",
                        help="IPv4 or IPv6 address, network (e.g. 192.168.1.0/24) or domain")
    parser.add_argument("-p", "--ports", metavar="SPEC",
                        help=f"80, 1-1024 or a list such as 22,80,8000-8100 "
                             f"(default {DEFAULT_PORTS})")
    parser.add_argument("--protocol", type=str.lower, choices=list(PROTOCOLS),
                        help="default tcp")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the progress while scanning")
    parser.add_argument("--no-nvd", action="store_true",
                        help="don't look up the known CVEs in NVD (on by default: it gets the "
                             "detected software name and version)")
    ai = parser.add_mutually_exclusive_group()
    ai.add_argument("--no-ai", action="store_true",
                    help="don't send the detected software names (never the raw banner) to "
                         f"OpenRouter for an AI explanation (on when "
                         f"{KEY_VARIABLES['openrouter']} is set)")
    ai.add_argument("--ai", action="store_true",
                    help="the default when the key is set; kept for older commands")
    abuse = parser.add_mutually_exclusive_group()
    abuse.add_argument("--no-abuseipdb", action="store_true",
                       help="don't send public target IPs to AbuseIPDB for a reputation check "
                            f"(on when {KEY_VARIABLES['abuseipdb']} is set)")
    abuse.add_argument("--abuseipdb", action="store_true",
                       help="the default when the key is set; kept for older commands")
    parser.add_argument("--scan-unresponsive", action="store_true",
                        help="also scan hosts that don't answer ping (default: skip them)")
    parser.add_argument("--export", type=str.lower, choices=list(EXPORT_FORMATS),
                        help="save the report in instance/scans/ (never overwrites a file)")
    parser.add_argument("--i-am-authorized", action="store_true",
                        help="confirm you are authorised to scan the target (for scripts)")
    nmap = parser.add_argument_group(
        "scan engine", "nmap is used only when it is installed and asked for (never chosen for\n"
                       "you); a scan it can't run stops before anything is sent")
    nmap.add_argument("--engine", type=str.lower, choices=list(ENGINES),
                      help=f"the toolkit's own scanner or nmap (default {DEFAULT_ENGINE})")
    nmap.add_argument("--service-detection", type=str.lower, choices=list(INTENSITIES),
                      help="nmap only: how hard -sV tries (--version-intensity "
                           f"{'/'.join(map(str, INTENSITIES.values()))}; default normal)")
    nmap.add_argument("--timing", type=str.lower, choices=list(TIMINGS),
                      help=f"nmap only: {', '.join(TIMINGS.values())} (default normal)")
    nmap.add_argument("--scripts", type=str.lower, choices=list(SCRIPT_SETS),
                      help=f"nmap only: also run the toolkit's {len(SCRIPTS)} scripts "
                           f"({', '.join(SCRIPTS)})")
    nmap.add_argument("--os-detect", action="store_true",
                      help="nmap only: OS detection (-O; raw packets: Npcap on Windows, root "
                           "elsewhere)")
    saved = parser.add_argument_group(
        "scan history", "every finished scan is saved in instance/scan_history; these read it\n"
                        "and scan nothing (no TARGET, no other flag)")
    one = saved.add_mutually_exclusive_group()
    one.add_argument("--history", action="store_true",
                     help="list the saved scans (newest first, 50 at a time) and exit")
    one.add_argument("--show", metavar="SCAN_ID", help="show one saved scan and exit")
    one.add_argument("--compare", nargs=2, metavar="SCAN_ID",
                     help="compare two saved scans (the older is A) and exit")
    one.add_argument("--delete", metavar="SCAN_ID",
                     help="delete a saved scan's record: names it and asks first")
    saved.add_argument("--before", metavar="SCAN_ID",
                       help="with --history: the scans saved before this one")
    return parser


def check_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """The flags that go together, or the parser's error (exit code 2): the scan flags only
    with a TARGET, nmap's only with --engine nmap, the history flags alone (F6d)."""
    used = ["--" + name.replace("_", "-") for name in SCAN_FLAGS if getattr(args, name)]
    if args.before is not None and not args.history:
        parser.error("--before goes with --history")
    if any(getattr(args, name) for name in HISTORY_FLAGS):
        if args.target is not None or used or args.i_am_authorized:
            parser.error("--history, --show, --compare and --delete go alone: no TARGET and "
                         "no scan flag")
        return
    if args.target is None and used:
        parser.error(f"{', '.join(used)}: only with a TARGET (without one the scanner asks "
                     "its questions)")
    nmap_only = ["--" + name.replace("_", "-") for name in NMAP_FLAGS if getattr(args, name)]
    if nmap_only and args.engine != "nmap":
        parser.error(f"{', '.join(nmap_only)}: only with --engine nmap")


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """The command, without exiting: the flags, the limits and the keys (added to the log's
    secrets), the request from the flags or the questions, then the scan. Its exit code: the scan's,
    2 for a bad flag, target or limit or no authorization, 130 when cancelled."""
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    check_flags(parser, args)
    if any(getattr(args, name) for name in HISTORY_FLAGS):
        return history(args)  # the saved scans (F6d): nothing is scanned
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
    except OSError as error:  # the report not saved: the disk is (too) full (HR4c)
        if error.errno != errno.ENOSPC:
            raise
        screen.say(error)
        return 2


def run(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    """The command: UTF-8 output, the tool's log, then ``main``; exits with its code.
    ``prog`` is the command its help shows (``python -m toolkit scan``)."""
    screen.utf8_output()
    setup_logging("scanner")  # instance/logs/scanner.log; warnings also on the console
    sys.exit(main(argv, prog))


if __name__ == "__main__":
    run()
