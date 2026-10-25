"""``toolkit web``: the web trainer's flags (``--port``, ``--no-browser``), then the server.

Started by ``toolkit.start`` (v5.2 step 2.3c), never by the command line; here, among the
web trainer's controllers, since step 2.3d (it was ``toolkit/controllers/cli/web.py``)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence

from ...core.data.settings import check_settings, port_number, web_config
from ..ui import console as messages
from .server import run_server

HELP_FLAGS = {"-h", "--help"}


def _port_argument(text: str) -> int:
    """``--port`` as a port number; argparse's error with the message otherwise."""
    try:
        return port_number(text, "--port")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """The command's flags (each one is in docs/reference.md)."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="The web trainer, on this computer only (127.0.0.1). It opens in your "
                    "default browser. Other settings: TOOLKIT_DEBUG and the other TOOLKIT_* "
                    "environment variables, in docs/reference.md.")
    parser.add_argument("--port", type=_port_argument,
                        help="port number (default: TOOLKIT_PORT, else 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open the browser (also TOOLKIT_NO_BROWSER=1)")
    return parser


def main(argv: Sequence[str], prog: str) -> int:
    """From ``toolkit.start``: a bad ``TOOLKIT_*`` setting of the web trainer stops it, one
    line each and exit code 2, before its log starts (as the command line does for its
    tools, v5.1 step 1.5e; ``--help`` always works); else ``run``; 0 once the server stops."""
    problems = [] if HELP_FLAGS.intersection(argv) else check_settings("web")
    if problems:
        messages.bad_settings(problems)
        return 2
    run(argv, prog)
    return 0


def run(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    """The command (``python -m toolkit web``): its flags, then ``run_server``; exits with
    code 2 for a bad setting or a port in use."""
    args = build_parser(prog).parse_args(argv)
    try:
        config = web_config()
    except ValueError as error:  # a TOOLKIT_* setting that isn't a number, a bad port
        messages.bad_setting(error)
        sys.exit(2)
    if args.port:
        config = dataclasses.replace(config, port=args.port)
    if args.no_browser:
        config = dataclasses.replace(config, open_browser=False)
    code = run_server(config)
    if code:
        sys.exit(code)
