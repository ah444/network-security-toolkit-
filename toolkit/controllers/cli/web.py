"""``toolkit web``: the web trainer's flags (``--port``, ``--no-browser``), then the server."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence

from ...core.data.settings import port_number, web_config
from ...ui.cli import web as messages
from ..web.server import run_server


def _port_argument(text: str) -> int:
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
