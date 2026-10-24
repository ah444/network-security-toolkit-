"""One entry point for the toolkit (v5.1 step 1.4b, plan D4, R9).

    python -m toolkit                  the menu
    python -m toolkit scan [flags]     port scanner         (python -m toolkit scan --help)
    python -m toolkit ids [flags]      IDS with your exports (python -m toolkit ids --help)
    python -m toolkit web              web trainer on this computer

(from the project folder; ``python main.py`` is the same as ``python -m toolkit``)

The first word picks the tool; everything after it goes to that tool's own parser
unchanged, so each tool keeps its flags, help and exit code. The menu runs each tool as a
child process (``python -m toolkit <command>``) from the project folder: its memory (a
large model bundle) is freed when it ends, and Ctrl+C or a crash in it brings the menu back.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import NamedTuple

from .config import REPO_ROOT, instance_dir, is_checkout

PROG = "python -m toolkit"

BANNER = r"""
 _       __       __
| |     / /___   / /_____ ____   ____ ___   ___
| | /| / // _ \ / // ___// __ \ / __ `__ \ / _ \
| |/ |/ //  __// // /__ / /_/ // / / / / //  __/
|__/|__/ \___//_/ \___/ \____//_/ /_/ /_/ \___/

Network Recon and Detection Tool
Your one-stop solution for network recon and intrusion detection using Machine Learning.
Let's secure your network with these tools."""

MENU = ("1. Network Recon", "2. Intrusion Detection (CLI)", "3. Intrusion Detection (Web)",
        "4. Exit")
MENU_COMMANDS = {"1": "scan", "2": "ids", "3": "web"}
EXIT_CHOICES = {"4", "q", "quit", "exit"}


# The tools are imported only when they run, so the menu and --help start at once
# (the IDS loads pandas and scikit-learn, the web trainer Flask).

def _scan(argv: list[str], prog: str) -> None:
    from .recon.cli import run
    run(argv, prog)


def _ids(argv: list[str], prog: str) -> None:
    from .ids_cli.cli import run
    run(argv, prog)


def _web(argv: list[str], prog: str) -> None:
    from .web import run
    run(argv, prog)


class Command(NamedTuple):
    help: str
    start: Callable[[list[str], str], None]  # (arguments after the command, prog); exits


COMMANDS = {
    "scan": Command("port scanner: open ports, banners, an OS guess and known CVEs", _scan),
    "ids": Command("intrusion detection with a model you exported from the web trainer", _ids),
    "web": Command("the web trainer on this computer (http://127.0.0.1:5000/ by default)",
                   _web),
}


def build_parser() -> argparse.ArgumentParser:
    """For ``--help`` and errors only: a command's own arguments go to its own parser."""
    parser = argparse.ArgumentParser(
        prog=PROG, description="Network Security Toolkit: network recon and ML intrusion "
                               "detection.\nWithout a command: the menu.",
        epilog=f"Help for one tool: {PROG} <command> --help\n"
               f"Data (exports, runs, signing key, logs): {instance_dir()}",
        formatter_class=argparse.RawDescriptionHelpFormatter)  # the epilog's two lines
    commands = parser.add_subparsers(dest="command", metavar="command")
    for name, command in COMMANDS.items():
        commands.add_parser(name, help=command.help, add_help=False)
    return parser


def run_child(command: str) -> int:
    """``python -m toolkit <command>`` as a child process from the project folder (an
    installed copy: the current folder); its code."""
    try:
        return subprocess.run([sys.executable, "-m", "toolkit", command],
                              cwd=REPO_ROOT if is_checkout() else None, check=False).returncode
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except OSError as error:
        print(f"Could not start {command}: {error}")
        return 1


def menu() -> int:
    """The flat menu: the banner once, then the choices after every tool, until Exit."""
    print(BANNER)
    while True:
        print()
        print("\n".join(MENU))
        try:
            choice = input("Enter your choice [1-4]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if choice in MENU_COMMANDS:
            run_child(MENU_COMMANDS[choice])
        elif choice in EXIT_CHOICES:
            print("Goodbye!")
            return 0
        else:
            print(f"{choice!r} is not a choice: enter 1, 2, 3 or 4 (q also exits).")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return menu()
    name, rest = args[0], args[1:]
    if name in COMMANDS:
        COMMANDS[name].start(rest, f"{PROG} {name}")  # the tool exits with its own code
        return 0
    build_parser().parse_args(args)  # --help prints and exits 0, anything else exits 2
    return 2


def run(argv: Sequence[str] | None = None) -> None:
    """The command: ``main``; exits with its code."""
    sys.exit(main(argv))
