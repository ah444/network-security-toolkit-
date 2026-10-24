"""One entry point for the toolkit (v5.1 step 1.4b, plan D4, R9).

    python -m toolkit                  the menu
    python -m toolkit scan [flags]     port scanner         (python -m toolkit scan --help)
    python -m toolkit ids [flags]      IDS with your exports (python -m toolkit ids --help)
    python -m toolkit web              web trainer on this computer

(from the project folder; ``python main.py`` and, once installed, ``toolkit`` are the same
as ``python -m toolkit``). What the menu shows is ``toolkit.ui.cli.menu``.

The first word picks the tool; everything after it goes to that tool's own parser
unchanged, so each tool keeps its flags, help and exit code. The menu runs each tool as a
child process (``python -m toolkit <command>``) from the project folder: its memory (a
large model bundle) is freed when it ends, and Ctrl+C or a crash in it brings the menu back.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import NamedTuple

from ...data.processes import run_toolkit
from ...data.settings import REPO_ROOT, check_settings, instance_dir, is_checkout
from ...ui.cli import menu as screen

PROG = "python -m toolkit"

MENU_COMMANDS = {"1": "scan", "2": "ids", "3": "web"}
EXIT_CHOICES = {"4", "q", "quit", "exit"}
HELP_FLAGS = {"-h", "--help"}


# The tools are imported only when they run, so the menu and --help start at once
# (the IDS loads pandas and scikit-learn, the web trainer Flask).

def _scan(argv: list[str], prog: str) -> None:
    from .scan import run
    run(argv, prog)


def _ids(argv: list[str], prog: str) -> None:
    from .ids import run
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
        return run_toolkit(command, REPO_ROOT if is_checkout() else None)
    except KeyboardInterrupt:
        screen.stopped()
        return 130
    except OSError as error:
        screen.could_not_start(command, error)
        return 1


def menu() -> int:
    """The flat menu: the banner once, then the choices after every tool, until Exit."""
    screen.show_banner()
    while True:
        try:
            choice = screen.ask_choice()
        except (EOFError, KeyboardInterrupt):
            screen.goodbye(after_interrupt=True)
            return 0
        if choice in MENU_COMMANDS:
            run_child(MENU_COMMANDS[choice])
        elif choice in EXIT_CHOICES:
            screen.goodbye()
            return 0
        else:
            screen.not_a_choice(choice)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return menu()
    name, rest = args[0], args[1:]
    if name in COMMANDS:
        # a bad TOOLKIT_* setting of this tool: one line each, exit code 2, before its log
        # starts (v5.1 step 1.5e); --help always works
        problems = [] if HELP_FLAGS.intersection(rest) else check_settings(name)
        if problems:
            screen.bad_settings(problems)
            return 2
        COMMANDS[name].start(rest, f"{PROG} {name}")  # the tool exits with its own code
        return 0
    build_parser().parse_args(args)  # --help prints and exits 0, anything else exits 2
    return 2


def run(argv: Sequence[str] | None = None) -> None:
    """The command: ``main``; exits with its code."""
    sys.exit(main(argv))
