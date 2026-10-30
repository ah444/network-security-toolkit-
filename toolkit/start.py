"""The toolkit's one entry point (v5.2 step 2.3c): ``python -m toolkit``, ``python main.py``
and the installed ``toolkit`` command all start here.

The first word picks the interface: ``web`` is the web trainer's command, anything else the
command line (the menu, ``scan``, ``ids``, the toolkit's own ``--help``). Each is imported
only when it runs, so the command line never imports the web trainer (Flask) and the web
trainer never imports the command line; the menu starts ``web`` as a child process.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

PROG = "python -m toolkit"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command in ``argv`` (default: this process's arguments); its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["web"]:
        from .web.controllers.command import main as web

        return web(args[1:], f"{PROG} web")
    from .cli.controllers.main import main as command_line

    return command_line(args)


def run(argv: Sequence[str] | None = None) -> None:
    """``main``, then exit with its code."""
    sys.exit(main(argv))
