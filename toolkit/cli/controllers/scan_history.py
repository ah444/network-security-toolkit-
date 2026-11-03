"""The scan history from the command line (v5.4 F6d): ``scan --history`` (``--before``),
``--show SCAN_ID``, ``--compare A B``, ``--delete SCAN_ID`` and ``--report SCAN_ID`` (F5c).
They read the saved scans (``toolkit.core.services.recon.saved_scans``) and never scan or
send anything; a scan is deleted only after its record was named and the user said yes."""

from __future__ import annotations

import argparse

from ...core.services.recon.saved_scans import (
    NotAScan,
    remove_scan,
    saved_scan,
    saved_scans,
    scan_view,
)
from ...core.services.recon.scan_comparison import compare_saved
from ...core.services.recon.scan_report import export_saved_report
from ..ui import scan as screen
from ..ui import scans as scans_screen

HISTORY_FLAGS = ("history", "show", "compare", "delete", "report")  # each alone, no TARGET


def list_scans(before: str | None = None) -> int:
    """``--history``: the saved scans, newest first, 50 at a time. Exit code 0."""
    scans_screen.scans_list(saved_scans(before))
    return 0


def show_scan(scan_id: str) -> int:
    """``--show SCAN_ID``: one saved scan, with the command that scans it again. Exit code 0."""
    record = saved_scan(scan_id)
    scans_screen.show(scan_view(record), record["hosts"])
    return 0


def compare_scans(a: str, b: str) -> int:
    """``--compare A B``: two saved scans compared, the older as A whatever the order. Exit 0."""
    older, newer = sorted((a, b))
    scans_screen.comparison(compare_saved(older, newer))
    return 0


def delete_scan(scan_id: str) -> int:
    """``--delete SCAN_ID``: names the record and asks; only a yes removes it. Exit code 0, or
    2 when it could not be removed (the scan is kept)."""
    saved_scan(scan_id)  # a bad id or no such scan: said before anything is asked
    scans_screen.delete_notice(scan_id)
    if not scans_screen.confirm_delete():
        scans_screen.deleted(None)
        return 0
    try:
        remove_scan(scan_id)
    except FileNotFoundError:
        raise
    except OSError:
        screen.say(f"{scan_id}.json could not be removed: another program may have it open. "
                   "Close it and try again.")
        return 2
    scans_screen.deleted(scan_id)
    return 0


def report_scan(scan_id: str) -> int:
    """``--report SCAN_ID`` (F5c): the HTML report written as a new file in instance/scans/.
    Exit code 0, or 2 when it couldn't be written (a full disk: nothing is left half-written).
    See also: core/services/recon/scan_report.py (what the report says)."""
    try:
        path = export_saved_report(scan_id)
    except (NotAScan, FileNotFoundError):
        raise
    except OSError as error:
        screen.say(error)
        return 2
    scans_screen.report_written(path)
    return 0


def history(args: argparse.Namespace) -> int:
    """The history flag given; exit code 2 with the message for a bad scan id, a file that is
    not a scan, or a scan that isn't saved."""
    try:
        if args.show:
            return show_scan(args.show)
        if args.compare:
            return compare_scans(*args.compare)
        if args.delete:
            return delete_scan(args.delete)
        if args.report:
            return report_scan(args.report)
        return list_scans(args.before)
    except NotAScan as error:
        screen.say(error)
    except FileNotFoundError as error:
        screen.say(f"There is no saved scan {error}.")
    return 2
