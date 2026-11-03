"""The saved scans read back (F6b): the list, one scan as the scan page shows a finished one,
with the choices to scan it again, its report, and its delete.

Works on the scan history's records (``history``, in ``instance/scan_history/``). A file there
is data someone could have edited, so a scan id is checked against the form ``history`` writes
before any path is made from it, and a file that isn't a scan record this toolkit can show is
left out of a list and counted, never an error of the whole list. Nothing here changes a
record; a scan is deleted only when the user asks.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...data.recon import scan_records
from ...data.recon.scan_files import report_text
from .history import SCHEMA, is_scan_id
from .targets import parse_ports

PAGE_SIZE = 50  # scans in one list, newest first
# the severities from the worst: NOT CHECKED before the clean ones (never shown as clean, HS4)
WORST_FIRST = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NOT CHECKED", "UNKNOWN", "NO KNOWN CVE")
RISK_ORDER = (*WORST_FIRST, "no lookup", "no open port")  # a scan's ``risk``, the worst first
PROTOCOL_CHOICE = {("tcp",): "tcp", ("udp",): "udp", ("tcp", "udp"): "both"}
DEFAULT_SWITCHES = {"scan_unresponsive": False, "nvd": True, "ai": True, "abuseipdb": True}
NMAP_DEFAULTS = {"service_detection": "normal", "timing": "normal", "scripts": "",
                 "os_detect": False}
# What the user reads about a saved scan, said once for the CLI (cli/ui/scans.py) and the
# report (scan_report.py); the web pages say the same in scanresults.js and scancompare.js.
HOST_STATES = {"scanned": "scanned", "skipped": "skipped: it didn't answer ping",
               "failed": "the engine couldn't scan it"}
ONLINE = (("nvd", "NVD (CVEs, services.nvd.nist.gov)"), ("ai", "AI explanation (OpenRouter)"),
          ("abuseipdb", "IP reputation (AbuseIPDB)"))  # as a scan's opening notice says them
_FIELDS: dict[str, type | tuple[type, ...]] = {
    "started": str, "seconds": (int, float), "source": str, "target": str, "addresses": list,
    "ports": str, "protocols": list, "engine": str, "online": dict, "hosts": list,
    "reputation": dict, "entries": list, "summary": dict, "exit_code": int}


class NotAScan(ValueError):
    """A scan id in the wrong form, or a file that is not a scan record this toolkit can show."""


def when(started: str) -> str:
    """``2026-11-03T10:15:00Z`` as ``2026-11-03 10:15 UTC``."""
    if len(started) >= 16 and started[10] == "T":
        return f"{started[:10]} {started[11:16]} UTC"
    return started


def _path(scan_id: str, directory: str | Path | None) -> Path:
    """The record's path, once ``scan_id`` is checked (so the path stays in the folder)."""
    if not is_scan_id(scan_id):
        raise NotAScan("That is not a scan id.")
    folder = Path(directory) if directory is not None else scan_records.history_dir()
    return folder / f"{scan_id}.json"


def _checked(path: Path) -> dict[str, Any]:
    """The record at ``path`` if it is a scan record the pages can show; NotAScan otherwise."""
    try:
        record = scan_records.read_scan(path)
        if not (isinstance(record, dict) and record.get("kind") == "scan"
                and type(record.get("schema")) is int and 1 <= record["schema"] <= SCHEMA
                and record.get("scan_id") == path.stem):
            raise ValueError("not a scan record of this toolkit")
        if any(not isinstance(record.get(name), kind) for name, kind in _FIELDS.items()):
            raise ValueError("a part is missing")
        summary(record)
        return record
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise NotAScan(f"Scan {path.stem} is not a scan record this toolkit can show.") \
            from error


def _worst(entries: list[Mapping[str, Any]]) -> tuple[str | None, bool]:
    """The worst severity of the entries whose software was looked up (None: no port was, as
    the scan page says "no lookup"), and whether every entry with it is unconfirmed (its worst
    CVE found by the product's name only)."""
    looked_up = [e for e in entries if e["software"]]
    found = [e["severity"] for e in looked_up]
    worst = next((level for level in WORST_FIRST if level in found), None)
    return worst, worst is not None and all(
        e["severity_unconfirmed"] for e in looked_up if e["severity"] == worst)


def summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """One line of the list: when, from where, what, how each host ended, the open ports and
    the worst severity. Raises KeyError or TypeError on a record that lacks a part."""
    states = [host["state"] for host in record["hosts"]]
    worst, unconfirmed = _worst(record["entries"])
    return {
        "scan_id": record["scan_id"], "started": record["started"], "source": record["source"],
        "target": record["target"], "ports": record["ports"],
        "port_count": len(parse_ports(record["ports"])), "protocols": record["protocols"],
        "engine": record["engine"],
        "hosts": {state: states.count(state) for state in ("scanned", "skipped", "failed")},
        "open_ports": len(record["entries"]), "worst": worst, "worst_unconfirmed": unconfirmed,
        "exit_code": record["exit_code"],
    }


def risk(scan: Mapping[str, Any]) -> str:
    """A saved scan's risk in a word (its ``summary``): its worst severity, "no lookup" when
    none of its open ports was looked up, "no open port" when it found none."""
    if scan["worst"] is not None:
        return str(scan["worst"])
    return "no lookup" if scan["open_ports"] else "no open port"


def targets_at_risk(scans: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The dashboard's targets (F10): the newest scan of each target among ``scans`` (newest
    first, as ``saved_scans`` lists them), the riskiest first, the newest first within a risk."""
    newest: dict[str, Mapping[str, Any]] = {}
    for scan in scans:
        newest.setdefault(scan["target"], scan)
    rows = [{"target": scan["target"], "scan_id": scan["scan_id"], "started": scan["started"],
             "risk": risk(scan), "unconfirmed": scan["worst_unconfirmed"],
             "open_ports": scan["open_ports"]} for scan in newest.values()]
    return sorted(rows, key=lambda row: RISK_ORDER.index(row["risk"]))


def saved_scan(scan_id: str, directory: str | Path | None = None) -> dict[str, Any]:
    """The saved record of ``scan_id``: NotAScan for a bad id or record, FileNotFoundError when
    there is none."""
    path = _path(scan_id, directory)
    if not path.is_file():
        raise FileNotFoundError(scan_id)
    return _checked(path)


def saved_scans(before: str | None = None, limit: int = PAGE_SIZE,
                directory: str | Path | None = None) -> dict[str, Any]:
    """The summaries of the saved scans, newest first, at most ``limit``, older than ``before``
    when given: ``{"scans", "more", "left_out"}``. ``left_out`` counts the files met while
    filling this list that are not scan records this toolkit can show (or not named as one)."""
    if before is not None and not is_scan_id(before):
        raise NotAScan("That is not a scan id.")
    paths = [path for path in reversed(scan_records.list_scans(directory))
             if before is None or path.stem < before]
    scans: list[dict[str, Any]] = []
    left_out = read = 0
    for path in paths:
        if len(scans) == limit:
            break
        read += 1
        try:
            if not is_scan_id(path.stem):
                raise NotAScan(path.stem)
            scans.append(summary(_checked(path)))
        except NotAScan:
            left_out += 1
    return {"scans": scans, "more": read < len(paths), "left_out": left_out}


def _engine_notice(record: Mapping[str, Any]) -> list[str]:
    """The engine and its choices in a line (the scan's own notice named this computer's
    folders, so the record keeps the choices instead)."""
    if record["engine"] != "nmap":
        return ["built-in"]
    nmap = {**NMAP_DEFAULTS, **(record.get("nmap") or {})}
    parts = [f"service detection {nmap['service_detection']}", f"timing {nmap['timing']}"]
    parts += [f"the {nmap['scripts']} scripts"] if nmap["scripts"] else []
    parts += ["OS detection"] if nmap["os_detect"] else []
    return ["nmap: " + ", ".join(parts)]


def _again(record: Mapping[str, Any]) -> dict[str, Any]:
    """The scan page's choices to scan the same again (the authorization is asked again)."""
    switches = {**DEFAULT_SWITCHES, **(record.get("choices") or {})}
    again = {"target": record["target"], "ports": record["ports"],
             "protocol": PROTOCOL_CHOICE.get(tuple(sorted(record["protocols"])), "tcp"),
             "engine": record["engine"], **switches}
    if record["engine"] == "nmap":
        again |= {**NMAP_DEFAULTS, **(record.get("nmap") or {})}
    return again


def scan_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """One saved scan as the scan page shows a finished one (its answer's keys, ``ports`` the
    number of ports), with its id, when it started, where it came from, its port list and the
    choices to scan it again (``again``)."""
    failed = [host["target"] for host in record["hosts"] if host["state"] == "failed"]
    return {
        "message": f"Scanned {len(record['hosts'])} host(s): {len(record['entries'])} open "
                   f"port(s)." + (f" Could not scan {len(failed)}." if failed else ""),
        "target": record["target"], "ports": len(parse_ports(record["ports"])),
        "port_list": record["ports"], "protocols": record["protocols"],
        "engine": record["engine"], "engine_notice": _engine_notice(record),
        "online": record["online"],
        "hosts": [{"target": h["target"], "state": h["state"], "os": h["os"]}
                  for h in record["hosts"]],
        "reputation": record["reputation"], "entries": record["entries"],
        "summary": {**record["summary"], "seconds": record["seconds"]},
        "failed_targets": failed,
        "scan_id": record["scan_id"], "started": record["started"], "source": record["source"],
        "again": _again(record),
    }


def saved_report(scan_id: str, fmt: str, directory: str | Path | None = None) -> str:
    """The saved scan's report as text in ``fmt`` (txt, json or csv: the download's); the same
    errors as ``saved_scan``, ValueError for another format."""
    return report_text(saved_scan(scan_id, directory)["entries"], fmt)


def remove_scan(scan_id: str, directory: str | Path | None = None) -> str:
    """Deletes the saved scan's record, the name of the file removed; NotAScan for a bad id,
    FileNotFoundError when there is none, an OSError when it can't be removed (on Windows: a
    file another program has open)."""
    path = _path(scan_id, directory)
    if not path.is_file():
        raise FileNotFoundError(scan_id)
    scan_records.remove_file(path)
    return path.name
