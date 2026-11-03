"""Two saved scans compared (F6c), address by address and port by port; A is the older.

The comparison says only what both scans saw. A port open in one scan only is ``opened`` or
``closed`` only when the other scan checked it (its port list and protocols), else it was
``not_scanned`` there. A host that one scan skipped (no answer to the ping) or couldn't scan is
not compared. CVEs are compared only when both scans looked the port's software up: NVD off or
unreachable in one scan makes them "not checked", never gone. Scans of different targets are
compared on the addresses both scanned, and the answer names the others.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .history import is_scan_id
from .saved_scans import NotAScan, saved_scan, summary
from .targets import parse_ports

Key = tuple[int, str]  # (port, protocol)
SIDE_FIELDS = ("service", "software", "severity", "max_cvss")  # what a "changed" compares
NOT_CHECKED = "NOT CHECKED"


def _checked_keys(record: Mapping[str, Any]) -> set[Key]:
    """Every port and protocol the scan checked."""
    return {(number, protocol) for number in parse_ports(record["ports"])
            for protocol in record["protocols"]}


def _open(record: Mapping[str, Any], address: str) -> dict[Key, Mapping[str, Any]]:
    """The open ports the scan found on ``address``, by port and protocol."""
    return {(e["port"], e["protocol"]): e for e in record["entries"] if e["target"] == address}


def _side(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What a row shows of one scan's port."""
    if entry is None:
        return None
    return {**{name: entry[name] for name in SIDE_FIELDS},
            "severity_unconfirmed": entry["severity_unconfirmed"], "cves": len(entry["cves"])}


def _cves(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[list[str] | None,
                                                               list[str] | None, str]:
    """The CVEs B has that A hasn't and the other way, when both scans looked them up; else
    None and why not."""
    for entry, side in ((a, "A"), (b, "B")):
        if not entry["software"]:
            return None, None, "software not recognised"
        if entry["severity"] == NOT_CHECKED:
            return None, None, f"not checked in {side}"
    ids_a = [c["id"] for c in a["cves"]]
    ids_b = [c["id"] for c in b["cves"]]
    return ([i for i in ids_b if i not in ids_a], [i for i in ids_a if i not in ids_b], "")


def _row(key: Key, a: Mapping[str, Any] | None, b: Mapping[str, Any] | None,
         checked_a: set[Key], checked_b: set[Key]) -> dict[str, Any]:
    """One port of a host both scans scanned."""
    row: dict[str, Any] = {"port": key[0], "protocol": key[1], "a": _side(a), "b": _side(b),
                           "changes": [], "cves_added": None, "cves_gone": None,
                           "cves_note": ""}
    if a is None:
        row["status"] = "opened" if key in checked_a else "not_scanned_in_a"
    elif b is None:
        row["status"] = "closed" if key in checked_b else "not_scanned_in_b"
    else:
        row["changes"] = [name for name in SIDE_FIELDS if a[name] != b[name]]
        row["cves_added"], row["cves_gone"], row["cves_note"] = _cves(a, b)
        changed = row["changes"] or row["cves_added"] or row["cves_gone"]
        row["status"] = "changed" if changed else "same"
    return row


def _host(address: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """One address both scans had: its state in each, and its ports when both scanned it."""
    state_a = next(h["state"] for h in a["hosts"] if h["target"] == address)
    state_b = next(h["state"] for h in b["hosts"] if h["target"] == address)
    compared = state_a == state_b == "scanned"
    rows = []
    if compared:
        open_a, open_b = _open(a, address), _open(b, address)
        checked_a, checked_b = _checked_keys(a), _checked_keys(b)
        rows = [_row(key, open_a.get(key), open_b.get(key), checked_a, checked_b)
                for key in sorted(open_a.keys() | open_b.keys())]
    return {"address": address, "a": state_a, "b": state_b, "compared": compared, "rows": rows}


def _totals(hosts: list[dict[str, Any]]) -> dict[str, int]:
    """How many ports opened, closed, changed, stayed the same or were not scanned in one
    scan, and the CVEs added and gone."""
    rows = [row for host in hosts for row in host["rows"]]
    status = [row["status"] for row in rows]
    return {"opened": status.count("opened"), "closed": status.count("closed"),
            "changed": status.count("changed"), "same": status.count("same"),
            "not_scanned": sum(s.startswith("not_scanned") for s in status),
            "cves_added": sum(len(row["cves_added"] or []) for row in rows),
            "cves_gone": sum(len(row["cves_gone"] or []) for row in rows)}


def compare_scans(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Two scan records compared (A the older): both summaries, whether they had the same
    target, the addresses both had and each one's own, each common address's ports, and the
    totals."""
    common = [address for address in a["addresses"] if address in b["addresses"]]
    hosts = [_host(address, a, b) for address in common]
    return {"a": summary(a), "b": summary(b), "same_target": a["target"] == b["target"],
            "common": common,
            "only_in_a": [address for address in a["addresses"] if address not in common],
            "only_in_b": [address for address in b["addresses"] if address not in common],
            "hosts": hosts, "totals": _totals(hosts)}


def compare_saved(a: str, b: str, directory: str | Path | None = None) -> dict[str, Any]:
    """``compare_scans`` of two saved scans: NotAScan for a bad id, the same id twice or a file
    that isn't a scan record, FileNotFoundError when one has none (both ids are checked before
    either record is read)."""
    if a == b:
        raise NotAScan("Choose two different scans.")
    if not (is_scan_id(a) and is_scan_id(b)):
        raise NotAScan("That is not a scan id.")
    return compare_scans(saved_scan(a, directory), saved_scan(b, directory))
