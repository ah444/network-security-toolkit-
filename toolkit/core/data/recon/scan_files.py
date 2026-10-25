"""Scan report files: one entry per open port, saved as UTF-8 JSON, CSV or TXT in
``instance/scans/`` without ever overwriting a file (B12). CSV cells that look like
formulas are written as text (S14); the entries are made (and their network text
cleaned) by ``toolkit.core.services.recon.report``.
"""

import csv
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO, TypedDict

from ..settings import instance_dir
from .findings import CveEntry
from .text import csv_cell

EXPORT_FORMATS = ("txt", "json", "csv")
CSV_HEADER = ["Target", "Port", "Protocol", "Service", "Banner", "Software", "Severity (CVSS)",
              "Max CVSS", "CVEs (NVD)", "AI explanation (may be wrong)", "Lookup errors"]


class ReportEntry(TypedDict):
    """One open port in a report (the keys of the JSON export)."""

    target: str
    port: int
    protocol: str
    service: str
    banner: str
    software: str
    cves: list[CveEntry]
    max_cvss: float | None
    severity: str
    ai_explanation: str
    lookup_errors: list[str]


def _cve_list(entry: ReportEntry) -> str:
    return ", ".join(f"{c['id']} ({c['cvss'] if c['cvss'] is not None else '?'})"
                     for c in entry["cves"])


def _write_json(entries: Sequence[ReportEntry], handle: TextIO) -> None:
    json.dump(entries, handle, ensure_ascii=False, indent=2)


def _write_csv(entries: Sequence[ReportEntry], handle: TextIO) -> None:
    writer = csv.writer(handle)
    writer.writerow(CSV_HEADER)
    for e in entries:
        writer.writerow([csv_cell(v) for v in (
            e["target"], e["port"], e["protocol"], e["service"], e["banner"], e["software"],
            e["severity"], e["max_cvss"], _cve_list(e), e["ai_explanation"],
            "; ".join(e["lookup_errors"]))])


def _write_txt(entries: Sequence[ReportEntry], handle: TextIO) -> None:
    for e in entries:
        handle.write(f"Target: {e['target']}\nPort: {e['port']}/{e['protocol']}\n"
                     f"Service: {e['service']}\nBanner: {e['banner']}\n"
                     f"Software: {e['software']}\nSeverity (CVSS): {e['severity']}\n"
                     f"CVEs (NVD): {_cve_list(e) or 'none'}\n"
                     f"AI explanation (may be wrong): {e['ai_explanation']}\n"
                     f"Lookup errors: {'; '.join(e['lookup_errors']) or 'none'}\n"
                     + "-" * 40 + "\n")


WRITERS: dict[str, Callable[[Sequence[ReportEntry], TextIO], None]] = {
    "json": _write_json, "csv": _write_csv, "txt": _write_txt}


def scans_dir() -> Path:
    return instance_dir() / "scans"


def export_results(entries: Sequence[ReportEntry], fmt: str,
                   directory: str | Path | None = None) -> Path:
    """Write the report as UTF-8 to a new file in instance/scans/ (never overwrites)."""
    if fmt not in WRITERS:
        raise ValueError(f"Unknown format {fmt!r}; use one of {', '.join(EXPORT_FORMATS)}.")
    directory = Path(directory) if directory else scans_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for number in range(1000):
        path = directory / f"scan-{stamp}{f'-{number}' if number else ''}.{fmt}"
        try:
            handle = path.open("x", encoding="utf-8", newline="" if fmt == "csv" else None)
        except FileExistsError:
            continue
        with handle:
            WRITERS[fmt](entries, handle)
        return path
    raise FileExistsError(f"Too many reports named scan-{stamp} in {directory}")
