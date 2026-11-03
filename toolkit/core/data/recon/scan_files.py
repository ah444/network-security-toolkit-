"""Scan report files: one entry per open port, saved as UTF-8 JSON, CSV or TXT in
``instance/scans/`` without ever overwriting a file (B12). CSV cells that look like
formulas are written as text (S14); the entries are made (and their network text
cleaned) by ``toolkit.core.services.recon.report``.
"""

import csv
import io
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO, TypedDict

from ..disk import require_free
from ..settings import instance_dir, min_free_mb
from .findings import CveEntry, match_label
from .text import csv_cell

EXPORT_FORMATS = ("txt", "json", "csv")
CSV_HEADER = ["Target", "Port", "Protocol", "Service", "Banner", "Software", "Severity (CVSS)",
              "Max CVSS", "CVEs (NVD)", "AI explanation (may be wrong)", "Lookup errors",
              "OS hint", "CVE source", "Engine", "CPE (nmap)", "Severity confirmed",
              "Unconfirmed CVEs (name match)"]  # the last two since HC3, at the end


class ReportEntry(TypedDict):
    """One open port in a report (the keys of the JSON export)."""

    target: str
    port: int
    protocol: str
    service: str
    banner: str
    software: str
    cves: list[CveEntry]
    cve_source: str  # how NVD found them: by version (CPE names) or by keyword
    max_cvss: float | None
    severity: str
    severity_unconfirmed: bool  # the worst CVE was found by the product's name only
    ai_explanation: str
    lookup_errors: list[str]
    os_hint: str  # the host's OS guess and its evidence, the same on each of its entries
    engine: str  # the scan engine that found the port: builtin or nmap (step N4)
    cpe: list[str]  # nmap's CPE names for the service, as nmap gave them (not looked up)


def _cve_list(entry: ReportEntry) -> str:
    """The entry's CVEs as ``ID (CVSS)``, separated by commas (``?`` for no score)."""
    return ", ".join(f"{c['id']} ({c['cvss'] if c['cvss'] is not None else '?'})"
                     for c in entry["cves"])


def _matched_cves(entry: ReportEntry) -> str:
    """The entry's CVEs as ``ID (CVSS, how it was found)``, separated by commas."""
    return ", ".join(f"{c['id']} ({c['cvss'] if c['cvss'] is not None else '?'}, "
                     f"{match_label(c)})" for c in entry["cves"])


def severity_text(entry: ReportEntry) -> str:
    """The severity, with "(unconfirmed)" when the worst CVE was found by name only."""
    return entry["severity"] + (" (unconfirmed)" if entry["severity_unconfirmed"] else "")


def _confirmed(entry: ReportEntry) -> str:
    """The CSV's "Severity confirmed": empty without a CVE."""
    if not entry["cves"]:
        return ""
    return "no (name match only)" if entry["severity_unconfirmed"] else "yes"


def _write_json(entries: Sequence[ReportEntry], handle: TextIO) -> None:
    """The entries as indented JSON, non-ASCII text kept as it is."""
    json.dump(entries, handle, ensure_ascii=False, indent=2)


def _write_csv(entries: Sequence[ReportEntry], handle: TextIO) -> None:
    """A header row, then a row per entry; every cell through ``csv_cell``, so a cell that starts
    like a formula stays text in a spreadsheet (S14)."""
    writer = csv.writer(handle)
    writer.writerow(CSV_HEADER)
    for e in entries:
        writer.writerow([csv_cell(v) for v in (
            e["target"], e["port"], e["protocol"], e["service"], e["banner"], e["software"],
            e["severity"], e["max_cvss"], _cve_list(e), e["ai_explanation"],
            "; ".join(e["lookup_errors"]), e["os_hint"], e["cve_source"], e["engine"],
            " ".join(e["cpe"]), _confirmed(e),
            ", ".join(c["id"] for c in e["cves"] if c["match"] != "version"))])


def _no_cves(entry: ReportEntry) -> str:
    """What the CVE line says without a CVE: ``not checked`` when they were not looked up
    (HS4: never shown as clean), else ``none``."""
    return "not checked" if entry["severity"] == "NOT CHECKED" else "none"


def _write_txt(entries: Sequence[ReportEntry], handle: TextIO) -> None:
    """A block of lines per entry, separated by a dashed line."""
    for e in entries:
        handle.write(f"Target: {e['target']}\nOS hint: {e['os_hint'] or 'none'}\n"
                     f"Port: {e['port']}/{e['protocol']}\n"
                     f"Service: {e['service']}\nBanner: {e['banner']}\n"
                     f"Software: {e['software']}\nSeverity (CVSS): {severity_text(e)}\n"
                     f"CVEs (NVD): {_matched_cves(e) or _no_cves(e)}\n"
                     f"CVE source: {e['cve_source'] or 'none'}\n"
                     f"AI explanation (may be wrong): {e['ai_explanation']}\n"
                     f"Lookup errors: {'; '.join(e['lookup_errors']) or 'none'}\n"
                     f"Engine: {e['engine']}\nCPE (nmap): {' '.join(e['cpe']) or 'none'}\n"
                     + "-" * 40 + "\n")


WRITERS: dict[str, Callable[[Sequence[ReportEntry], TextIO], None]] = {
    "json": _write_json, "csv": _write_csv, "txt": _write_txt}


def scans_dir() -> Path:
    """``instance/scans``, where the reports are saved."""
    return instance_dir() / "scans"


def report_text(entries: Sequence[ReportEntry], fmt: str) -> str:
    """The report in ``fmt`` (one of EXPORT_FORMATS) as text, written nowhere: the web scan's
    download (v5.2 step 2.4c) and what ``export_results`` saves. Lines end in ``\\n``, CSV rows
    in ``\\r\\n`` (the csv module's own); ValueError for any other format."""
    if fmt not in WRITERS:
        raise ValueError(f"Unknown format {fmt!r}; use one of {', '.join(EXPORT_FORMATS)}.")
    text = io.StringIO()
    WRITERS[fmt](entries, text)
    return text.getvalue()


def export_results(entries: Sequence[ReportEntry], fmt: str,
                   directory: str | Path | None = None) -> Path:
    """Write the report (``report_text``) as UTF-8 to a new file in instance/scans/ (never
    overwrites); TXT and JSON lines end the system's way, CSV rows in ``\\r\\n``."""
    text = report_text(entries, fmt)
    directory = Path(directory) if directory else scans_dir()
    directory.mkdir(parents=True, exist_ok=True)
    require_free(directory, len(text.encode("utf-8")), min_free_mb())  # HR4c: NoSpace
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for number in range(1000):
        path = directory / f"scan-{stamp}{f'-{number}' if number else ''}.{fmt}"
        try:
            handle = path.open("x", encoding="utf-8", newline="" if fmt == "csv" else None)
        except FileExistsError:
            continue
        with handle:
            handle.write(text)
        return path
    raise FileExistsError(f"Too many reports named scan-{stamp} in {directory}")
