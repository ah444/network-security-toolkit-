"""One report entry per open port (B12); every text from the network is cleaned here
(S14). ``toolkit.core.data.recon.scan_files`` saves them."""

from ...data.recon.scan_files import ReportEntry
from .intel import LookupResult
from ...data.recon.scanner import PortResult
from ...data.recon.text import clean_text, one_line


def make_entry(target: str, result: PortResult, software: str | None,
               intel: LookupResult, os_hint: str = "") -> ReportEntry:
    """One open port for the report, with its host's OS hint (``OsHint.summary``); every text
    from the network is cleaned here."""
    return {
        "target": target,
        "port": result.port,
        "protocol": result.protocol,
        "service": one_line(result.service, 40),
        "banner": clean_text(result.banner, 500),
        "software": software or "",
        "cves": intel["cves"],
        "max_cvss": intel["max_cvss"],
        "severity": intel["severity"],
        "ai_explanation": intel["ai_explanation"],
        "lookup_errors": intel["errors"],
        "os_hint": os_hint,
    }
