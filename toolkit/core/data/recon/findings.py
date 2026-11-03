"""What the intel gateways (NVD, OpenRouter, AbuseIPDB) return."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

# How a CVE was found, as the reports name it
MATCH_LABELS = {"version": "version match", "name": "name match, unconfirmed"}


class CveEntry(TypedDict):
    """One CVE as every "cves" provider reports it; ``severity`` comes from the score.
    ``match`` says how it was found: "version" (NVD lists the version as affected) or "name"
    (a search on the product's name: the version was not checked, so it is unconfirmed)."""

    id: str
    cvss: float | None
    cvss_version: str
    severity: str
    match: str


def match_label(cve: CveEntry) -> str:
    """How ``cve`` was found, in the reports' words."""
    return MATCH_LABELS.get(cve["match"], "match unknown")


@dataclass(frozen=True)
class IntelFinding:
    """What one provider found; each kind of provider fills its own field."""

    cves: tuple[CveEntry, ...] = ()  # kind "cves"
    source: str = ""  # kind "cves": how they were found ("version match (...)", ...)
    text: str = ""  # kind "explanation": plain text, already made safe to show
    data: Mapping[str, Any] = field(default_factory=dict)  # kind "ip_reputation": raw facts
