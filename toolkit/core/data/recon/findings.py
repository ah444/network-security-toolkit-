"""What the intel gateways (NVD, OpenRouter, AbuseIPDB) return."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict


class CveEntry(TypedDict):
    """One CVE as every "cves" provider reports it; ``severity`` comes from the score."""

    id: str
    cvss: float | None
    cvss_version: str
    severity: str


@dataclass(frozen=True)
class IntelFinding:
    """What one provider found; each kind of provider fills its own field."""

    cves: tuple[CveEntry, ...] = ()  # kind "cves"
    source: str = ""  # kind "cves": how they were found ("version match (...)", ...)
    text: str = ""  # kind "explanation": plain text, already made safe to show
    data: Mapping[str, Any] = field(default_factory=dict)  # kind "ip_reputation": raw facts
