"""Online lookups for open ports (B10, S12, S13, R3): the ``Intel`` façade over the
IntelProviders (``toolkit.ports.IntelProvider``): NVD, OpenRouter, AbuseIPDB.
"""

from collections.abc import Iterable, Mapping
from typing import Any, TypedDict

import requests

from ...ports import CveEntry, IntelFinding, IntelProvider
from .abuseipdb import AbuseIpdbProvider
from .keys import ai_model
from .llm import OpenRouterProvider
from .lookup import INTEL_TIMEOUT, LOOKUP_ERRORS, describe
from .nvd import NvdProvider


class LookupResult(TypedDict):
    """What ``Intel.lookup`` found about one product."""

    cves: list[CveEntry]  # worst first
    max_cvss: float | None
    severity: str  # of the worst CVE; "NO KNOWN CVE" if none, "UNKNOWN" if no source answered
    ai_explanation: str
    errors: list[str]  # one per failed lookup: "<provider>: <reason>"


class Intel:
    """Online lookups for open ports, one at a time, each with a timeout (R3). A failed
    lookup is recorded and the scan goes on.

    The sources are IntelProviders: by default NVD, plus OpenRouter after consent and
    AbuseIPDB when there is a key; ``providers`` replaces them (offline sources, tests).
    """

    def __init__(self, keys: Mapping[str, str], use_ai: bool = False,
                 session: requests.Session | None = None, timeout: float = INTEL_TIMEOUT,
                 providers: Iterable[IntelProvider] | None = None) -> None:
        self.keys = keys
        self.use_ai = bool(use_ai and keys.get("openrouter"))
        self.session = session or requests.Session()
        self.timeout = timeout
        self.model = ai_model()
        if providers is None:
            defaults: list[IntelProvider] = [
                NvdProvider(self.session, keys.get("nvd", ""), timeout)]
            if self.use_ai:
                defaults.append(OpenRouterProvider(self.session, keys["openrouter"],
                                                   self.model, timeout))
            if keys.get("abuseipdb"):
                defaults.append(AbuseIpdbProvider(self.session, keys["abuseipdb"], timeout))
            providers = defaults
        self.providers = list(providers)

    def _ask(self, kind: str, subject: str, errors: list[str]) -> list[IntelFinding]:
        """Findings of every provider of ``kind``; failures go to ``errors``."""
        found = []
        for provider in (p for p in self.providers if p.kind == kind):
            try:
                found.append(provider.lookup(subject))
            except LOOKUP_ERRORS as error:
                errors.append(f"{provider.name}: {describe(error)}")
        return found

    def lookup(self, software: str) -> LookupResult:
        """CVEs, worst CVSS severity and the optional AI text for one product."""
        result: LookupResult = {"cves": [], "max_cvss": None, "severity": "UNKNOWN",
                                "ai_explanation": "", "errors": []}
        answers = self._ask("cves", software, result["errors"])
        if answers:
            cves = [cve for answer in answers for cve in answer.cves]
            result["cves"] = sorted(cves, key=lambda c: (-(c["cvss"] or 0), c["id"]))
            worst = result["cves"][0] if result["cves"] else None
            result["max_cvss"] = worst["cvss"] if worst else None
            result["severity"] = worst["severity"] if worst else "NO KNOWN CVE"
        texts = self._ask("explanation", software, result["errors"])
        result["ai_explanation"] = "\n\n".join(t.text for t in texts if t.text)
        return result

    def abuse_report(self, ip: str) -> tuple[dict[str, Any] | None, str | None]:
        """(reputation data, error message)."""
        errors: list[str] = []
        if not any(p.kind == "ip_reputation" for p in self.providers):
            return None, "AbuseIPDB: no API key"
        found = self._ask("ip_reputation", ip, errors)
        return (dict(found[0].data), None) if found else (None, errors[0])
