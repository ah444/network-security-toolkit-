"""Online lookups for open ports (B10, S12, S13, R3): the ``Intel`` façade over the
IntelProviders (``toolkit.core.services.ports.IntelProvider``): NVD, OpenRouter, AbuseIPDB.
"""

import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict

from ...data.recon.findings import CveEntry, IntelFinding
from ..ports import IntelProvider
from ...data.recon.abuseipdb import ABUSEIPDB_URL, AbuseIpdbProvider
from ...data.keys import ai_model
from ...data.recon.openrouter import OPENROUTER_API_URL, OpenRouterProvider
from ...data.recon.lookup import (
    INTEL_TIMEOUT,
    INTEL_WORKERS,
    LOOKUP_ERRORS,
    Session,
    describe,
    host_of,
)
from ...data.recon.nvd import NVD_URL, NvdProvider

# The online services a scan may ask, by host, and their names in its totals (step A5)
SERVICES = {host_of(url): name for url, name in (
    (NVD_URL, "NVD"), (OPENROUTER_API_URL, "AI (OpenRouter)"), (ABUSEIPDB_URL, "AbuseIPDB"))}


class LookupResult(TypedDict):
    """What ``Intel.lookup`` found about one product."""

    cves: list[CveEntry]  # worst first
    cve_source: str  # how they were found: "version match (...)" or "keyword search (...)"
    max_cvss: float | None
    severity: str  # of the worst CVE; "NO KNOWN CVE" if none, "UNKNOWN" if no source answered
    ai_explanation: str
    errors: list[str]  # one per failed lookup: "<provider>: <reason>"


class Intel:
    """Online lookups for open ports, each with a timeout (R3); ``lookup_many`` runs up to
    INTEL_WORKERS at a time (step 1.4e). A failed lookup is recorded and the scan goes on;
    a product looked up without an error is not sent again in this scan.

    The sources are IntelProviders: by default NVD, plus OpenRouter after consent and
    AbuseIPDB when there is a key; ``providers`` replaces them (offline sources, tests).
    """

    def __init__(self, keys: Mapping[str, str], use_ai: bool = False,
                 session: Session | None = None, timeout: float = INTEL_TIMEOUT,
                 providers: Iterable[IntelProvider] | None = None) -> None:
        self.keys = keys
        self.use_ai = bool(use_ai and keys.get("openrouter"))
        self.session = session or Session()
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
        self._done: dict[str, LookupResult] = {}  # products looked up without an error
        self._lock = threading.Lock()

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
        """CVEs, worst CVSS severity and the optional AI text for one product; once per
        scan, unless it failed (then it is tried again the next time, e.g. on the next host)."""
        with self._lock:
            if software in self._done:
                return self._done[software]
        result: LookupResult = {"cves": [], "cve_source": "", "max_cvss": None,
                                "severity": "UNKNOWN", "ai_explanation": "", "errors": []}
        answers = self._ask("cves", software, result["errors"])
        if answers:
            cves = [cve for answer in answers for cve in answer.cves]
            result["cves"] = sorted(cves, key=lambda c: (-(c["cvss"] or 0), c["id"]))
            result["cve_source"] = "; ".join(answer.source for answer in answers if answer.source)
            worst = result["cves"][0] if result["cves"] else None
            result["max_cvss"] = worst["cvss"] if worst else None
            result["severity"] = worst["severity"] if worst else "NO KNOWN CVE"
        texts = self._ask("explanation", software, result["errors"])
        result["ai_explanation"] = "\n\n".join(t.text for t in texts if t.text)
        if not result["errors"]:
            with self._lock:
                self._done[software] = result
        return result

    def lookup_many(self, products: Iterable[str],
                    workers: int = INTEL_WORKERS) -> dict[str, LookupResult]:
        """``lookup`` of each distinct product, up to ``workers`` at a time; in the order
        given. On Ctrl+C the queued lookups are cancelled (running ones end within their
        timeout)."""
        distinct = list(dict.fromkeys(products))
        if not distinct:
            return {}
        results: dict[str, LookupResult] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(distinct)),
                                thread_name_prefix="intel") as executor:
            futures = {executor.submit(self.lookup, product): product for product in distinct}
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        return {product: results[product] for product in distinct}

    def sent(self) -> dict[str, int]:
        """The requests this scan sent to each online service, counted by its session: every
        service in ``SERVICES`` by name (0 if none), any other host by itself; all 0 with a
        session that doesn't count (tests)."""
        counted: Mapping[str, int] = getattr(self.session, "sent", {})
        totals = {name: counted.get(host, 0) for host, name in SERVICES.items()}
        totals.update({host: n for host, n in counted.items() if host not in SERVICES})
        return totals

    def abuse_report(self, ip: str) -> tuple[dict[str, Any] | None, str | None]:
        """(AbuseIPDB's data for ``ip``, None), or (None, why it failed); without a key
        nothing is sent. The caller passes only a public address, after consent."""
        errors: list[str] = []
        if not any(p.kind == "ip_reputation" for p in self.providers):
            return None, "AbuseIPDB: no API key"
        found = self._ask("ip_reputation", ip, errors)
        return (dict(found[0].data), None) if found else (None, errors[0])
