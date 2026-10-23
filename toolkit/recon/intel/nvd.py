"""CVEs and CVSS scores from the NVD CVE API 2.0 (B10). Severity is computed here from
the CVSS score, never taken from the AI (S13).
"""

import re
import threading
import time
from collections.abc import Mapping
from typing import Any

import requests

from ...ports import CveEntry, IntelFinding
from .lookup import INTEL_TIMEOUT

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_RESULTS = 20
# NVD asks for about 6 s between requests without a key (5 per 30 s); 50 per 30 s with one
NVD_INTERVAL = {False: 6.0, True: 0.6}
CVE_ID = re.compile(r"^CVE-\d{4}-\d{4,}$")
METRIC_ORDER = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")


def cvss_severity(score: float | None, version: str = "") -> str:
    """Severity from the CVSS score, computed here (S13: never taken from the AI)."""
    if score is None:
        return "UNKNOWN"
    if str(version).startswith("2"):  # CVSS v2 has three bands
        return "LOW" if score < 4 else "MEDIUM" if score < 7 else "HIGH"
    if score == 0:
        return "NONE"
    return ("LOW" if score < 4 else "MEDIUM" if score < 7 else "HIGH" if score < 9
            else "CRITICAL")


def _cvss(metrics: Mapping[str, Any]) -> tuple[float | None, str]:
    """(base score, CVSS version) of the newest CVSS version present, primary source first."""
    for name in METRIC_ORDER:
        entries = [m for m in metrics.get(name, []) if isinstance(m.get("cvssData"), dict)]
        entries.sort(key=lambda m: m.get("type") != "Primary")
        for metric in entries:
            score = metric["cvssData"].get("baseScore")
            if isinstance(score, int | float) and 0 <= score <= 10:
                return float(score), str(metric["cvssData"].get("version", ""))
    return None, ""


def parse_nvd(data: Mapping[str, Any]) -> list[CveEntry]:
    """CVEs of an NVD CVE API 2.0 reply with CVSS score and severity, worst first."""
    found: list[CveEntry] = []
    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        if not isinstance(cve_id, str) or not CVE_ID.match(cve_id):
            continue
        score, version = _cvss(cve.get("metrics", {}))
        found.append({"id": cve_id, "cvss": score, "cvss_version": version,
                      "severity": cvss_severity(score, version)})
    return sorted(found, key=lambda c: (-(c["cvss"] or 0), c["id"]))


class NvdProvider:
    """CVEs from the NVD CVE API 2.0 keyword search (IntelProvider, kind "cves"); keeps
    NVD's request spacing and caches each product's answer. Thread-safe (step 1.4e): each
    request takes the next start slot, NVD_INTERVAL after the previous request's start, so
    parallel lookups never start more than NVD's limit in any 30 s."""

    name = "NVD"
    kind = "cves"

    def __init__(self, session: requests.Session, key: str = "",
                 timeout: float = INTEL_TIMEOUT) -> None:
        self.session = session
        self.key = key
        self.timeout = timeout
        self._cache: dict[str, IntelFinding] = {}
        self._next = 0.0  # the earliest start of the next request (time.monotonic)
        self._lock = threading.Lock()

    def lookup(self, subject: str) -> IntelFinding:
        with self._lock:
            cached = self._cache.get(subject)
            if cached is not None:
                return cached
            start = max(time.monotonic(), self._next)
            self._next = start + NVD_INTERVAL[bool(self.key)]
        wait = start - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        params: dict[str, str | int] = {"keywordSearch": subject, "resultsPerPage": NVD_RESULTS}
        response = self.session.get(NVD_URL, params=params,
                                    headers={"apiKey": self.key} if self.key else {},
                                    timeout=self.timeout)
        response.raise_for_status()
        finding = IntelFinding(cves=tuple(parse_nvd(response.json())))
        with self._lock:
            self._cache[subject] = finding
        return finding
