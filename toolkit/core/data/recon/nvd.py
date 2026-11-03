"""CVEs and CVSS scores from the NVD CVE API 2.0 (B10, step A4).

A product in ``CPE_PRODUCTS`` is matched by its version (``virtualMatchString``: the CVEs whose
affected versions, as NVD records them, include it). Any other product, or one the version
match found nothing for, is looked up by a keyword search on its name and version, which can
list CVEs of other versions and miss some. Each answer says which it was
(``IntelFinding.source``). Rejected CVEs are left out. Severity is computed here from the CVSS
score, never taken from the AI (S13).
"""

import re
import threading
import time
from collections.abc import Mapping
from typing import Any

import requests

from .findings import CveEntry, IntelFinding
from .lookup import INTEL_TIMEOUT
from .scanner import ScanStopped

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_RESULTS = 20  # a keyword search's CVEs
PAGE = 500  # CVEs per request of a version match
MAX_CVES = 2000  # per CPE name: at most 4 requests
# NVD asks for about 6 s between requests without a key (5 per 30 s); 50 per 30 s with one
NVD_INTERVAL = {False: 6.0, True: 0.6}
CVE_ID = re.compile(r"^CVE-\d{4}-\d{4,}$")
METRIC_ORDER = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")
VERSION = re.compile(r"^\d+(?:\.\d+)*$")
# NVD's vendor:product for each product name ``software.product_version`` gives, lower case.
# Each was checked against NVD on 2026-10-25 with a version known to have CVEs (the list and
# the counts: v5.2 step A4); nginx's CVEs are all under f5 now.
CPE_PRODUCTS: dict[str, tuple[str, ...]] = {
    "openssh": ("openbsd:openssh",),
    "apache": ("apache:http_server",),
    "nginx": ("f5:nginx",),
    "microsoft-iis": ("microsoft:internet_information_services",),
    "vsftpd": ("vsftpd_project:vsftpd",),
    "proftpd": ("proftpd:proftpd",),
    "filezilla server": ("filezilla-project:filezilla_server",),
    "mysql": ("oracle:mysql", "mysql:mysql"),  # older MySQL CVEs are under mysql:mysql
    "mariadb": ("mariadb:mariadb",),
    "redis": ("redis:redis",),
    "memcached": ("memcached:memcached",),
    "lighttpd": ("lighttpd:lighttpd",),
    "dropbear": ("dropbear_ssh_project:dropbear_ssh",),
    "libssh": ("libssh:libssh",),
    "exim": ("exim:exim",),
    "sendmail": ("sendmail:sendmail",),
    "elasticsearch": ("elastic:elasticsearch",),
}


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


def _worst_first(cves: list[CveEntry]) -> list[CveEntry]:
    """The CVEs by CVSS score, highest first (no score last), then by ID."""
    return sorted(cves, key=lambda c: (-(c["cvss"] or 0), c["id"]))


def parse_nvd(data: Mapping[str, Any], match: str) -> list[CveEntry]:
    """CVEs of an NVD CVE API 2.0 reply with CVSS score and severity, worst first; ``match``
    is how the request found them ("version" or "name")."""
    found: list[CveEntry] = []
    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        if not isinstance(cve_id, str) or not CVE_ID.match(cve_id):
            continue
        score, version = _cvss(cve.get("metrics", {}))
        found.append({"id": cve_id, "cvss": score, "cvss_version": version,
                      "severity": cvss_severity(score, version), "match": match})
    return _worst_first(found)


def cpe_names(software: str) -> list[str]:
    """The CPE names that match ``software`` ("OpenSSH 8.9") by version, or [] when its product
    is not in ``CPE_PRODUCTS`` or it has no plain version."""
    name, _, version = software.rpartition(" ")
    if not name or not VERSION.match(version):
        return []
    return [f"cpe:2.3:a:{product}:{version}" for product in CPE_PRODUCTS.get(name.lower(), ())]


class NvdProvider:
    """CVEs from the NVD CVE API 2.0 (IntelProvider, kind "cves"): by version for the products
    of ``CPE_PRODUCTS``, else by keyword; keeps NVD's request spacing and caches each product's
    answer. Thread-safe (step 1.4e): each request takes the next start slot, NVD_INTERVAL
    after the previous request's start, so parallel lookups never start more than NVD's limit
    in any 30 s. A set ``stop`` ends the wait for a slot at once (ScanStopped, v5.2 step
    2.4c): without a key a slot can be a minute away."""

    name = "NVD"
    kind = "cves"

    def __init__(self, session: requests.Session, key: str = "",
                 timeout: float = INTEL_TIMEOUT, stop: threading.Event | None = None) -> None:
        self.session = session
        self.key = key
        self.timeout = timeout
        self.stop = stop or threading.Event()
        self._cache: dict[str, IntelFinding] = {}
        self._next = 0.0  # the earliest start of the next request (time.monotonic)
        self._lock = threading.Lock()

    def lookup(self, subject: str) -> IntelFinding:
        """The CVEs NVD has for ``subject`` (a product and version), from the cache if this scan
        already asked."""
        with self._lock:
            cached = self._cache.get(subject)
        if cached is not None:
            return cached
        finding = self._find(subject)
        with self._lock:
            self._cache[subject] = finding
        return finding

    def _find(self, subject: str) -> IntelFinding:
        """By version when the product has CPE names and that finds CVEs, else by keyword; the
        source says which, and why a keyword search was used."""
        names = cpe_names(subject)
        if names:
            cves = self._by_version(names)
            if cves:
                return IntelFinding(cves=tuple(cves), source=f"version match ({', '.join(names)})")
            why = "the version match found none"
        else:
            why = "no version match for this product"
        data = self._get({"keywordSearch": subject, "resultsPerPage": NVD_RESULTS})
        return IntelFinding(cves=tuple(parse_nvd(data, "name")),
                            source=f"keyword search ({why})")

    def _by_version(self, names: list[str]) -> list[CveEntry]:
        """Every CVE that matches one of ``names`` (at most MAX_CVES each), once, worst first."""
        found: dict[str, CveEntry] = {}
        for name in names:
            for start in range(0, MAX_CVES, PAGE):
                data = self._get({"virtualMatchString": name, "resultsPerPage": PAGE,
                                  "startIndex": start})
                page = parse_nvd(data, "version")
                for cve in page:
                    found.setdefault(cve["id"], cve)
                if not page or start + PAGE >= int(data.get("totalResults", 0)):
                    break
        return _worst_first(list(found.values()))

    def _get(self, params: Mapping[str, str | int]) -> Mapping[str, Any]:
        """One NVD request (rejected CVEs left out), in its start slot: it waits for it first,
        and sends nothing once ``stop`` is set (ScanStopped)."""
        with self._lock:
            start = max(time.monotonic(), self._next)
            self._next = start + NVD_INTERVAL[bool(self.key)]
        if self.stop.wait(max(0.0, start - time.monotonic())):
            raise ScanStopped("the scan was stopped before this NVD request")
        response = self.session.get(NVD_URL, params={**params, "noRejected": ""},
                                    headers={"apiKey": self.key} if self.key else {},
                                    timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):  # a lookup error (R3), not a crash
            raise ValueError("NVD's reply is not a JSON object")
        return data
