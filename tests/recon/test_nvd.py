"""B10: CVEs from the NVD CVE API 2.0 (mocked with real NVD replies)."""

import json

import pytest

from toolkit.recon import probes
from toolkit.recon import intel as recon_intel
from toolkit.recon.intel import nvd

# Real replies of the NVD CVE API 2.0, saved on 2026-10-20 (tests/fixtures/README.md).
OPENSSH = "nvd_openssh_8.9.json"
EMPTY = "nvd_no_results.json"


@pytest.fixture
def nvd_reply(fixtures_dir):
    return lambda name: json.loads((fixtures_dir / name).read_text(encoding="utf-8"))


def test_parses_cves_and_cvss(nvd_reply):
    cves = nvd.parse_nvd(nvd_reply(OPENSSH))
    assert cves == [
        {"id": "CVE-2023-28531", "cvss": 9.8, "cvss_version": "3.1", "severity": "CRITICAL"},
        {"id": "CVE-2021-36368", "cvss": 3.7, "cvss_version": "3.1", "severity": "LOW"},
    ], "worst first; CVSS 3.1 primary score preferred over v2 and secondary sources"
    assert nvd.parse_nvd(nvd_reply(EMPTY)) == []


def test_severity_bands():
    scores = (None, 0, 0.1, 3.9, 4.0, 6.9, 7.0, 8.9, 9.0, 10)
    assert [nvd.cvss_severity(s) for s in scores] == [
        "UNKNOWN", "NONE", "LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "CRITICAL",
        "CRITICAL"]
    assert nvd.cvss_severity(10.0, "2.0") == "HIGH"  # CVSS v2 has no CRITICAL band


def test_bad_entries_are_skipped():
    data = {"vulnerabilities": [
        {"cve": {"id": "CVE-2024-0001<script>", "metrics": {}}},
        {"cve": {"id": "CVE-2024-12345", "metrics": {"cvssMetricV31": [
            {"type": "Primary", "cvssData": {"version": "3.1", "baseScore": 42}}]}}},
        {"cve": {}},
    ]}
    assert nvd.parse_nvd(data) == [
        {"id": "CVE-2024-12345", "cvss": None, "cvss_version": "", "severity": "UNKNOWN"}]


def test_lookup_sends_only_the_software_name(nvd_reply, fake_session, no_nvd_wait):
    session = fake_session(get=nvd_reply(OPENSSH))
    intel = recon_intel.Intel({"nvd": "nvd-test-key"}, session=session)
    found = intel.lookup("OpenSSH 8.9")
    assert [c["id"] for c in found["cves"]] == ["CVE-2023-28531", "CVE-2021-36368"]
    assert (found["severity"], found["max_cvss"], found["errors"]) == ("CRITICAL", 9.8, [])
    ((method, url, kwargs),) = session.calls
    assert url == nvd.NVD_URL and kwargs["params"]["keywordSearch"] == "OpenSSH 8.9"
    assert kwargs["headers"] == {"apiKey": "nvd-test-key"} and kwargs["timeout"]
    intel.lookup("OpenSSH 8.9")
    assert len(session.calls) == 1, "cached per software"
    no_cves = recon_intel.Intel({}, session=fake_session(get=nvd_reply(EMPTY))).lookup("nginx 1.25")
    assert no_cves["severity"] == "NO KNOWN CVE"


def test_requests_are_spaced_without_a_key(nvd_reply, fake_session, monkeypatch):
    sleeps = []
    monkeypatch.setattr(nvd.time, "sleep", sleeps.append)
    intel = recon_intel.Intel({}, session=fake_session(get=nvd_reply(EMPTY)))
    intel.lookup("nginx 1.25")
    intel.lookup("Apache 2.4.49")
    assert len(sleeps) == 1 and 5 < sleeps[0] <= 6


@pytest.mark.parametrize(("banner", "software"), [
    ("SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1", "OpenSSH 8.9"),
    ("Apache/2.4.41 (Ubuntu)", "Apache 2.4.41"),
    ("220 (vsFTPd 3.0.3)", "vsFTPd 3.0.3"),
    ("SimpleHTTP/0.6 Python/3.14.3", "SimpleHTTP 0.6"),
    ("Microsoft-IIS/10.0", "Microsoft-IIS 10.0"),
    ("MySQL 8.0.33", "MySQL 8.0.33"),
    ("# Server\r\nredis_version:7.0.11\r\n", "Redis 7.0.11"),
    ("HTTP/1.1 200 OK", None),
    ("nginx", None),
    ("", None),
])
def test_product_version(banner, software):
    assert probes.product_version(banner) == software
