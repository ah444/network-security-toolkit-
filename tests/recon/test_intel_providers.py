"""The IntelProvider port: NVD, OpenRouter and AbuseIPDB behind one shape (step 1.1c)."""

import requests

from toolkit.data.recon.findings import IntelFinding
from toolkit.ports import IntelProvider
from toolkit.recon import intel as recon_intel
from toolkit.data.recon import abuseipdb, openrouter as llm, nvd


class OfflineCves:
    """An offline CVE source: what the port makes possible without touching the scan code."""

    name = "Offline"
    kind = "cves"

    def __init__(self, cves):
        self.cves = cves
        self.asked = []

    def lookup(self, subject):
        self.asked.append(subject)
        return IntelFinding(cves=tuple(self.cves.get(subject, ())))


class NoNetwork:
    def get(self, *args, **kwargs):
        raise AssertionError("no network call expected")

    post = get


def test_built_in_providers_fulfil_the_port():
    session = NoNetwork()
    providers = (nvd.NvdProvider(session), llm.OpenRouterProvider(session, "k", "m"),
                 abuseipdb.AbuseIpdbProvider(session, "k"))
    assert all(isinstance(provider, IntelProvider) for provider in providers)
    assert [(p.name, p.kind) for p in providers] == \
        [("NVD", "cves"), ("AI", "explanation"), ("AbuseIPDB", "ip_reputation")]


def test_intel_builds_providers_from_keys_and_consent():
    def names(intel):
        return [provider.name for provider in intel.providers]

    session = NoNetwork()
    assert names(recon_intel.Intel({}, session=session)) == ["NVD"]
    assert names(recon_intel.Intel({"openrouter": "k"}, session=session)) == ["NVD"]  # no consent
    assert names(recon_intel.Intel({"openrouter": "k", "abuseipdb": "k"}, use_ai=True,
                                   session=session)) == ["NVD", "AI", "AbuseIPDB"]


def test_offline_provider_swapped_in():
    offline = OfflineCves({"Apache 2.4.49": [
        {"id": "CVE-2021-41773", "cvss": 7.5, "cvss_version": "3.1", "severity": "HIGH"},
        {"id": "CVE-2021-42013", "cvss": 9.8, "cvss_version": "3.1", "severity": "CRITICAL"}]})
    intel = recon_intel.Intel({}, session=NoNetwork(), providers=[offline])

    found = intel.lookup("Apache 2.4.49")
    assert [c["id"] for c in found["cves"]] == ["CVE-2021-42013", "CVE-2021-41773"]  # worst first
    assert found["max_cvss"] == 9.8 and found["severity"] == "CRITICAL"
    assert found["errors"] == [] and found["ai_explanation"] == ""
    assert intel.lookup("nginx 1.25")["severity"] == "NO KNOWN CVE"
    assert offline.asked == ["Apache 2.4.49", "nginx 1.25"]


def test_failed_provider_is_named_and_scan_goes_on():
    class Broken:
        name = "Mirror"
        kind = "cves"

        def lookup(self, subject):
            raise requests.ConnectionError("down")

    found = recon_intel.Intel({}, session=NoNetwork(), providers=[Broken()]).lookup("x 1.0")
    assert found["errors"] == ["Mirror: no connection"] and found["severity"] == "UNKNOWN"


def test_ip_reputation_through_the_port():
    class Reputation:
        name = "AbuseIPDB"
        kind = "ip_reputation"

        def lookup(self, subject):
            return IntelFinding(data={"ipAddress": subject, "abuseConfidenceScore": 0})

    intel = recon_intel.Intel({}, session=NoNetwork(), providers=[Reputation()])
    assert intel.abuse_report("8.8.8.8") == ({"ipAddress": "8.8.8.8",
                                              "abuseConfidenceScore": 0}, None)
    assert recon_intel.Intel({}, session=NoNetwork()).abuse_report("8.8.8.8") == \
        (None, "AbuseIPDB: no API key")
