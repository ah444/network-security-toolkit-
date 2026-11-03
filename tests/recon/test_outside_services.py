"""HS4 (RD5): the outside services, NVD, the AI (OpenRouter) and AbuseIPDB, are on by default and
each can be turned off; one without its key is off and says so; one that fails during a scan is
off for the rest of that scan, and the scan says why at its end. A port whose CVEs were not
looked up is "NOT CHECKED", never shown as clean. No test reaches the network."""

import requests

from toolkit.cli.ui import scan as screen
from toolkit.core.data.recon import lookup
from toolkit.core.data.recon.findings import IntelFinding
from toolkit.core.data.recon.scan_files import report_text
from toolkit.core.data.recon.scanner import PortResult
from toolkit.core.services.recon import intel as recon_intel
from toolkit.core.services.recon import scan as scan_service
from toolkit.core.services.recon.report import make_entry

KEYS = {"openrouter": "or-key", "abuseipdb": "ab-key", "nvd": "nvd-key"}


class Provider:
    """An IntelProvider that answers, or raises ``error`` every time; counts what it was asked."""

    def __init__(self, name, kind, error=None, finding=None):
        self.name, self.kind, self.error = name, kind, error
        self.finding = finding or IntelFinding(cves=(), source="offline")
        self.asked = []

    def lookup(self, subject):
        self.asked.append(subject)
        if self.error is not None:
            raise self.error
        return self.finding


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


# --- A service that fails is off for the rest of the scan --------------------------------

def test_a_service_that_fails_is_not_asked_again_in_this_scan():
    nvd = Provider("NVD", "cves", requests.ConnectionError("down"))
    intel = recon_intel.Intel({}, providers=[nvd])
    first = intel.lookup("OpenSSH 8.9")
    second = intel.lookup("nginx 1.25.3")
    assert nvd.asked == ["OpenSSH 8.9"], "asked once, then off"
    assert first["severity"] == second["severity"] == "NOT CHECKED"
    assert first["errors"] == ["NVD: no connection"]
    assert intel.unavailable() == {"NVD": "no connection"}


def test_the_next_scan_asks_again():
    nvd = Provider("NVD", "cves", requests.Timeout())
    recon_intel.Intel({}, providers=[nvd]).lookup("OpenSSH 8.9")
    recon_intel.Intel({}, providers=[nvd]).lookup("OpenSSH 8.9")
    assert nvd.asked == ["OpenSSH 8.9", "OpenSSH 8.9"], "one Intel per scan"


def test_one_failing_service_leaves_the_others_on():
    nvd = Provider("NVD", "cves")
    ai = Provider("AI (OpenRouter)", "explanation", _http_error(401))
    intel = recon_intel.Intel({}, providers=[nvd, ai])
    intel.lookup("OpenSSH 8.9")
    intel.lookup("nginx 1.25.3")
    assert nvd.asked == ["OpenSSH 8.9", "nginx 1.25.3"] and ai.asked == ["OpenSSH 8.9"]
    assert intel.unavailable() == {"AI (OpenRouter)": "the key was refused (HTTP 401)"}


def test_abuseipdb_off_after_a_failure_says_why():
    abuse = Provider("AbuseIPDB", "ip_reputation", _http_error(429))
    intel = recon_intel.Intel({}, providers=[abuse])
    assert intel.abuse_report("8.8.8.8") == (None, "AbuseIPDB: too many requests (HTTP 429)")
    assert intel.abuse_report("1.1.1.1") == (None, "AbuseIPDB: unavailable, too many requests "
                                                   "(HTTP 429)")
    assert abuse.asked == ["8.8.8.8"]


def test_the_reasons_are_short():
    assert lookup.describe(_http_error(401)) == "the key was refused (HTTP 401)"
    assert lookup.describe(_http_error(403)) == "the key was refused (HTTP 403)"
    assert lookup.describe(_http_error(429)) == "too many requests (HTTP 429)"
    assert lookup.describe(_http_error(503)) == "HTTP 503"
    assert lookup.describe(requests.Timeout()) == "timed out"
    assert lookup.describe(requests.ConnectionError()) == "no connection"


# --- NVD off ------------------------------------------------------------------------------

def test_nvd_off_asks_nothing_and_checks_nothing():
    intel = recon_intel.Intel(KEYS, nvd=False)  # nothing is sent: the counts stay 0
    result = intel.lookup("OpenSSH 8.9")
    assert result["severity"] == "NOT CHECKED" and result["cves"] == []
    assert intel.sent() == {"NVD": 0, "AI (OpenRouter)": 0, "AbuseIPDB": 0}
    assert not [p for p in intel.providers if p.kind == "cves"]


def test_nvd_answering_no_cve_is_still_no_known_cve():
    intel = recon_intel.Intel({}, providers=[Provider("NVD", "cves")])
    assert intel.lookup("OpenSSH 8.9")["severity"] == "NO KNOWN CVE"


# --- The plan: on by default, off with its flag or without its key -----------------------

def _request(**options):
    return scan_service.ScanRequest("8.8.8.8", ["8.8.8.8"], [22], ["tcp"], **options)


def test_the_plan_of_each_service():
    on = scan_service.online_plan(_request(use_ai=True, abuseipdb=True), KEYS)
    assert (on.nvd.startswith("on"), on.ai.startswith("on"), on.abuseipdb.startswith("on")) == (
        True, True, True)
    off = scan_service.online_plan(_request(nvd=False), KEYS)
    assert (off.nvd, off.ai, off.abuseipdb) == ("off: turned off (--no-nvd)",
                                                 "off: turned off (--no-ai)",
                                                 "off: turned off (--no-abuseipdb)")
    asked = scan_service.online_plan(_request(nvd=False, interactive=True), KEYS)
    assert (asked.nvd, asked.ai) == ("off: you said no", "off: you said no")


# --- A port whose CVEs were not looked up is never shown as clean -------------------------

def _entry(severity):
    found = {"cves": [], "cve_source": "", "max_cvss": None, "severity": severity,
             "severity_unconfirmed": False, "ai_explanation": "", "errors": []}
    return make_entry("10.0.0.5", PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1"),
                      "OpenSSH 8.9", found)


def test_not_checked_on_the_console_and_in_the_report(capsys):
    screen.print_entry(_entry("NOT CHECKED"))
    screen.print_entry(_entry("NO KNOWN CVE"))
    out = capsys.readouterr().out
    assert "CVEs (NVD): not checked\nSeverity (from CVSS): NOT CHECKED" in out
    assert "CVEs (NVD): none found\nSeverity (from CVSS): NO KNOWN CVE" in out
    text = report_text([_entry("NOT CHECKED")], "txt")
    assert "Severity (CVSS): NOT CHECKED\nCVEs (NVD): not checked\n" in text
