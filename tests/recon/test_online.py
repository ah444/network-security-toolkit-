"""Step A5: every scan says what it may send to each online service, or why it sends nothing,
and at the end how many requests went to each (counted by the scan's own HTTP session)."""

import threading

import pytest
import requests

from toolkit.cli.ui import scan as screen
from toolkit.core.data.recon import lookup
from toolkit.core.services.recon import intel as recon_intel
from toolkit.core.services.recon import scan as scan_service

KEYS = {"openrouter": "or-key", "abuseipdb": "ab-key", "nvd": "nvd-key"}
NO_KEYS = {"openrouter": "", "abuseipdb": "", "nvd": ""}


def _request(targets=("10.0.0.5",), interactive=False, use_ai=False, abuseipdb=False):
    return scan_service.ScanRequest("x", list(targets), [22], ["tcp"], use_ai=use_ai,
                                    abuseipdb=abuseipdb, interactive=interactive)


@pytest.mark.parametrize(("request_args", "keys", "ai", "abuse"), [
    ({}, NO_KEYS, "off: no OPENROUTER_API_KEY",
     "off: no public address (a private one is never sent)"),
    ({"targets": ["8.8.8.8"]}, NO_KEYS, "off: no OPENROUTER_API_KEY", "off: no ABUSEIPDB_API_KEY"),
    ({"targets": ["8.8.8.8"]}, KEYS, "off: turned off (--no-ai)",
     "off: turned off (--no-abuseipdb)"),
    ({"targets": ["8.8.8.8"], "use_ai": True, "abuseipdb": True}, KEYS,
     "on, the software name and version only", "on, the public addresses only"),
    ({"targets": ["8.8.8.8"], "interactive": True}, KEYS, "off: you said no",
     "asked before each public address"),
    ({"targets": ["192.168.1.10", "8.8.8.8"], "interactive": True}, NO_KEYS,
     "off: no OPENROUTER_API_KEY", "off: no ABUSEIPDB_API_KEY"),
    ({"targets": ["192.168.1.10"], "abuseipdb": True}, KEYS, "off: turned off (--no-ai)",
     "off: no public address (a private one is never sent)"),
])
def test_the_plan_says_why_not(request_args, keys, ai, abuse):
    plan = scan_service.online_plan(_request(**request_args), keys)
    assert (plan.ai, plan.abuseipdb) == (ai, abuse)


def test_nvd_is_always_on_and_says_its_pace():
    assert scan_service.online_plan(_request(), NO_KEYS).nvd == (
        "on, the software name and version only; no NVD_API_KEY, so one request every 6 s")
    assert scan_service.online_plan(_request(), KEYS).nvd == (
        "on, the software name and version only")


@pytest.fixture
def no_network(monkeypatch):
    """requests.Session.request answers without the network; the URLs it was asked are kept."""
    asked = []

    class Answer:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"totalResults": 0, "vulnerabilities": []}

    def answer(self, method, url, *args, **kwargs):
        asked.append(url)
        if "fails" in url:
            raise requests.ConnectionError("no connection")
        return Answer()

    monkeypatch.setattr(requests.Session, "request", answer)
    return asked


def test_the_session_counts_by_host(no_network):
    session = lookup.Session()
    session.get("https://services.nvd.nist.gov/rest/json/cves/2.0", params={"x": 1})
    session.get("https://services.nvd.nist.gov/rest/json/cves/2.0")
    session.post("https://openrouter.ai/api/v1/chat/completions", json={})
    with pytest.raises(requests.ConnectionError):
        session.get("https://fails.example/")
    assert session.sent == {"services.nvd.nist.gov": 2, "openrouter.ai": 1, "fails.example": 1}
    assert len(no_network) == 4, "a failed request was still sent, so it is counted"


def test_the_count_is_exact_across_threads(no_network):
    session = lookup.Session()

    def ask():
        for _ in range(50):
            session.get("https://api.abuseipdb.com/api/v2/check")

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert session.sent == {"api.abuseipdb.com": 400}


def test_intel_names_each_service(no_network, no_nvd_wait):
    intel = recon_intel.Intel({})
    assert isinstance(intel.session, lookup.Session)
    assert intel.sent() == {"NVD": 0, "AI (OpenRouter)": 0, "AbuseIPDB": 0}
    intel.lookup("nginx 1.25.3")  # the version match, none, then the keyword search
    intel.session.get("https://elsewhere.example/")
    assert intel.sent() == {"NVD": 2, "AI (OpenRouter)": 0, "AbuseIPDB": 0,
                            "elsewhere.example": 1}
    assert recon_intel.Intel({}, session=object()).sent() == {
        "NVD": 0, "AI (OpenRouter)": 0, "AbuseIPDB": 0}, "a session that doesn't count"


def test_the_console_lines(capsys):
    view = screen.ConsoleScan()
    view.lookups_notice(scan_service.OnlinePlan("on, x", "off: no OPENROUTER_API_KEY", "off: y"))
    view.summary(1, 2, [], 3.0, {"NVD": 4, "AI (OpenRouter)": 0, "AbuseIPDB": 0}, {})
    out = capsys.readouterr().out
    assert ("Online in this scan:\n  NVD (CVEs, services.nvd.nist.gov): on, x\n"
            "  AI explanation (OpenRouter): off: no OPENROUTER_API_KEY\n"
            "  IP reputation (AbuseIPDB): off: y\n") in out
    assert "Requests sent online: NVD 4, AI (OpenRouter) 0, AbuseIPDB 0\n" in out
