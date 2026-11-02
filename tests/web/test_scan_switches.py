"""HS4b (RD5): the web scan's outside services, NVD, the AI (OpenRouter) and AbuseIPDB, are on
by default, each with its switch on the page; one without its key is off and the answer says
why (never a refusal); one that fails during the scan is named in the summary with why, and
a port whose CVEs were not looked up says "not checked". The network is faked
(``web_scan.fake_network``): nothing leaves this computer."""

import re
import time

import pytest
import requests

from toolkit.core.data.settings import ScanConfig
from toolkit.core.services.recon import scan as core_scan
from toolkit.core.services.recon.intel import Intel
from toolkit.web.services import jobs, scan
from toolkit.web.services.workspace import Workspace
from web_js import SCAN_TEMPLATE, js_sources
from web_scan import fake_network

LIMITS = ScanConfig()


@pytest.fixture
def net(monkeypatch):
    return fake_network(monkeypatch)


def _scan(target="8.8.8.8", **switches):
    job = scan.start_scan(Workspace(), scan.ScanChoices(target=target, ports="22,8080",
                                                        authorized=True, **switches),
                          jobs.JobRunner(jobs.SCAN), LIMITS)
    deadline = time.monotonic() + 30
    while job.state not in jobs.FINISHED:
        assert time.monotonic() < deadline, "the scan job didn't end"
        time.sleep(0.01)
    assert job.state == "done", job.error
    return job.result


def test_every_service_is_on_by_default():
    choices = scan.ScanChoices()
    assert (choices.nvd, choices.ai, choices.abuseipdb) == (True, True, True)


def test_without_its_key_a_service_is_off_and_says_why(net):
    answer = _scan()  # no key at all: still a scan, never a refusal
    assert answer["online"]["ai"] == "off: no OPENROUTER_API_KEY"
    assert answer["online"]["abuseipdb"] == "off: no ABUSEIPDB_API_KEY"
    assert answer["online"]["nvd"].startswith("on") and net.sent["reputation"] == []


def test_with_its_key_a_service_is_on(net):
    net.keys["abuseipdb"] = "ab-test-key-123456"
    answer = _scan()
    assert answer["online"]["abuseipdb"] == "on, the public addresses only"
    assert net.sent["reputation"] == ["8.8.8.8"]


def test_a_switch_turned_off_on_the_page(net):
    net.keys.update(openrouter="or-test-key-123456", abuseipdb="ab-test-key-123456")
    answer = _scan(nvd=False, ai=False, abuseipdb=False)
    assert answer["online"] == {"nvd": "off: turned off on the page",
                                "ai": "off: turned off on the page",
                                "abuseipdb": "off: turned off on the page"}
    assert net.sent["cves"] == net.sent["reputation"] == [], "nothing sent"
    assert [entry["severity"] for entry in answer["entries"]] == ["NOT CHECKED"] * 2
    assert answer["summary"]["sent"] == {"NVD": 0, "AI (OpenRouter)": 0, "AbuseIPDB": 0}


def test_a_service_that_fails_is_named_in_the_summary(net, monkeypatch):
    class Down:
        name, kind = "NVD", "cves"

        def lookup(self, subject):
            net.sent["cves"].append(subject)
            raise requests.ConnectionError("down")

    monkeypatch.setattr(core_scan, "Intel", lambda keys, use_ai=False, stop=None, nvd=True:
                        Intel(keys, use_ai, stop=stop, providers=[Down()]))
    answer = _scan("10.0.0.5")
    assert net.sent["cves"] == ["OpenSSH 8.9"], "asked once, then not again in this scan"
    assert answer["summary"]["unavailable"] == {"NVD": "no connection"}
    assert [entry["severity"] for entry in answer["entries"]] == ["NOT CHECKED"] * 2


# --- The page --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def page():
    return SCAN_TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def modules():
    return js_sources()


def test_each_service_has_a_switch_on_by_default(page, modules):
    for box in ("scanNvd", "scanAbuse", "scanAi"):
        assert re.search(rf'id="{box}" checked aria-describedby="{box}Note"', page), box
    request = modules["scan.js"]
    for field, box in (("nvd", "scanNvd"), ("abuseipdb", "scanAbuse"), ("ai", "scanAi")):
        assert f'{field}: byId("{box}").checked' in request


def test_without_its_key_the_switch_is_off_and_says_how_to_fix_it(modules):
    script = modules["scan.js"]
    assert "byId(box).disabled = !options.keys[key];" in script
    assert "byId(box).checked = options.keys[key];" in script
    assert '"No API key: add " + variable + " to .env, then refresh this page."' in script


def test_the_results_say_unavailable_and_not_checked(modules):
    results = modules["scanresults.js"]
    assert 'name + ": unavailable, " + why' in results
    assert 'statTile("Unavailable", unavailableText(totals.unavailable))' in results
    assert 'notChecked(entry) ? "not checked" : "none found"' in results
    assert 'SEVERITY_CLASS[severity] || "unknown"' in results, "NOT CHECKED: UNKNOWN's colour"
