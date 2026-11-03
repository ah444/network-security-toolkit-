"""v5.2 step 2.4e: the port scanner's page (/scan) - served like the trainer, linked from
every page, its form (the authorization asked again for every target), the running scan's
log, the results (CVE IDs as plain text, never links) and the CVSS colours readable in both
themes. The scan contract itself is the API's (tests/web/test_scan_routes.py); every module
is also under the XSS rules (tests/security/test_xss.py)."""

import re

import pytest

from web_contrast import colour, contrast, stylesheet, themes
from web_js import SCAN_TEMPLATE, TEMPLATES, js_sources

SCAN_LINK = '<a href="/scan" class="nav-link"'


@pytest.fixture(scope="module")
def page():
    return SCAN_TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def modules():
    sources = js_sources()
    return {name: sources[name] for name in ("scan.js", "scanrun.js", "scanresults.js")}


def test_the_page_is_served_like_the_trainer(client):
    response = client.get("/scan")
    assert response.status_code == 200 and response.mimetype == "text/html"
    assert response.headers["Cache-Control"] == "no-store", "it holds the session's token"
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    html = response.get_data(as_text=True)
    assert re.search(r'<meta name="csrf-token" content="[^"]+">', html)
    assert "<title>Port Scanner - Intrusion Detection System</title>" in html


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_every_page_links_to_the_scanner(template):
    html = template.read_text(encoding="utf-8")
    assert html.count(SCAN_LINK) == 1
    assert (SCAN_LINK + ' aria-current="page">' in html) == (template == SCAN_TEMPLATE)


def test_the_home_page_has_the_scanner(client):
    html = client.get("/").get_data(as_text=True)
    assert "<h2>Port Scanner</h2>" in html and '<a href="/scan" class="custom-btn">' in html
    assert "Only scan systems you own or have written permission to test." in html


def test_the_form_asks_for_the_authorization(page, modules):
    assert re.search(r'<input type="checkbox" id="scanAuthorized" required>', page)
    scan = modules["scan.js"]
    changed = re.search(r"function targetChanged\(\) \{.*?\n\}", scan, re.S).group(0)
    assert 'byId("scanAuthorized").checked = false' in changed, "asked again for a new target"
    assert '"I own " + target + " or have written permission to scan it."' in changed
    assert 'byId("scanTarget").addEventListener("input", targetChanged)' in scan
    assert 'authorized: byId("scanAuthorized").checked' in scan


def test_nmaps_choices_are_sent_only_with_nmap(modules):
    request = re.search(r"function request\(\) \{.*?\n\}", modules["scan.js"], re.S).group(0)
    assert 'if (body.engine === "nmap")' in request
    for field in ("service_detection", "timing", "scripts", "os_detect"):
        assert request.index(field + ":") > request.index('if (body.engine === "nmap")'), field


def test_the_log_is_asked_for_from_its_last_line(page, modules):
    assert 'getJSON(jobUrl, { since: logSize })' in modules["scanrun.js"]
    assert ('<ol id="scanLog" class="scan-log" tabindex="0" role="log" aria-live="off" '
            'aria-labelledby="scanLogTitle">') in page, "reachable by keyboard, not read aloud"
    assert re.search(r'<div id="scanStatus" class="message" role="status"', page)


def test_cve_ids_are_text_never_links(modules):
    results = modules["scanresults.js"]
    assert 'el("a"' not in results and "href" not in results and "nvd.nist.gov" not in results
    assert 'el("span", { class: "mono" }, cve.id)' in results


def test_every_cve_shows_how_it_was_matched(modules):
    results = modules["scanresults.js"]
    assert '"version match"' in results and '"name match, unconfirmed"' in results
    assert results.count("severityText(entry)") == 2, "the summary and the details"


def test_the_downloads_are_the_report_route(page):
    links = re.findall(r'<a class="custom-btn" href="([^"]+)" download>(\w+)</a>', page)
    assert links == [(f"/api/v1/scan/report?format={fmt.lower()}", fmt)
                     for fmt in ("TXT", "JSON", "CSV")]


def test_every_cvss_level_is_readable_in_both_themes(modules):
    css = stylesheet()
    levels = re.findall(r"\.cvss-(\w+) \{ --cvss-colour: var\((--[\w-]+)\); \}", css)
    built = re.search(r"const SEVERITY_CLASS = \{(.*?)\};", modules["scanresults.js"], re.S)
    assert sorted(name for name, _ in levels) == sorted(re.findall(r':\s*"(\w+)"', built.group(1)))
    light, dark = themes()
    for theme in (light, dark):
        for _, token in levels:
            for surface in ("--surface", "--surface-inset"):
                ratio = contrast(colour(token, theme, light), colour(surface, theme, light))
                assert ratio >= 4.5, (token, surface, round(ratio, 2))
