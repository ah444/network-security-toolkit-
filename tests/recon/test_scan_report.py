"""F5a: a saved scan's HTML report. It says what each CVE match means (by version: a possible
match to check; by name: unconfirmed), never shows unchecked CVEs as clean, marks the AI's text
as possibly wrong and ties each advice to its port. Every value is escaped, an NVD link is made
only from a real CVE id, and the file runs no script and loads nothing."""

import base64
import hashlib
import socket
from html.parser import HTMLParser

import pytest

from toolkit.core.data.recon import report_render
from toolkit.core.services.recon import scan_report
from toolkit.core.services.recon.saved_scans import NotAScan

from .test_saved_scans import A, entry, record, save

XSS = "<img src=x onerror=alert(1)>\"'"
NVD = "https://nvd.nist.gov/vuln/detail/"


class Page(HTMLParser):
    """The tags, their attributes and the visible text of a page."""

    def __init__(self, html):
        super().__init__()
        self.tags, self.parts, self.style = [], [], ""
        self._in_style = False
        self.feed(html)
        self.text = " ".join(" ".join(self.parts).split())

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self._in_style = tag == "style"

    def handle_endtag(self, tag):
        self._in_style = False

    def handle_data(self, data):
        if self._in_style:
            self.style += data
        else:
            self.parts.append(data)

    def links(self):
        return [attrs["href"] for tag, attrs in self.tags if tag == "a" and "href" in attrs]


def cve(cve_id, cvss, severity, match="version"):
    return {"id": cve_id, "cvss": cvss, "cvss_version": "3.1", "severity": severity,
            "match": match}


def found(port, severity, cves, **over):
    worst = max((c["cvss"] for c in cves if c["cvss"] is not None), default=None)
    return {**entry(port, severity), "cves": cves, "max_cvss": worst,
            "cve_source": "version match (CPE names)", **over}


def page(**over):
    return Page(report_render.report_html(scan_report.report_model(record(A, **over))))


def test_the_report_names_the_scan_and_what_was_looked_up():
    shown = page()
    assert f"Scan report {A}" in shown.text
    for words in ("Started 2026-11-01 10:00 UTC", "asked from this computer",
                  "10.0.0.4/31 (10.0.0.4, 10.0.0.5)", "22,80-82 (4 ports, TCP)", "built-in",
                  "NVD (CVEs, services.nvd.nist.gov): on",
                  "AI explanation (OpenRouter): off: no OPENROUTER_API_KEY",
                  "Requests sent online: NVD 1, AI (OpenRouter) 0, AbuseIPDB 0"):
        assert words in shown.text, words


def test_every_value_is_shown_as_text_never_as_markup():
    bad = {**found(22, "HIGH", [cve(XSS, 7.5, "HIGH")]), "target": "10.0.0.5", "service": XSS,
           "banner": XSS + "\x1b[31m", "software": XSS, "cve_source": XSS,
           "ai_explanation": XSS, "lookup_errors": [XSS], "os_hint": XSS, "cpe": [XSS]}
    shown = page(source=XSS, target=XSS, entries=[bad], engine="nmap",
                 nmap={"service_detection": "light", "timing": "normal", "scripts": XSS,
                       "os_detect": False},
                 online={"nvd": XSS, "ai": XSS, "abuseipdb": XSS},
                 hosts=[{"target": "10.0.0.5", "state": "scanned", "os": XSS, "message": ""},
                        {"target": "10.0.0.4", "state": "failed", "os": "", "message": XSS}],
                 reputation={"10.0.0.5": {"abuseConfidenceScore": XSS, "totalReports": XSS},
                             "10.0.0.4": {"error": XSS}},
                 summary={"targets": 2, "open_ports": 1, "failed_lookups": [XSS],
                          "sent": {XSS: 1}, "unavailable": {XSS: XSS}})
    names = {tag for tag, _ in shown.tags}
    assert not names & {"img", "script", "iframe", "object", "embed", "svg"}
    assert not [name for _, attrs in shown.tags for name in attrs if name.startswith("on")]
    assert shown.text.count(XSS) >= 15, "each value is shown, as text"
    assert all(link.startswith((NVD, "#port-")) for link in shown.links())


def test_control_characters_from_the_network_are_removed():
    shown = page(entries=[{**entry(22), "banner": "SSH-2.0\x1b[31m-x\x07"}])
    assert "SSH-2.0-x" in shown.text and "\x1b" not in shown.text and "\x07" not in shown.text


def test_each_cve_says_how_it_was_found_and_links_only_a_real_id():
    cves = [cve("CVE-2023-38408", 9.8, "CRITICAL"), cve("CVE-2016-20012", 5.3, "MEDIUM", "name"),
            cve("javascript:alert(1)", None, "UNKNOWN")]
    shown = page(entries=[found(22, "CRITICAL", cves)])
    assert shown.links().count(NVD + "CVE-2023-38408") == 1
    assert shown.links().count(NVD + "CVE-2016-20012") == 1
    assert not [link for link in shown.links() if "javascript" in link]
    assert "javascript:alert(1)" in shown.text, "a bad id is shown as text, not linked"
    for words in ("version match", "name match, unconfirmed",
                  "CVSS 9.8 (v3.1)", "Found in NVD during the scan, 2026-11-01 10:00 UTC",
                  "Update OpenSSH 8.9 to a version that fixes CVE-2023-38408",
                  "Check the exact version of OpenSSH 8.9 first: CVE-2016-20012 was found by "
                  "its name only"):
        assert words in shown.text, words


def test_a_possible_match_is_never_called_a_confirmed_vulnerability():
    shown = page(entries=[found(22, "HIGH", [cve("CVE-2023-38408", 7.5, "HIGH")])])
    assert "a possible match to check, not a confirmed vulnerability" in shown.text


def test_cves_not_looked_up_are_never_shown_as_clean():
    shown = page(entries=[{**entry(22, "NOT CHECKED")}])
    assert "not a clean result" in shown.text and "NOT CHECKED" in shown.text
    assert "NO KNOWN CVE" not in shown.text and "NVD listed no CVE" not in shown.text


def test_the_advice_without_a_cve():
    shown = page(entries=[entry(22), {**entry(5040, "UNKNOWN"), "software": ""}])
    assert ("NVD listed no CVE for OpenSSH 8.9 on 2026-11-01. Keep it updated, and close the "
            "port if it isn't needed.") in shown.text
    assert ("The software on this port wasn't recognised, so no CVE was looked up. Check what "
            "runs on 5040/tcp and whether it needs to be open.") in shown.text


def test_the_ai_text_is_marked_as_possibly_wrong():
    shown = page(entries=[{**entry(22), "ai_explanation": "Patch it today."}])
    assert "AI explanation (may be wrong): Patch it today." in shown.text


def test_the_worst_ports_are_listed_first():
    shown = page(entries=[found(80, "LOW", [cve("CVE-2020-0001", 2.0, "LOW")]),
                          found(22, "CRITICAL", [cve("CVE-2023-38408", 9.8, "CRITICAL")]),
                          found(443, "HIGH", [cve("CVE-2021-0002", 7.5, "HIGH")])])
    inside = [link for link in shown.links() if link.startswith("#port-")]
    assert inside[:2] == ["#port-2", "#port-3"], "first CRITICAL, then HIGH; LOW is not listed"
    assert inside[2:] == ["#port-1", "#port-2", "#port-3"], "then the table, in scan order"
    ids = [attrs["id"] for tag, attrs in shown.tags if tag == "section" and "id" in attrs]
    assert ids == ["port-1", "port-2", "port-3"]
    assert "Ports: 3 open; CRITICAL 1, HIGH 1, LOW 1" in shown.text


def test_hosts_skipped_or_failed_and_what_the_report_cant_say():
    shown = page(hosts=[{"target": "10.0.0.4", "state": "skipped", "os": "", "message": ""},
                        {"target": "10.0.0.5", "state": "failed", "os": "",
                         "message": "nmap stopped"}], entries=[])
    for words in ("10.0.0.4 skipped: it didn't answer ping",
                  "10.0.0.5 the engine couldn't scan it (nmap stopped)",
                  "No open port was found.",
                  "Only ports 22,80-82 (TCP) were checked",
                  "1 host didn't answer ping and was skipped: it was not scanned.",
                  "It shows the network as it was on 2026-11-01."):
        assert words in shown.text, words


def test_the_file_runs_no_script_and_loads_nothing():
    html = report_render.report_html(scan_report.report_model(record(A)))
    shown = Page(html)
    names = [tag for tag, _ in shown.tags]
    assert "script" not in names and "link" not in names
    assert not [attrs for _, attrs in shown.tags if "src" in attrs]
    policy = next(attrs["content"] for tag, attrs in shown.tags
                  if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy")
    digest = base64.b64encode(hashlib.sha256(shown.style.encode("utf-8")).digest()).decode()
    assert policy == (f"default-src 'none'; style-src 'sha256-{digest}'; base-uri 'none'; "
                      "form-action 'none'")
    assert shown.style == report_render.stylesheet(), "the CSS is put in as it is"


def test_the_template_never_turns_escaping_off():
    template = report_render.TEMPLATE.read_text(encoding="utf-8")
    for unsafe in ("|safe", "| safe", "{% autoescape", "Markup", "{% include", "{% import",
                   "{% from", "{% extends"):
        assert unsafe not in template, unsafe


def test_the_page_prints_on_a4_and_follows_the_dark_theme_on_screen():
    css = report_render.stylesheet()
    assert "@media print" in css and "@page" in css and "break-inside: avoid" in css
    assert "prefers-color-scheme: dark" in css


def test_nothing_is_sent_while_a_report_is_made(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("the report went online")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    save(A, entries=[found(22, "HIGH", [cve("CVE-2023-38408", 7.5, "HIGH")])])
    assert f"Scan report {A}" in Page(scan_report.saved_report_html(A)).text


def test_a_saved_scan_by_its_id():
    save(A)
    assert A in scan_report.saved_report_html(A)
    with pytest.raises(NotAScan):
        scan_report.saved_report_html("../x")
    with pytest.raises(FileNotFoundError):
        scan_report.saved_report_html("20261101T100000Z-0000000f")


def test_a_record_kept_from_before_f6b_has_a_report_too():
    old = record(A)
    del old["choices"], old["nmap"]
    shown = Page(report_render.report_html(scan_report.report_model(old)))
    assert f"Scan report {A}" in shown.text
