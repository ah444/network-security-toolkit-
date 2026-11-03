"""F6b: the Saved Scans page, /scans, and its routes. It reads the scan history
(``instance/scan_history/``) through /api/v1/scans only: the list, one scan, its report; its one
change is a scan's delete, a POST with the session's token. A bad scan id is a 400 before any
path is made, a missing one a 404. The scan page says where its scan was saved, and Scan again
fills its form from a saved scan with the authorization unticked."""

import json
import re

from toolkit.core.data.recon import scan_records
from toolkit.core.services.recon import history
from web_js import JS_DIR, REPO_ROOT, SCAN_TEMPLATE, TEMPLATES, js_sources

A = "20261101T100000Z-0000000a"
B = "20261102T100000Z-0000000b"
XSS = "<img src=x onerror=alert(1)>"


def _record(scan_id, **over):
    made = {
        "kind": "scan", "schema": history.SCHEMA, "scan_id": scan_id,
        "started": "2026-11-01T10:00:00Z", "seconds": 1.5, "source": "this computer",
        "target": "127.0.0.1", "addresses": ["127.0.0.1"], "ports": "22", "protocols": ["tcp"],
        "engine": "builtin", "nmap": None,
        "choices": {"scan_unresponsive": False, "nvd": True, "ai": False, "abuseipdb": False},
        "online": {"nvd": "on", "ai": "off", "abuseipdb": "off"},
        "hosts": [{"target": "127.0.0.1", "state": "scanned", "os": "", "message": ""}],
        "reputation": {},
        "entries": [{"target": "127.0.0.1", "port": 22, "protocol": "tcp", "service": "ssh",
                     "banner": XSS, "software": "", "cves": [], "cve_source": "",
                     "max_cvss": None, "severity": "UNKNOWN", "severity_unconfirmed": False,
                     "ai_explanation": "", "lookup_errors": [], "os_hint": "",
                     "engine": "builtin", "cpe": []}],
        "summary": {"targets": 1, "open_ports": 1, "failed_lookups": [], "sent": {},
                    "unavailable": {}},
        "exit_code": 0}
    made.update(over)
    return made


def _save(scan_id, **over):
    scan_records.write_scan(scan_id, _record(scan_id, **over))


def test_the_list_and_one_scan(client):
    _save(A)
    _save(B)
    answer = client.get("/api/v1/scans")
    assert answer.status_code == 200
    assert [s["scan_id"] for s in answer.get_json()["scans"]] == [B, A]
    older = client.get("/api/v1/scans", query_string={"before": B}).get_json()
    assert [s["scan_id"] for s in older["scans"]] == [A]
    one = client.get(f"/api/v1/scans/{A}")
    assert one.status_code == 200 and one.get_json()["entries"][0]["banner"] == XSS
    assert one.get_json()["again"]["target"] == "127.0.0.1"


def test_a_bad_id_is_400_and_a_missing_one_404(client):
    for url in ("/api/v1/scans/not-an-id", "/api/v1/scans/not-an-id/report",
                "/api/v1/scans?before=..%2Fx"):
        assert client.get(url).status_code == 400, url
    assert client.post("/api/v1/scans/not-an-id/delete", json={}).status_code == 400
    missing = client.get(f"/api/v1/scans/{A}")
    assert missing.status_code == 404 and missing.get_json()["error"]["code"] == "NOT_FOUND"
    assert client.post(f"/api/v1/scans/{A}/delete", json={}).status_code == 404
    (scan_records.history_dir()).mkdir(parents=True, exist_ok=True)
    (scan_records.history_dir() / f"{B}.json").write_text("{}", encoding="utf-8")
    assert client.get(f"/api/v1/scans/{B}").status_code == 400  # not a scan record


def test_its_report_downloads_as_a_file(client):
    _save(A)
    for fmt, media in (("txt", "text/plain"), ("json", "application/json"), ("csv", "text/csv")):
        answer = client.get(f"/api/v1/scans/{A}/report", query_string={"format": fmt})
        assert answer.status_code == 200
        assert answer.content_type == f"{media}; charset=utf-8"
        assert answer.headers["Content-Disposition"] == f'attachment; filename="scan-{A}.{fmt}"'
    assert json.loads(client.get(f"/api/v1/scans/{A}/report?format=json").get_data(
        as_text=True))[0]["port"] == 22
    assert client.get(f"/api/v1/scans/{A}/report?format=pdf").status_code == 400


def test_delete_removes_the_record(client):
    _save(A)
    _save(B)
    answer = client.post(f"/api/v1/scans/{A}/delete", json={})
    assert answer.status_code == 200 and answer.get_json() == {"scan_id": A,
                                                               "removed": f"{A}.json"}
    assert [p.name for p in scan_records.list_scans()] == [f"{B}.json"]


def test_a_record_that_cant_be_removed_is_a_409(client, monkeypatch):
    _save(A)

    def busy(path):
        raise PermissionError("in use")

    monkeypatch.setattr(scan_records, "remove_file", busy)
    answer = client.post(f"/api/v1/scans/{A}/delete", json={})
    assert answer.status_code == 409 and answer.get_json()["error"]["code"] == "NOT_REMOVED"
    assert f"{A}.json" in answer.get_json()["error"]["message"]
    assert scan_records.list_scans()


def test_the_page_is_served_and_every_page_links_to_it(client):
    html = client.get("/scans").get_data(as_text=True)
    assert '<a href="/scans" class="nav-link" aria-current="page">Saved Scans</a>' in html
    assert "js/scans.js" in html
    for url in ("/", "/dynamic_ids", "/scan", "/runs", "/scans"):
        assert 'href="/scans" class="nav-link"' in client.get(url).get_data(as_text=True), url
    for template in TEMPLATES:
        assert 'href="/scans"' in template.read_text(encoding="utf-8"), template.name


def test_the_page_reads_the_history_and_only_deletes():
    code = js_sources()["scans.js"]
    assert "FormData" not in code
    assert re.findall(r"send\(([^,]+), \"(\w+)\"", code) == [
        ('"/api/v1/scans/" + encodeURIComponent(scanId) + "/delete"', "POST")]
    assert set(re.findall(r'"(/api/v1/[^"]*)"', code)) == {"/api/v1/scans", "/api/v1/scans/"}


def test_a_scan_in_the_address_is_checked_as_the_server_checks_it():
    server = history.SCAN_ID.pattern
    for name in ("scans.js", "scan.js"):
        page = re.search(r"const SCAN_ID = /\^(.+)\$/;", js_sources()[name]).group(1)
        assert page == server, name


def test_the_scan_page_says_where_its_scan_was_saved():
    html = SCAN_TEMPLATE.read_text(encoding="utf-8")
    assert "nothing is saved on the server" not in html
    assert 'id="scanSaved"' in html
    code = js_sources()["scan.js"]
    assert '"/scans#" + answer.saved.scan_id' in code
    assert "Not saved in the scan history: " in code


def test_scan_again_fills_the_form_and_asks_for_the_authorization_again():
    code = js_sources()["scan.js"]
    again = code[code.index("async function scanAgain"):]
    again = again[:again.index("\n}\n")]
    assert 'getJSON("/api/v1/scans/" + encodeURIComponent(scanId))' in again
    assert "targetChanged()" in again, "unticks the authorization and names the new target"
    assert again.index('byId("scanTarget").value') < again.index("targetChanged()")
    assert "scanAuthorized\").checked = true" not in again
    assert (JS_DIR / "scans.js").is_file() and REPO_ROOT.is_dir()
