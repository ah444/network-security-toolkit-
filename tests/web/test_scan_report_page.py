"""F5b: a saved scan's report on the web. ``/scans/<id>/report`` opens it in a page of its own
(its CSS from ``/scans/report.css``, the app's policy, no script), to print or save as PDF;
``/api/v1/scans/<id>/report?format=html`` downloads the same report as one file that stands
alone. Saved Scans links both, and the scan page links the report of the scan it saved."""

import pytest

from recon.test_scan_report import XSS, Page
from test_scans_page import A, _save
from toolkit.core.data.recon import report_render
from web_js import JS_DIR, SCAN_TEMPLATE, SCANS_TEMPLATE

MISSING = "20261101T100000Z-0000000f"


def test_the_report_opens_in_a_page_of_its_own(client):
    _save(A, source=XSS)
    answer = client.get(f"/scans/{A}/report")
    assert answer.status_code == 200 and answer.mimetype == "text/html"
    shown = Page(answer.get_data(as_text=True))
    assert f"Scan report {A}" in shown.text and XSS in shown.text
    names = [tag for tag, _ in shown.tags]
    assert "script" not in names and "style" not in names and "img" not in names
    assert ("link", {"rel": "stylesheet", "href": "/scans/report.css"}) in shown.tags
    assert not [attrs for tag, attrs in shown.tags if tag == "meta"
                and attrs.get("http-equiv") == "Content-Security-Policy"], "the app's policy"
    assert "script-src 'self'" in answer.headers["Content-Security-Policy"]


def test_its_stylesheet(client):
    answer = client.get("/scans/report.css")
    assert answer.status_code == 200 and answer.mimetype == "text/css"
    assert answer.get_data(as_text=True) == report_render.stylesheet()


def test_the_report_downloads_as_one_file(client):
    _save(A)
    answer = client.get(f"/api/v1/scans/{A}/report", query_string={"format": "html"})
    assert answer.status_code == 200 and answer.mimetype == "text/html"
    assert answer.headers["Content-Disposition"] == f'attachment; filename="scan-{A}.html"'
    shown = Page(answer.get_data(as_text=True))
    assert shown.style == report_render.stylesheet() and "link" not in [t for t, _ in shown.tags]


@pytest.mark.parametrize("url", ["/scans/{}/report", "/api/v1/scans/{}/report?format=html"])
def test_a_bad_or_missing_scan(client, url):
    assert client.get(url.format("..%2Fx")).status_code in (400, 404)
    assert client.get(url.format("not-an-id")).status_code == 400
    assert client.get(url.format(MISSING)).status_code == 404


def test_saved_scans_links_the_report_and_its_file():
    page = SCANS_TEMPLATE.read_text(encoding="utf-8")
    script = (JS_DIR / "scans.js").read_text(encoding="utf-8")
    assert 'id="reportOpen"' in page and 'target="_blank"' in page and 'id="reportHtml"' in page
    assert '["reportHtml", "html"]' in script
    assert 'byId("reportOpen").href = "/scans/" + encodeURIComponent(scanId) + "/report"' in script


def test_the_scan_page_links_the_report_of_the_scan_it_saved():
    script = (JS_DIR / "scan.js").read_text(encoding="utf-8")
    assert '"/scans/" + encodeURIComponent(saved.scan_id) + "/report"' in script
    assert "Open its report" in script
    assert SCAN_TEMPLATE.is_file()
