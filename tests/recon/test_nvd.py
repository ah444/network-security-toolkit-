"""B10, step A4: CVEs from the NVD CVE API 2.0 (mocked with real NVD replies): by version for
the products NVD names (``CPE_PRODUCTS``), else by keyword; each answer says which."""

import json
import threading

import pytest

from toolkit.core.services.recon import intel as recon_intel
from toolkit.core.data.recon import nvd

# Real replies of the NVD CVE API 2.0 (tests/fixtures/README.md)
OPENSSH = "nvd_openssh_8.9.json"  # keywordSearch=OpenSSH 8.9, 2026-10-20
OPENSSH_VERSION = "nvd_openssh_8.9_version.json"  # virtualMatchString, 2026-10-25
EMPTY = "nvd_no_results.json"
OPENSSH_CPE = "cpe:2.3:a:openbsd:openssh:8.9"


@pytest.fixture
def nvd_reply(fixtures_dir):
    return lambda name: json.loads((fixtures_dir / name).read_text(encoding="utf-8"))


class ByQuery:
    """A session that answers each request from ``replies``, keyed by its CPE name (and start)
    or its keyword; records the params."""

    def __init__(self, replies):
        self.replies = replies
        self.params = []

    def get(self, url, params, headers, timeout):
        self.params.append(dict(params))
        key = params.get("virtualMatchString", params.get("keywordSearch"))
        reply = self.replies.get((key, params.get("startIndex")), self.replies.get(key))
        return Reply(reply if reply is not None else {"vulnerabilities": []})


class Reply:
    """An HTTP 200 answer with a JSON body."""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _cve(cve_id, score):
    return {"cve": {"id": cve_id, "metrics": {"cvssMetricV31": [
        {"type": "Primary", "cvssData": {"version": "3.1", "baseScore": score}}]}}}


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


@pytest.mark.parametrize(("software", "names"), [
    ("OpenSSH 8.9", [OPENSSH_CPE]),
    ("Apache 2.4.49", ["cpe:2.3:a:apache:http_server:2.4.49"]),
    ("nginx 1.18.0", ["cpe:2.3:a:f5:nginx:1.18.0"]),
    ("MySQL 5.0.51", ["cpe:2.3:a:oracle:mysql:5.0.51", "cpe:2.3:a:mysql:mysql:5.0.51"]),
    ("FileZilla Server 1.7.0", ["cpe:2.3:a:filezilla-project:filezilla_server:1.7.0"]),
    ("Microsoft-IIS 10.0", ["cpe:2.3:a:microsoft:internet_information_services:10.0"]),
    ("Microsoft-HTTPAPI 2.0", []),  # not in the table: keyword search
    ("OpenSSH", []),
    ("OpenSSH 8.9:*:*:*", []),  # never more than a plain version in a CPE name
    ("", []),
])
def test_cpe_names(software, names):
    assert nvd.cpe_names(software) == names


def test_a_known_product_is_matched_by_version(nvd_reply, fake_session, no_nvd_wait):
    session = fake_session(get=nvd_reply(OPENSSH_VERSION))
    intel = recon_intel.Intel({"nvd": "nvd-test-key"}, session=session)
    found = intel.lookup("OpenSSH 8.9")
    ids = [c["id"] for c in found["cves"]]
    assert len(ids) == 32 and "CVE-2024-6387" in ids and "CVE-2023-48795" in ids, \
        "regreSSHion and Terrapin: the keyword search found neither"
    assert [c["cvss"] or 0 for c in found["cves"]] == sorted(
        (c["cvss"] or 0 for c in found["cves"]), reverse=True), "worst first"
    assert found["cve_source"] == f"version match ({OPENSSH_CPE})"
    ((method, url, kwargs),) = session.calls
    assert url == nvd.NVD_URL and kwargs["params"] == {
        "virtualMatchString": OPENSSH_CPE, "resultsPerPage": nvd.PAGE, "startIndex": 0,
        "noRejected": ""}
    assert kwargs["headers"] == {"apiKey": "nvd-test-key"} and kwargs["timeout"]
    intel.lookup("OpenSSH 8.9")
    assert len(session.calls) == 1, "cached per software"


def test_another_product_is_looked_up_by_keyword(nvd_reply, fake_session, no_nvd_wait):
    session = fake_session(get=nvd_reply(OPENSSH))
    found = recon_intel.Intel({}, session=session).lookup("Jetty 9.4.1")
    assert [c["id"] for c in found["cves"]] == ["CVE-2023-28531", "CVE-2021-36368"]
    assert found["cve_source"] == "keyword search (no version match for this product)"
    ((_, _, kwargs),) = session.calls
    assert kwargs["params"] == {"keywordSearch": "Jetty 9.4.1", "resultsPerPage": nvd.NVD_RESULTS,
                                "noRejected": ""}
    assert kwargs["headers"] == {}


def test_no_version_match_falls_back_to_the_keyword_search(nvd_reply, no_nvd_wait):
    session = ByQuery({"nginx 1.25.3": nvd_reply(OPENSSH)})
    found = recon_intel.Intel({}, session=session).lookup("nginx 1.25.3")
    assert [p.get("virtualMatchString") or p["keywordSearch"] for p in session.params] == [
        "cpe:2.3:a:f5:nginx:1.25.3", "nginx 1.25.3"]
    assert found["cve_source"] == "keyword search (the version match found none)"
    assert len(found["cves"]) == 2
    nothing = recon_intel.Intel({}, session=ByQuery({})).lookup("nginx 1.25.3")
    assert (nothing["severity"], nothing["cves"]) == ("NO KNOWN CVE", [])


def test_every_cpe_name_once_each_cve_once(no_nvd_wait):
    session = ByQuery({
        "cpe:2.3:a:oracle:mysql:5.0.51": {"totalResults": 2, "vulnerabilities": [
            _cve("CVE-2009-4484", 7.5), _cve("CVE-2008-0226", 7.5)]},
        "cpe:2.3:a:mysql:mysql:5.0.51": {"totalResults": 2, "vulnerabilities": [
            _cve("CVE-2008-0226", 7.5), _cve("CVE-2007-5925", 4.9)]},
    })
    found = recon_intel.Intel({}, session=session).lookup("MySQL 5.0.51")
    assert [c["id"] for c in found["cves"]] == ["CVE-2008-0226", "CVE-2009-4484", "CVE-2007-5925"]
    assert found["cve_source"] == ("version match (cpe:2.3:a:oracle:mysql:5.0.51, "
                                   "cpe:2.3:a:mysql:mysql:5.0.51)")
    assert len(session.params) == 2


def test_a_long_list_is_read_page_by_page(no_nvd_wait):
    name = "cpe:2.3:a:apache:http_server:2.2.8"
    pages = {(name, start): {"totalResults": 1100, "vulnerabilities": [
        _cve(f"CVE-2010-{start + n:04d}", 5.0) for n in range(nvd.PAGE if start < 1000 else 100)]}
        for start in (0, 500, 1000)}
    session = ByQuery(pages)
    found = recon_intel.Intel({}, session=session).lookup("Apache 2.2.8")
    assert [p["startIndex"] for p in session.params] == [0, 500, 1000]
    assert len(found["cves"]) == 1100


def test_pages_stop_at_the_limit_and_at_an_empty_page(no_nvd_wait):
    name = "cpe:2.3:a:apache:http_server:2.2.8"
    endless = ByQuery({(name, start): {"totalResults": 10**6, "vulnerabilities": [
        _cve(f"CVE-2011-{start:05d}", 5.0)]} for start in range(0, nvd.MAX_CVES, nvd.PAGE)})
    recon_intel.Intel({}, session=endless).lookup("Apache 2.2.8")
    assert len(endless.params) == nvd.MAX_CVES // nvd.PAGE, "at most MAX_CVES per CPE name"
    lying = ByQuery({(name, 0): {"totalResults": 10**6,
                                 "vulnerabilities": [_cve("CVE-2011-0001", 5)]}})
    recon_intel.Intel({}, session=lying).lookup("Apache 2.2.8")
    assert [p.get("startIndex") for p in lying.params] == [0, 500], "an empty page ends it"


@pytest.mark.parametrize("reply", [[1, 2], "text", None])
def test_a_reply_that_is_not_an_object_is_a_lookup_error(fake_session, no_nvd_wait, reply):
    found = recon_intel.Intel({}, session=fake_session(get=reply)).lookup("Jetty 9.4.1")
    assert found["errors"] == ["NVD: ValueError"] and found["severity"] == "UNKNOWN"


def test_requests_are_spaced_without_a_key(nvd_reply, fake_session):
    sleeps = []

    class Waits(threading.Event):
        """The scan's stop event (v5.2 step 2.4c: NVD waits for its slot on it): records each
        wait instead of waiting, never set."""

        def wait(self, timeout=None):
            if timeout:
                sleeps.append(timeout)
            return False

    intel = recon_intel.Intel({}, session=fake_session(get=nvd_reply(EMPTY)), stop=Waits())
    intel.lookup("Jetty 9.4.1")
    intel.lookup("Tomcat 9.0.1")
    assert len(sleeps) == 1 and 5 < sleeps[0] <= 6
    sleeps.clear()
    intel.lookup("nginx 1.25.3")  # the version match, then the keyword search: two requests
    assert len(sleeps) == 2 and 5 < sleeps[1] - sleeps[0] <= 6, "each its own slot"


def test_the_source_is_shown_and_saved(tmp_path, capsys):
    from toolkit.cli.ui import scan as screen
    from toolkit.core.data.recon import scan_files
    from toolkit.core.data.recon.scanner import PortResult
    from toolkit.core.services.recon import report

    found = {"cves": [{"id": "CVE-2024-6387", "cvss": 8.1, "cvss_version": "3.1",
                       "severity": "HIGH"}], "cve_source": f"version match ({OPENSSH_CPE})",
             "max_cvss": 8.1, "severity": "HIGH", "ai_explanation": "", "errors": []}
    entry = report.make_entry("10.0.0.5", PortResult(22, "tcp", "open", "ssh", "SSH-2.0-x"),
                              "OpenSSH 8.9", found)
    screen.print_entry(entry)
    out = capsys.readouterr().out
    assert "CVEs (NVD): CVE-2024-6387 (CVSS 8.1, HIGH)\n" in out
    assert f"CVE source: version match ({OPENSSH_CPE})\n" in out
    csv_file = scan_files.export_results([entry], "csv", tmp_path)
    header, row = csv_file.read_text(encoding="utf-8").splitlines()
    assert header.endswith(",CVE source,Engine,CPE (nmap)")
    assert row.endswith(f",version match ({OPENSSH_CPE}),builtin,")
    txt = scan_files.export_results([entry], "txt", tmp_path).read_text(encoding="utf-8")
    assert f"CVE source: version match ({OPENSSH_CPE})\n" in txt
