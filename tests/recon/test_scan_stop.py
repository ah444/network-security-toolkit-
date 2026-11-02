"""v5.2 step 2.4c: a scan stops when asked (the web scan's Cancel), nothing more sent to the
targets or online, and the report as text for a download that saves no file."""

import json
import os
import threading
import time

import pytest

from toolkit.core.data.recon import nvd, scan_files
from toolkit.core.data.recon.findings import IntelFinding
from toolkit.core.data.recon.scanner import PortResult, ScanStopped
from toolkit.core.services.recon import engines, report
from toolkit.core.services.recon import scan as scan_service
from toolkit.core.services.recon.intel import Intel

OPEN = [PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9"),
        PortResult(80, "tcp", "open", "http", "Server: nginx/1.25.3"),
        PortResult(3306, "tcp", "open", "mysql", "5.0.51a-log")]


class Cves:
    """An offline CVE source that records what it was asked; ``on_ask`` runs first."""

    name, kind = "Offline", "cves"

    def __init__(self, on_ask=lambda subject: None):
        self.asked, self.on_ask = [], on_ask

    def lookup(self, subject):
        self.on_ask(subject)
        self.asked.append(subject)
        return IntelFinding(cves=(), source="offline")


class Events:
    """A ScanView that records what it is told; ``on`` runs a step's own action."""

    def __init__(self, on=None):
        self.seen, self.on = [], on or {}

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.seen.append(name)
            return self.on.get(name, lambda: None)()
        return record


@pytest.fixture
def network(monkeypatch):
    """The targets and the online services, faked: what was pinged, scanned and looked up."""
    sent = {"ping": [], "scan": [], "stops": []}

    class Engine(engines.BuiltinEngine):
        def scan(self, host, ports, protocols, progress=None, stop=None):
            sent["scan"].append(host)
            sent["stops"].append(stop)
            return list(OPEN)

    def ping(host):
        sent["ping"].append(host)
        return True, 64

    monkeypatch.setitem(engines.ENGINES, "builtin", Engine)
    monkeypatch.setattr(scan_service, "ping_host", ping)
    cves = Cves()
    real_intel = Intel

    def offline_intel(keys, use_ai=False, stop=None, nvd=True):
        return real_intel(keys, use_ai, providers=[cves], stop=stop)

    monkeypatch.setattr(scan_service, "Intel", offline_intel)
    sent["lookups"] = cves.asked
    sent["cves"] = cves
    return sent


def _request(*targets, **options):
    return scan_service.ScanRequest(" ".join(targets), list(targets), [22, 80, 3306], ["tcp"],
                                    **options)


# --- The scan -----------------------------------------------------------------------------

def test_a_scan_nobody_stops_runs_as_before(network):
    view = Events()
    assert scan_service.run_scan(_request("10.0.0.5"), {}, time.time(), view) == 0
    assert network["scan"] == ["10.0.0.5"] and network["stops"] == [None]
    assert sorted(network["lookups"]) == ["OpenSSH 8.9", "nginx 1.25.3"]
    assert "summary" in view.seen


def test_a_stopped_scan_sends_nothing(network):
    stop = threading.Event()
    stop.set()
    view = Events()
    with pytest.raises(ScanStopped):
        scan_service.run_scan(_request("8.8.8.8"), {"abuseipdb": "k"}, time.time(), view, stop)
    assert network["ping"] == network["scan"] == network["lookups"] == []
    assert "summary" not in view.seen and "exported" not in view.seen


def test_the_engine_gets_the_stop(network):
    stop = threading.Event()
    scan_service.run_scan(_request("10.0.0.5"), {}, time.time(), Events(), stop)
    assert network["stops"] == [stop]


def test_a_stop_after_the_ping_scans_no_port(network, monkeypatch):
    stop = threading.Event()

    def ping(host):
        network["ping"].append(host)
        stop.set()  # the Cancel while the ping was out
        return True, 64

    monkeypatch.setattr(scan_service, "ping_host", ping)
    with pytest.raises(ScanStopped):
        scan_service.run_scan(_request("10.0.0.5"), {}, time.time(), Events(), stop)
    assert network["ping"] == ["10.0.0.5"] and network["scan"] == []


def test_a_stop_between_targets_leaves_the_next_alone(network):
    stop = threading.Event()
    view = Events(on={"target_done": stop.set})
    with pytest.raises(ScanStopped):
        scan_service.run_scan(_request("10.0.0.5", "10.0.0.6"), {}, time.time(), view, stop)
    assert network["ping"] == network["scan"] == ["10.0.0.5"]
    assert "summary" not in view.seen


def test_a_stop_after_the_last_target_shows_no_summary(network):
    stop = threading.Event()
    view = Events(on={"target_done": stop.set})
    with pytest.raises(ScanStopped):
        scan_service.run_scan(_request("10.0.0.5"), {}, time.time(), view, stop)
    assert "summary" not in view.seen and "exported" not in view.seen


def test_a_stop_during_the_lookups_ends_the_scan(network):
    stop = threading.Event()
    network["cves"].on_ask = lambda subject: stop.set()  # the Cancel during the first lookup
    view = Events()
    with pytest.raises(ScanStopped):
        scan_service.run_scan(_request("10.0.0.5"), {}, time.time(), view, stop)
    assert "entry" not in view.seen and "summary" not in view.seen
    asked = len(network["lookups"])
    time.sleep(0.1)
    assert len(network["lookups"]) == asked, "nothing asked after the stop"


# --- The lookups --------------------------------------------------------------------------

def test_a_stopped_intel_asks_no_provider():
    stop = threading.Event()
    stop.set()
    cves = Cves()
    intel = Intel({}, providers=[cves], stop=stop)
    with pytest.raises(ScanStopped):
        intel.lookup("OpenSSH 8.9")
    with pytest.raises(ScanStopped):
        intel.lookup_many(["OpenSSH 8.9", "nginx 1.25.3"])
    assert cves.asked == []


def test_a_stop_cancels_the_queued_lookups():
    stop = threading.Event()
    cves = Cves(on_ask=lambda subject: stop.set())
    intel = Intel({}, providers=[cves], stop=stop)
    with pytest.raises(ScanStopped):
        intel.lookup_many([f"Product {n}.0" for n in range(8)], workers=1)
    assert len(cves.asked) == 1, "the first finished, none of the others was asked"


def test_a_stopped_lookup_is_not_kept():
    stop = threading.Event()
    calls = []

    class Explainer:
        name, kind = "AI", "explanation"

        def lookup(self, subject):
            calls.append(subject)
            return IntelFinding(text="explained")

    cves = Cves(on_ask=lambda subject: None if cves.asked else stop.set())
    intel = Intel({}, providers=[cves, Explainer()], stop=stop)
    with pytest.raises(ScanStopped):
        intel.lookup("OpenSSH 8.9")
    assert calls == [], "the explanation was never asked"
    stop.clear()
    assert intel.lookup("OpenSSH 8.9")["ai_explanation"] == "explained", "asked again in full"


def test_nvd_stops_waiting_for_its_slot(fake_session):
    session = fake_session(get={"vulnerabilities": []})
    stop = threading.Event()
    provider = nvd.NvdProvider(session, stop=stop)
    provider._next = time.monotonic() + 60  # the next slot is a minute away (no key)
    threading.Timer(0.1, stop.set).start()
    began = time.monotonic()
    with pytest.raises(ScanStopped):
        provider.lookup("nginx 1.25.3")
    assert time.monotonic() - began < 5 and session.calls == [], "nothing sent"


def test_intel_gives_nvd_its_stop():
    stop = threading.Event()
    (provider,) = Intel({}, stop=stop).providers
    assert isinstance(provider, nvd.NvdProvider) and provider.stop is stop


# --- The report as text -------------------------------------------------------------------

def _entries():
    found = {"cves": [{"id": "CVE-2021-41773", "cvss": 9.8, "cvss_version": "3.1",
                       "severity": "CRITICAL", "match": "version"}],
             "max_cvss": 9.8, "severity": "CRITICAL", "ai_explanation": "", "errors": [],
             "severity_unconfirmed": False,
             "cve_source": "version match (cpe:2.3:a:apache:http_server:2.4.49)"}
    result = PortResult(8080, "tcp", "open", "http-alt", "=cmd|' /C calc'!A0 Apache/2.4.49 café")
    return [report.make_entry("127.0.0.1", result, "Apache 2.4.49", found, "Windows", "builtin")]


@pytest.mark.parametrize("fmt", scan_files.EXPORT_FORMATS)
def test_the_report_text_is_what_the_export_saves(fmt):
    entries = _entries()
    text = scan_files.report_text(entries, fmt)
    saved = scan_files.export_results(entries, fmt).read_bytes()
    expected = text if fmt == "csv" else text.replace("\n", os.linesep)
    assert saved == expected.encode("utf-8")
    assert "café" in text and "CVE-2021-41773" in text


def test_the_report_text_in_each_format():
    entries = _entries()
    assert json.loads(scan_files.report_text(entries, "json")) == entries
    csv_text = scan_files.report_text(entries, "csv")
    assert csv_text.startswith("Target,Port,Protocol,") and csv_text.endswith("\r\n")
    assert "'=cmd|" in csv_text, "a cell that looks like a formula stays text (S14)"
    assert scan_files.report_text(entries, "txt").startswith("Target: 127.0.0.1\nOS hint: Windows\n")
    assert scan_files.report_text([], "json") == "[]" and scan_files.report_text([], "txt") == ""
    assert scan_service.report_text is scan_files.report_text


def test_an_unknown_format_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="Unknown format 'exe'"):
        scan_files.report_text(_entries(), "exe")
    with pytest.raises(ValueError, match="Unknown format"):
        scan_files.export_results(_entries(), "exe", tmp_path / "scans")
    assert not (tmp_path / "scans").exists()
