"""F6b: the saved scans read back from ``instance/scan_history/``: the list, newest first and
50 at a time; one scan as the scan page shows it, with the choices to scan again; its report;
its delete. A file there is data someone could have edited: a scan id is checked before a path
is made from it, and a file that isn't a scan record is left out of a list and counted."""

import json

import pytest

from toolkit.core.data.recon import scan_records
from toolkit.core.data.recon.scanner import PortResult
from toolkit.core.services.recon import history, saved_scans

from .test_scan_flags import FakeIntel

AUTHORIZED = "--i-am-authorized"
A = "20261101T100000Z-0000000a"
B = "20261102T100000Z-0000000b"
C = "20261103T100000Z-0000000c"


def entry(port, severity="NO KNOWN CVE", unconfirmed=False, target="10.0.0.5"):
    return {"target": target, "port": port, "protocol": "tcp", "service": "ssh",
            "banner": "SSH-2.0-OpenSSH_8.9p1", "software": "OpenSSH 8.9", "cves": [],
            "cve_source": "", "max_cvss": None, "severity": severity,
            "severity_unconfirmed": unconfirmed, "ai_explanation": "", "lookup_errors": [],
            "os_hint": "", "engine": "builtin", "cpe": []}


def record(scan_id, **over):
    made = {
        "kind": "scan", "schema": history.SCHEMA, "scan_id": scan_id,
        "started": "2026-11-01T10:00:00Z", "seconds": 4.2, "source": "this computer",
        "target": "10.0.0.4/31", "addresses": ["10.0.0.4", "10.0.0.5"], "ports": "22,80-82",
        "protocols": ["tcp"], "engine": "builtin", "nmap": None,
        "choices": {"scan_unresponsive": False, "nvd": True, "ai": False, "abuseipdb": False},
        "online": {"nvd": "on", "ai": "off: no OPENROUTER_API_KEY", "abuseipdb": "off: x"},
        "hosts": [{"target": "10.0.0.4", "state": "skipped", "os": "", "message": ""},
                  {"target": "10.0.0.5", "state": "scanned", "os": "Linux", "message": ""}],
        "reputation": {}, "entries": [entry(22)],
        "summary": {"targets": 2, "open_ports": 1, "failed_lookups": [],
                    "sent": {"NVD": 1, "AI (OpenRouter)": 0, "AbuseIPDB": 0}, "unavailable": {}},
        "exit_code": 0}
    made.update(over)
    return made


def save(scan_id, **over):
    scan_records.write_scan(scan_id, record(scan_id, **over))


def test_the_list_is_newest_first_with_a_summary_of_each():
    save(A)
    save(B, entries=[entry(22, "HIGH", unconfirmed=True), entry(80, "LOW")])
    save(C, source="192.168.56.20")
    answer = saved_scans.saved_scans()
    assert [s["scan_id"] for s in answer["scans"]] == [C, B, A]
    assert answer["more"] is False and answer["left_out"] == 0
    first = answer["scans"][2]
    assert first == {"scan_id": A, "started": "2026-11-01T10:00:00Z", "source": "this computer",
                     "target": "10.0.0.4/31", "ports": "22,80-82", "port_count": 4,
                     "protocols": ["tcp"], "engine": "builtin",
                     "hosts": {"scanned": 1, "skipped": 1, "failed": 0}, "open_ports": 1,
                     "worst": "NO KNOWN CVE", "worst_unconfirmed": False, "exit_code": 0}
    assert answer["scans"][1]["worst"] == "HIGH" and answer["scans"][1]["worst_unconfirmed"]
    assert answer["scans"][0]["source"] == "192.168.56.20"


def test_the_worst_severity_is_of_the_ports_looked_up_only():
    unknown = {**entry(5040, "UNKNOWN"), "software": ""}  # no software: no lookup
    save(A, entries=[unknown])
    save(B, entries=[unknown, entry(22, "NOT CHECKED")])
    worst = {s["scan_id"]: s["worst"] for s in saved_scans.saved_scans()["scans"]}
    assert worst == {A: None, B: "NOT CHECKED"}


def test_the_newest_scan_of_each_target_the_riskiest_first():
    """F10: the dashboard's targets. An older scan of a target never shows; within a risk,
    the newest first."""
    save(A, target="10.0.0.9", entries=[entry(22, "CRITICAL")])  # older: replaced by C
    save(B, target="10.0.0.5", entries=[entry(22, "HIGH", unconfirmed=True)])
    save(C, target="10.0.0.9", entries=[entry(22, "LOW")])
    save("20261104T100000Z-0000000d", target="10.0.0.7", entries=[])
    save("20261105T100000Z-0000000e", target="10.0.0.8",
         entries=[{**entry(5040, "UNKNOWN"), "software": ""}])
    rows = saved_scans.targets_at_risk(saved_scans.saved_scans()["scans"])
    assert [(r["target"], r["risk"], r["unconfirmed"]) for r in rows] == [
        ("10.0.0.5", "HIGH", True), ("10.0.0.9", "LOW", False),
        ("10.0.0.8", "no lookup", False), ("10.0.0.7", "no open port", False)]
    assert rows[1]["scan_id"] == C and rows[1]["open_ports"] == 1


def test_the_risk_of_a_scan_in_a_word():
    assert saved_scans.risk({"worst": "NOT CHECKED", "open_ports": 1}) == "NOT CHECKED"
    assert saved_scans.risk({"worst": None, "open_ports": 2}) == "no lookup"
    assert saved_scans.risk({"worst": None, "open_ports": 0}) == "no open port"


def test_the_list_comes_fifty_at_a_time():
    for day in range(1, 4):
        save(f"202611{day:02d}T100000Z-0000000{day}")
    first = saved_scans.saved_scans(limit=2)
    assert [s["scan_id"][:8] for s in first["scans"]] == ["20261103", "20261102"]
    assert first["more"] is True
    rest = saved_scans.saved_scans(before=first["scans"][-1]["scan_id"], limit=2)
    assert [s["scan_id"][:8] for s in rest["scans"]] == ["20261101"] and rest["more"] is False
    assert saved_scans.PAGE_SIZE == 50
    with pytest.raises(saved_scans.NotAScan):
        saved_scans.saved_scans(before="../x")


def test_a_file_that_isnt_a_scan_record_is_left_out_and_counted():
    save(A)
    folder = scan_records.history_dir()
    (folder / f"{B}.json").write_text("{not json", encoding="utf-8")
    (folder / f"{C}.json").write_text(json.dumps(record(A)), encoding="utf-8")  # renamed
    (folder / "20261104T100000Z-0000000d.json").write_text(
        json.dumps(record("20261104T100000Z-0000000d", kind="training")), encoding="utf-8")
    (folder / "20261105T100000Z-0000000e.json").write_text(
        json.dumps(record("20261105T100000Z-0000000e", schema=history.SCHEMA + 1)),
        encoding="utf-8")
    (folder / "notes.json").write_text("{}", encoding="utf-8")
    answer = saved_scans.saved_scans()
    assert [s["scan_id"] for s in answer["scans"]] == [A] and answer["left_out"] == 5
    for bad in (B, C, "20261104T100000Z-0000000d", "20261105T100000Z-0000000e"):
        with pytest.raises(saved_scans.NotAScan):
            saved_scans.saved_scan(bad)


def test_a_scan_id_is_checked_before_any_path_is_made():
    for bad in ("../../runs/x", "", "20261101T100000Z-0000000A", A + ".json"):
        with pytest.raises(saved_scans.NotAScan):
            saved_scans.saved_scan(bad)
        with pytest.raises(saved_scans.NotAScan):
            saved_scans.remove_scan(bad)
    with pytest.raises(FileNotFoundError):
        saved_scans.saved_scan(A)


def test_one_scan_is_shown_as_the_scan_page_shows_a_finished_scan():
    save(A)
    view = saved_scans.scan_view(saved_scans.saved_scan(A))
    assert view["message"] == "Scanned 2 host(s): 1 open port(s)."
    assert (view["target"], view["ports"], view["port_list"], view["protocols"]) == (
        "10.0.0.4/31", 4, "22,80-82", ["tcp"])
    assert view["engine"] == "builtin" and view["engine_notice"] == ["built-in"]
    assert view["summary"]["seconds"] == 4.2 and view["summary"]["sent"]["NVD"] == 1
    assert view["hosts"][0] == {"target": "10.0.0.4", "state": "skipped", "os": ""}
    assert view["entries"] == [entry(22)] and view["failed_targets"] == []
    assert (view["scan_id"], view["started"], view["source"]) == (
        A, "2026-11-01T10:00:00Z", "this computer")
    assert view["again"] == {"target": "10.0.0.4/31", "ports": "22,80-82", "protocol": "tcp",
                             "engine": "builtin", "scan_unresponsive": False, "nvd": True,
                             "ai": False, "abuseipdb": False}


def test_scan_again_keeps_nmaps_choices_and_the_protocols():
    nmap = {"service_detection": "light", "timing": "aggressive", "scripts": "safe",
            "os_detect": True}
    save(A, engine="nmap", nmap=nmap, protocols=["tcp", "udp"],
         hosts=[{"target": "10.0.0.5", "state": "failed", "os": "", "message": "nmap: x"}],
         exit_code=1)
    view = saved_scans.scan_view(saved_scans.saved_scan(A))
    assert view["again"]["protocol"] == "both" and view["again"]["engine"] == "nmap"
    assert {k: view["again"][k] for k in nmap} == nmap
    assert view["engine_notice"] == ["nmap: service detection light, timing aggressive, "
                                     "the safe scripts, OS detection"]
    assert view["failed_targets"] == ["10.0.0.5"]
    assert view["message"] == "Scanned 1 host(s): 1 open port(s). Could not scan 1."


def test_a_record_without_the_choices_scans_again_with_the_defaults():
    old = record(A)
    del old["choices"]
    scan_records.write_scan(A, old)
    again = saved_scans.scan_view(saved_scans.saved_scan(A))["again"]
    assert (again["scan_unresponsive"], again["nvd"], again["ai"], again["abuseipdb"]) == (
        False, True, True, True)


def test_its_report_is_the_downloads():
    save(A)
    text = saved_scans.saved_report(A, "json")
    assert json.loads(text) == [entry(22)]
    assert "Port: 22/tcp" in saved_scans.saved_report(A, "txt")
    with pytest.raises(ValueError):
        saved_scans.saved_report(A, "pdf")


def test_delete_removes_that_record_only():
    save(A)
    save(B)
    assert saved_scans.remove_scan(A) == f"{A}.json"
    assert [p.name for p in scan_records.list_scans()] == [f"{B}.json"]
    with pytest.raises(FileNotFoundError):
        saved_scans.remove_scan(A)


def test_a_finished_scan_keeps_its_choices_to_scan_again(scanner, fake_ports, scan_service,
                                                         monkeypatch):
    monkeypatch.setattr(scan_service, "ping_host", lambda host, timeout=1: (True, 64))
    fake_ports(lambda host, ports, protocols, progress=None: [
        PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1")])
    monkeypatch.setattr(scan_service, "Intel", FakeIntel)
    monkeypatch.setattr(scanner, "load_api_keys",
                        lambda: {"openrouter": "", "abuseipdb": "", "nvd": ""})
    assert scanner.main(["127.0.0.1", "-p", "22", "--no-nvd", "--scan-unresponsive",
                         AUTHORIZED]) == 0
    [path] = scan_records.list_scans()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["choices"] == {"scan_unresponsive": True, "nvd": False, "ai": False,
                                "abuseipdb": False}
    assert saved_scans.scan_view(saved_scans.saved_scan(path.stem))["again"]["nvd"] is False
