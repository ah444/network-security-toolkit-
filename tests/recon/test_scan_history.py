"""F6a: every finished scan, the CLI's and the web's, is saved in ``instance/scan_history/`` as
``<scan id>.json``, the id of its lines in the log (HS3): what was asked, what each host gave,
what went online and the report entries. A stopped or failed scan saves nothing; a full disk
keeps the scan and says it was not saved; no key is ever written. No test reaches the network."""

import json
import re
import threading
import time

import pytest

from toolkit.core.data.recon import scan_records
from toolkit.core.data.recon.nmap_run import NmapFailed
from toolkit.core.data.recon.scanner import PortResult, ScanStopped
from toolkit.core.services.recon import history

from .test_scan_flags import FakeIntel
from .test_scan_record import RECORD, lines

AUTHORIZED = "--i-am-authorized"
SCAN_ID = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")


@pytest.fixture
def offline(fake_ports, scan_service, scanner, monkeypatch):
    """The ping, the port checks and the lookups faked (one open port); no key set."""
    monkeypatch.setattr(scan_service, "ping_host", lambda host, timeout=1: (True, 64))
    fake_ports(lambda host, ports, protocols, progress=None: [
        PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1")])
    monkeypatch.setattr(scan_service, "Intel", FakeIntel)
    monkeypatch.setattr(scanner, "load_api_keys",
                        lambda: {"openrouter": "", "abuseipdb": "", "nvd": ""})


@pytest.fixture
def caplog_info(caplog):
    caplog.set_level("INFO", logger=RECORD)
    return caplog


def saved():
    """The records in the scan history, by name."""
    folder = scan_records.history_dir()
    return sorted(p.name for p in folder.glob("*.json")) if folder.is_dir() else []


def only_record():
    [name] = saved()
    return name, json.loads((scan_records.history_dir() / name).read_text(encoding="utf-8"))


def test_a_finished_cli_scan_is_saved_under_its_log_id(scanner, offline, caplog_info, capsys):
    assert scanner.main(["127.0.0.1", "-p", "22,80-82", AUTHORIZED]) == 0
    name, record = only_record()
    scan_id = lines(caplog_info)[0][1]["scan_id"]
    assert SCAN_ID.fullmatch(scan_id) and name == f"{scan_id}.json"
    assert record["kind"] == "scan" and record["schema"] == history.SCHEMA
    assert record["scan_id"] == scan_id and record["source"] == "this computer"
    assert (record["target"], record["addresses"], record["ports"], record["protocols"],
            record["engine"], record["nmap"]) == (
        "127.0.0.1", ["127.0.0.1"], "22,80-82", ["tcp"], "builtin", None)
    assert set(record["online"]) == {"nvd", "ai", "abuseipdb"}
    assert record["hosts"] == [{"target": "127.0.0.1", "state": "scanned",
                                "os": record["hosts"][0]["os"], "message": ""}]
    assert [(e["port"], e["protocol"], e["banner"]) for e in record["entries"]] == [
        (22, "tcp", "SSH-2.0-OpenSSH_8.9p1")]
    assert record["summary"]["open_ports"] == 1 and record["summary"]["sent"]["NVD"] == 1
    assert record["started"].endswith("Z") and record["seconds"] >= 0
    assert record["exit_code"] == 0
    assert f"Saved in the scan history as {scan_id}." in capsys.readouterr().out


def test_a_scan_with_no_open_port_is_saved_too(scanner, offline, fake_ports):
    fake_ports(lambda host, ports, protocols, progress=None: [])
    assert scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED]) == 0
    _, record = only_record()
    assert record["entries"] == [] and record["hosts"][0]["state"] == "scanned"


def test_a_stopped_scan_saves_nothing(scan_service, scan_screen, offline):
    stop = threading.Event()
    stop.set()
    request = scan_service.ScanRequest("127.0.0.1", ["127.0.0.1"], [22], ["tcp"])
    with pytest.raises(ScanStopped):
        scan_service.run_scan(request, {}, time.time(), scan_screen.ConsoleScan(), stop)
    assert saved() == []


def test_ctrl_c_and_an_unexpected_error_save_nothing(scanner, offline, fake_ports):
    def interrupted(host, ports, protocols, progress=None):
        raise KeyboardInterrupt

    fake_ports(interrupted)
    assert scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED]) == 130

    def broken(host, ports, protocols, progress=None):
        raise RuntimeError("the engine broke")

    fake_ports(broken)
    with pytest.raises(RuntimeError):
        scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED])
    assert saved() == []


def test_a_host_that_did_not_answer_is_skipped_not_closed(scan_service, scan_screen, offline,
                                                         monkeypatch):
    monkeypatch.setattr(scan_service, "ping_host", lambda host, timeout=1: (host != "10.0.0.2", 64))
    request = scan_service.ScanRequest("10.0.0.0/31", ["10.0.0.1", "10.0.0.2"], [22], ["tcp"])
    assert scan_service.run_scan(request, {}, time.time(), scan_screen.ConsoleScan()) == 0
    _, record = only_record()
    assert [(h["target"], h["state"]) for h in record["hosts"]] == [
        ("10.0.0.1", "scanned"), ("10.0.0.2", "skipped")]
    assert [e["target"] for e in record["entries"]] == ["10.0.0.1"]


def test_a_host_the_engine_could_not_scan_is_failed_with_why(scan_service, scan_screen, offline,
                                                            fake_ports):
    def checks(host, ports, protocols, progress=None):
        if host == "10.0.0.1":
            raise NmapFailed("nmap stopped with exit code 1: x")
        return [PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1")]

    fake_ports(checks)
    request = scan_service.ScanRequest("10.0.0.0/31", ["10.0.0.1", "10.0.0.2"], [22], ["tcp"])
    assert scan_service.run_scan(request, {}, time.time(), scan_screen.ConsoleScan()) == 1
    _, record = only_record()
    assert record["hosts"][0] == {"target": "10.0.0.1", "state": "failed", "os": "",
                                  "message": "nmap stopped with exit code 1: x"}
    assert record["hosts"][1]["state"] == "scanned" and record["exit_code"] == 1


def test_the_nmap_choices_are_kept(scan_service, scan_screen, offline):
    request = scan_service.ScanRequest(
        "127.0.0.1", ["127.0.0.1"], [22], ["tcp"],
        nmap_options=scan_service.NmapOptions(timing="aggressive", scripts="safe"))
    record = history.scan_record("20261103T120000Z-0123abcd", request,
                                 scan_service.online_plan(request, {}), history.ScanNotes(
                                     scan_screen.ConsoleScan(), request.targets), 0, 0.0, 0)
    assert record["nmap"] is None  # the built-in engine: nmap's choices don't apply
    nmap = scan_service.ScanRequest("127.0.0.1", ["127.0.0.1"], [22], ["tcp"], engine="nmap",
                                    nmap_options=request.nmap_options)
    record = history.scan_record("20261103T120000Z-0123abcd", nmap,
                                 scan_service.online_plan(nmap, {}), history.ScanNotes(
                                     scan_screen.ConsoleScan(), nmap.targets), 0, 0.0, 0)
    assert record["nmap"] == {"service_detection": "normal", "timing": "aggressive",
                              "scripts": "safe", "os_detect": False}


def test_no_key_is_in_the_record_and_the_reputation_is(scanner, offline, monkeypatch):
    monkeypatch.setattr(scanner, "load_api_keys", lambda: {
        "openrouter": "or-secret-value-1", "abuseipdb": "abuse-secret-value-2",
        "nvd": "nvd-secret-value-3"})
    assert scanner.main(["8.8.8.8", "-p", "22", "--abuseipdb", "--ai", AUTHORIZED]) == 0
    name, record = only_record()
    text = (scan_records.history_dir() / name).read_text(encoding="utf-8")
    assert "secret-value" not in text
    assert record["reputation"]["8.8.8.8"]["abuseConfidenceScore"] == "0"


def test_a_full_disk_keeps_the_scan_and_says_it_was_not_saved(scanner, offline, monkeypatch,
                                                             capsys):
    monkeypatch.setenv("TOOLKIT_MIN_FREE_MB", str(10 ** 12))
    assert scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED]) == 0
    out = capsys.readouterr().out
    assert "22/tcp" in out and "Not saved in the scan history: " in out
    assert saved() == []


def test_a_record_is_never_replaced(tmp_path):
    scan_records.write_scan("20261103T120000Z-0123abcd", {"kind": "scan", "n": 1}, tmp_path)
    with pytest.raises(FileExistsError):
        scan_records.write_scan("20261103T120000Z-0123abcd", {"kind": "scan", "n": 2}, tmp_path)
    assert json.loads((tmp_path / "20261103T120000Z-0123abcd.json").read_text(
        encoding="utf-8"))["n"] == 1
    assert [p.name for p in scan_records.list_scans(tmp_path)] == [
        "20261103T120000Z-0123abcd.json"]


def test_a_scan_id_is_a_time_and_eight_hex_digits():
    first, second = history.new_scan_id(), history.new_scan_id()
    assert SCAN_ID.fullmatch(first) and first != second
    assert history.is_scan_id(first)
    assert not any(history.is_scan_id(text) for text in (
        "", "../x", "20261103T120000Z-0123abcd.json", "20261103T120000Z-0123ABCD", "x" * 25))


def test_the_record_holds_no_object_json_cant_read(scanner, offline):
    FakeIntel.made.clear()
    assert scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED]) == 0
    name, record = only_record()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
