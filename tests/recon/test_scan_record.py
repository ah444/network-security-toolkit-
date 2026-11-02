"""HS3 (RD1): every scan writes one line to the log when it starts and one when it ends, from
one place (``run_scan``), so the CLI's scans and the web's are recorded alike: when, the
targets, the ports, where the request came from and the outcome. The values are JSON, so a
typed target can't forge a line. No test reaches the network."""

import json
import threading
import time

import pytest

from toolkit.core.data.recon.scanner import PortResult, ScanStopped
from toolkit.core.services.recon import record

from .test_scan_flags import FakeIntel

AUTHORIZED = "--i-am-authorized"
RECORD = "toolkit.core.services.recon.record"


def lines(caplog):
    """The record's lines as (event, fields)."""
    found = []
    for item in caplog.records:
        if item.name == RECORD:
            word, event, fields = item.getMessage().split(" ", 2)
            assert word == "Scan" and "\n" not in item.getMessage()
            found.append((event, json.loads(fields)))
    return found


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


def test_a_cli_scan_writes_a_line_when_it_starts_and_one_when_it_ends(scanner, offline,
                                                                       caplog_info):
    assert scanner.main(["127.0.0.1", "-p", "22,80-82", AUTHORIZED]) == 0
    [(first, started), (last, ended)] = lines(caplog_info)
    assert (first, last) == ("started", "ended")
    assert started == {"scan_id": started["scan_id"], "source": "this computer",
                       "target": "127.0.0.1", "addresses": ["127.0.0.1"], "ports": "22,80-82",
                       "protocols": ["tcp"], "engine": "builtin", "online": started["online"]}
    assert set(started["online"]) == {"nvd", "ai", "abuseipdb"}
    assert started["online"]["ai"].startswith("off")
    assert ended == {"scan_id": started["scan_id"], "outcome": "done", "open_ports": 1,
                     "engine_failed": 0, "seconds": ended["seconds"]}
    assert ended["seconds"] >= 0


def test_ctrl_c_during_a_cli_scan_is_recorded_as_stopped(scanner, offline, fake_ports,
                                                         caplog_info):
    def interrupted(host, ports, protocols, progress=None):
        raise KeyboardInterrupt

    fake_ports(interrupted)
    assert scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED]) == 130
    assert [(event, fields.get("outcome")) for event, fields in lines(caplog_info)] == [
        ("started", None), ("ended", "stopped")]


def test_a_stopped_scan_is_recorded_as_stopped(scan_service, scan_screen, offline, caplog_info):
    stop = threading.Event()
    stop.set()
    request = scan_service.ScanRequest("127.0.0.1", ["127.0.0.1"], [22], ["tcp"])
    with pytest.raises(ScanStopped):
        scan_service.run_scan(request, {}, time.time(), scan_screen.ConsoleScan(), stop)
    assert lines(caplog_info)[-1][1]["outcome"] == "stopped"


def test_an_unexpected_error_is_recorded_as_failed(scanner, offline, fake_ports, caplog_info):
    def broken(host, ports, protocols, progress=None):
        raise RuntimeError("the engine broke")

    fake_ports(broken)
    with pytest.raises(RuntimeError):
        scanner.main(["127.0.0.1", "-p", "22", AUTHORIZED])
    assert lines(caplog_info)[-1][1]["outcome"] == "failed"


def test_the_source_is_the_requests(scan_service, scan_screen, offline, caplog_info):
    request = scan_service.ScanRequest("127.0.0.1", ["127.0.0.1"], [22], ["tcp"],
                                       source="192.168.56.20")
    scan_service.run_scan(request, {}, time.time(), scan_screen.ConsoleScan())
    assert lines(caplog_info)[0][1]["source"] == "192.168.56.20"


def test_a_typed_target_cannot_forge_a_line(caplog_info):
    forged = 'x\n2026-11-03T10:00:00 INFO forged: Scan ended {"outcome": "done"} "q" '
    scan_id = record.scan_started(record.THIS_COMPUTER, forged, ["10.0.0.5"], [22, 80, 81],
                                  ["tcp"], "builtin", {"nvd": "on"})
    [(event, fields)] = lines(caplog_info)
    assert event == "started" and fields["target"] == forged and fields["scan_id"] == scan_id
    assert fields["ports"] == "22,80-81"
    message = caplog_info.records[-1].getMessage()
    assert message.isascii() and "\n" not in message and " " not in message


def test_no_key_is_in_the_lines(scanner, offline, monkeypatch, caplog_info):
    monkeypatch.setattr(scanner, "load_api_keys", lambda: {
        "openrouter": "or-secret-value-1", "abuseipdb": "abuse-secret-value-2",
        "nvd": "nvd-secret-value-3"})
    assert scanner.main(["8.8.8.8", "-p", "22", "--abuseipdb", "--ai", AUTHORIZED]) == 0
    text = " ".join(item.getMessage() for item in caplog_info.records if item.name == RECORD)
    assert "Scan started" in text and "secret-value" not in text
