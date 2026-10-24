"""S16: no scan without confirmation; host and port limits."""

import builtins
import socket
import subprocess

import pytest

from toolkit.data.settings import ScanConfig, scan_limits
from toolkit.recon import targets


def _answers(monkeypatch, answers):
    remaining = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(remaining))
    return remaining


@pytest.fixture
def no_network(monkeypatch):
    """Any packet sent to the target fails the test."""
    def forbidden(*args, **kwargs):
        raise AssertionError("the scan started")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def test_scan_requires_confirmation(scanner, monkeypatch, no_network, capsys):
    for answer in ("no", "", "y"):  # only the word "yes" confirms
        remaining = _answers(monkeypatch, ["127.0.0.1", "80", "80", "tcp", answer])
        assert scanner.main([]) == 2
        assert list(remaining) == []
        assert "Scan cancelled" in capsys.readouterr().out
    assert scanner.confirm_authorized("10.0.0.1", flag=True)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    assert scanner.confirm_authorized("10.0.0.1")


def test_flag_skips_the_question(scanner, monkeypatch):
    prompts = []
    answers = {"Enter target": "127.0.0.1", "Enter start ": "1", "Enter end po": "1",
               "Select proto": "tcp"}

    def answer(prompt=""):
        prompts.append(prompt)
        return answers.get(prompt[:12], "n")

    monkeypatch.setattr(builtins, "input", answer)
    monkeypatch.setattr(scanner, "ping_host", lambda host, timeout=1: (True, 128))
    monkeypatch.setattr(scanner, "scan_ports", lambda *args, **kwargs: [])
    assert scanner.main(["--i-am-authorized"]) == 0
    assert not any("authorised" in p for p in prompts)


def test_caps_enforced(scanner, monkeypatch, no_network, capsys):
    limits = ScanConfig()
    assert (limits.max_hosts, limits.max_probes) == (256, 65_536)
    assert len(targets.parse_target("192.168.1.0/24", limits.max_hosts)) == 254
    with pytest.raises(ValueError, match="TOOLKIT_SCAN_MAX_HOSTS"):
        targets.parse_target("10.0.0.0/16", limits.max_hosts)
    with pytest.raises(ValueError, match="TOOLKIT_SCAN_MAX_HOSTS"):
        targets.parse_target("2001:db8::/64", limits.max_hosts)  # never lists 2**64 hosts
    assert targets.check_limits(1, 65_535, limits) is None
    assert "TOOLKIT_SCAN_MAX_PROBES" in targets.check_limits(1, 2 * 65_535, limits)
    assert "TOOLKIT_SCAN_MAX_PROBES" in targets.check_limits(254, 254 * 1024, limits)
    assert "TOOLKIT_SCAN_MAX_HOSTS" in targets.check_limits(300, 300, limits)
    # refused before the authorization question and before any packet
    _answers(monkeypatch, ["192.168.1.0/24", "1", "1024", "tcp"])
    assert scanner.main([]) == 2
    assert "above the limit" in capsys.readouterr().out
    _answers(monkeypatch, ["10.0.0.0/8"])
    assert scanner.main([]) == 2


def test_limits_from_environment():
    assert scan_limits({"TOOLKIT_SCAN_MAX_HOSTS": "1024"}).max_hosts == 1024
    assert scan_limits({}).max_probes == 65_536
    for bad in ("0", "-5", "many"):
        with pytest.raises(ValueError, match="TOOLKIT_SCAN_MAX_PROBES"):
            scan_limits({"TOOLKIT_SCAN_MAX_PROBES": bad})
