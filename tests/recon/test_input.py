"""B11: port validation; empty ranges never crash."""

import builtins

import pytest


def test_empty_range_no_crash(scanner, monkeypatch, capsys):
    assert list(scanner.parse_port_range("80", "80")) == [80]
    assert list(scanner.parse_port_range(" 1", "3 ")) == [1, 2, 3]
    for start, end, words in (("100", "1", "after the end port"), ("0", "10", "between 1"),
                              ("1", "70000", "between 1"), ("a", "10", "whole numbers"),
                              ("", "", "whole numbers")):
        with pytest.raises(ValueError, match=words):
            scanner.parse_port_range(start, end)
    assert scanner.scan_ports("127.0.0.1", range(5, 5), ["tcp"]) == []
    scanner.verbose_progress(0, scanner.ProgressCounter())  # no division by zero
    remaining = iter(["127.0.0.1", "100", "1"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(remaining))
    assert scanner.main([]) == 2
    assert "after the end port" in capsys.readouterr().out


def test_bad_targets(scanner):
    for text, words in (("", "No target"), ("10.0.0.0/33", "not a valid network"),
                        ("no-such-host.invalid", "Could not resolve")):
        with pytest.raises(ValueError, match=words):
            scanner.parse_target(text, 256)
    assert scanner.parse_target("10.0.0.7/32", 256) == ["10.0.0.7"]


def test_protocol_asked_again(scanner, monkeypatch, capsys):
    remaining = iter(["ftp", "BOTH"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(remaining))
    assert scanner.ask_protocols() == ["tcp", "udp"]
    assert "Please type TCP, UDP or both" in capsys.readouterr().out
