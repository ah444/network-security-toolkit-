"""R3: lookups run one at a time, each with a timeout; a scan always finishes."""

import builtins

import requests

from toolkit.recon import intel as recon_intel
from toolkit.recon import scanner as recon_scanner
from toolkit.recon.intel import lookup


def test_scan_finishes_when_intel_times_out(scanner, fake_session, monkeypatch, capsys,
                                            no_nvd_wait):
    for variable, value in (("OPENROUTER_API_KEY", "k"), ("ABUSEIPDB_API_KEY", "k")):
        monkeypatch.setenv(variable, value)
    session = fake_session(get=requests.Timeout("read timed out"),
                           post=requests.Timeout("read timed out"))
    real_intel = recon_intel.Intel
    monkeypatch.setattr(scanner, "Intel",
                        lambda keys, use_ai=False: real_intel(keys, use_ai, session=session))
    open_ports = [recon_scanner.PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1"),
                  recon_scanner.PortResult(80, "tcp", "open", "http", "Apache/2.4.49 (Unix)")]
    monkeypatch.setattr(scanner, "scan_ports", lambda *args, **kwargs: open_ports)
    monkeypatch.setattr(scanner, "ping_host", lambda host, timeout=1: (True, 52))
    # public target, authorised, AI yes, verbose no, AbuseIPDB yes, export no
    answers = iter(["8.8.8.8", "22", "80", "tcp", "yes", "n", "y", "y", "n"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert scanner.main([]) == 0
    out = capsys.readouterr().out
    assert "AbuseIPDB: timed out" in out
    assert "Total open ports found: 2" in out
    assert "Failed lookups: 4" in out and "22/tcp NVD: timed out" in out
    assert "80/tcp AI: timed out" in out
    assert all(call[2]["timeout"] == lookup.INTEL_TIMEOUT for call in session.calls)


def test_lookup_errors_are_named(fake_session, fake_response, no_nvd_wait):
    cases = ((requests.ConnectionError("down"), "NVD: no connection"),
             (fake_response({}, status=403), "NVD: HTTP 403"),
             (fake_response(ValueError("not JSON")), "NVD: ValueError"))
    for answer, message in cases:
        found = recon_intel.Intel({}, session=fake_session(get=answer)).lookup("nginx 1.25")
        assert found["errors"] == [message] and found["severity"] == "UNKNOWN"
