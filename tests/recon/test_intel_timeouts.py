"""R3: every lookup has a timeout; a scan always finishes (parallel since step 1.4e)."""

import builtins

import requests

from toolkit.core.services.recon import intel as recon_intel
from toolkit.core.data.recon import scanner as recon_scanner
from toolkit.core.data.recon import lookup


def test_scan_finishes_when_intel_times_out(fake_ports, scan_service, scanner, fake_session,
                                            monkeypatch, capsys, no_nvd_wait):
    for variable, value in (("OPENROUTER_API_KEY", "k"), ("ABUSEIPDB_API_KEY", "k")):
        monkeypatch.setenv(variable, value)
    session = fake_session(get=requests.Timeout("read timed out"),
                           post=requests.Timeout("read timed out"))
    real_intel = recon_intel.Intel
    monkeypatch.setattr(scan_service, "Intel",
                        lambda keys, use_ai=False, stop=None, nvd=True:
                        real_intel(keys, use_ai, session=session, stop=stop, nvd=nvd))
    open_ports = [recon_scanner.PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1"),
                  recon_scanner.PortResult(80, "tcp", "open", "http", "Apache/2.4.49 (Unix)")]
    fake_ports(lambda *args, **kwargs: open_ports)
    monkeypatch.setattr(scan_service, "ping_host", lambda host, timeout=1: (True, 52))
    # public target, authorised, verbose no, NVD yes, AI yes, AbuseIPDB yes, export no
    answers = iter(["8.8.8.8", "22", "80", "tcp", "yes", "n", "y", "y", "y", "n"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert scanner.main([]) == 0
    out = capsys.readouterr().out
    assert "AbuseIPDB: timed out" in out
    assert "Total open ports found: 2" in out
    # HS4: a service that failed is not asked again in this scan, and the summary says why
    assert "Failed lookups: " in out and "/tcp NVD: timed out" in out
    [unavailable] = [line for line in out.splitlines() if line.startswith("Unavailable: ")]
    for part in ("AbuseIPDB (timed out)", "NVD (timed out)", "AI (timed out)"):
        assert part in unavailable
    assert all(call[2]["timeout"] == lookup.INTEL_TIMEOUT for call in session.calls)


def test_lookup_errors_are_named(fake_session, fake_response, no_nvd_wait):
    cases = ((requests.ConnectionError("down"), "NVD: no connection"),
             (fake_response({}, status=403), "NVD: the key was refused (HTTP 403)"),
             (fake_response(ValueError("not JSON")), "NVD: ValueError"))
    for answer, message in cases:
        found = recon_intel.Intel({}, session=fake_session(get=answer)).lookup("nginx 1.25")
        assert found["errors"] == [message] and found["severity"] == "NOT CHECKED"
