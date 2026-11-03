"""F6d: the scan history from the command line. ``scan --history`` lists the saved scans (newest
first, 50 at a time, ``--before`` for the older ones), ``--show`` prints one as a scan prints
its results, with the command that scans it again, ``--compare A B`` prints two compared as the
web page does, and ``--delete`` names the record and asks first. They read the history only:
nothing is scanned or sent, and they go with no TARGET and no scan flag."""

import pytest

from toolkit.cli.controllers import scan_history
from toolkit.cli.ui import scans as scans_screen
from toolkit.core.data.recon import scan_records

from .test_saved_scans import A, B, C, entry, record, save

XSS = "<img src=x onerror=alert(1)>\x1b[31m"


@pytest.fixture
def nothing_sent(scanner, scan_service, monkeypatch):
    """A scan or a lookup fails the test: the history is read only."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("something was scanned or sent")
    monkeypatch.setattr(scan_service, "run_scan", refuse)
    monkeypatch.setattr(scanner, "run_scan", refuse)
    monkeypatch.setattr(scan_service, "ping_host", refuse)


def _run(scanner, capsys, *argv):
    code = scanner.main(list(argv))
    return code, capsys.readouterr().out


def test_no_saved_scans_yet(scanner, capsys, nothing_sent):
    code, out = _run(scanner, capsys, "--history")
    assert code == 0 and out.strip() == scans_screen.NO_SCANS


def test_the_list_newest_first(scanner, capsys, nothing_sent):
    save(A)
    save(B, entries=[entry(22, "HIGH", unconfirmed=True)], source="192.168.56.20")
    code, out = _run(scanner, capsys, "--history")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith(f"1. {B}  2026-11-01 10:00 UTC  10.0.0.4/31  ports 22,80-82 "
                               "(4, TCP, built-in)")
    assert "hosts: 1 scanned, 1 skipped" in lines[0] and "open ports: 1" in lines[0]
    assert "worst: HIGH (unconfirmed)" in lines[0] and lines[0].endswith("from 192.168.56.20")
    assert lines[1].startswith(f"2. {A}") and "worst: NO KNOWN CVE" in lines[1]


def test_older_scans_page_by_page(scanner, capsys, nothing_sent, monkeypatch):
    real = scan_history.saved_scans
    monkeypatch.setattr(scan_history, "saved_scans", lambda before=None: real(before, limit=2))
    for scan_id in (A, B, C):
        save(scan_id)
    code, out = _run(scanner, capsys, "--history")
    assert [line[3:28] for line in out.splitlines()[:2]] == [C, B]
    assert out.splitlines()[-1] == f"Older scans: add --before {B}"
    code, out = _run(scanner, capsys, "--history", "--before", B)
    assert code == 0 and out.splitlines()[0].startswith(f"1. {A}")


def test_one_scan_as_a_scan_prints_it_with_the_command_to_scan_again(scanner, capsys,
                                                                      nothing_sent):
    save(A, entries=[entry(22, "HIGH")], reputation={"10.0.0.5": {"abuseConfidenceScore": "0",
                                                                   "totalReports": "0"}},
         engine="nmap", nmap={"service_detection": "light", "timing": "aggressive",
                              "scripts": "safe", "os_detect": False},
         choices={"scan_unresponsive": True, "nvd": False, "ai": False, "abuseipdb": False})
    code, out = _run(scanner, capsys, "--show", A)
    assert code == 0
    for line in (f"Scan {A}: started 2026-11-01 10:00 UTC, asked from this computer",
                 "Target: 10.0.0.4/31 (10.0.0.4, 10.0.0.5)", "Ports: 22,80-82 (4 ports, TCP)",
                 "Engine: nmap: service detection light, timing aggressive, the safe scripts",
                 "  NVD (CVEs, services.nvd.nist.gov): on",
                 "  10.0.0.4: skipped: it didn't answer ping",
                 "  10.0.0.5: scanned; OS: Linux; AbuseIPDB: confidence of abuse 0%, 0 report(s)",
                 "Port: 22/tcp", "Severity (from CVSS): HIGH",
                 "Scanned 2 host(s): 1 open port(s).",
                 "Requests sent online: NVD 1, AI (OpenRouter) 0, AbuseIPDB 0",
                 "To scan it again: python -m toolkit scan 10.0.0.4/31 -p 22,80-82 "
                 "--protocol tcp --engine nmap --service-detection light --timing aggressive "
                 "--scripts safe --scan-unresponsive --no-nvd --no-ai --no-abuseipdb"):
        assert line in out.splitlines(), line


def test_network_text_is_printed_on_one_line_and_cleaned(scanner, capsys, nothing_sent):
    save(A, entries=[{**entry(22), "banner": XSS + "\nforged line"}],
         hosts=[{"target": "10.0.0.5", "state": "failed", "os": "", "message": XSS}],
         addresses=["10.0.0.5"], target="10.0.0.5")
    code, out = _run(scanner, capsys, "--show", A)
    assert code == 0 and "\x1b" not in out and "\nforged line" not in out
    assert "  10.0.0.5: the engine couldn't scan it (<img src=x onerror=alert(1)>)" in out


def test_two_scans_compared(scanner, capsys, nothing_sent):
    one = {"target": "10.0.0.5", "addresses": ["10.0.0.5"], "ports": "22,80,443",
           "hosts": [{"target": "10.0.0.5", "state": "scanned", "os": "", "message": ""}]}
    save(A, entries=[entry(22), entry(80)], **one)
    save(B, entries=[{**entry(22), "software": "OpenSSH 9.6"}, entry(443)], **one)
    code, out = _run(scanner, capsys, "--compare", B, A)  # either order: A is the older
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == f"A, the older: scan {A}, 2026-11-01 10:00 UTC, 10.0.0.5, ports 22,80,443 (TCP)"
    assert lines[1] == f"B, the newer: scan {B}, 2026-11-01 10:00 UTC, 10.0.0.5, ports 22,80,443 (TCP)"
    assert ("1 port opened, 1 closed, 1 changed, 0 the same; CVEs: 0 added, 0 gone") in out
    assert "  22/tcp   changed: software" in out
    assert "  80/tcp   closed" in out and "  443/tcp  opened" in out


def test_compare_says_what_it_could_not_compare(scanner, capsys, nothing_sent):
    save(A)
    save(B, target="10.0.0.5", addresses=["10.0.0.5"],
         hosts=[{"target": "10.0.0.5", "state": "skipped", "os": "", "message": ""}])
    code, out = _run(scanner, capsys, "--compare", A, B)
    assert code == 0
    assert "The two scans had different targets: only the addresses both scanned are compared." in out
    assert "Only in A: 10.0.0.4 (not compared)." in out
    assert ("  10.0.0.5: not compared: in A it was scanned, in B skipped: it didn't answer "
            "ping.") in out


def test_delete_names_the_record_and_asks(scanner, capsys, nothing_sent, monkeypatch):
    save(A)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    code, out = _run(scanner, capsys, "--delete", A)
    assert code == 0 and f"removes its record, {A}.json" in out and "Nothing deleted." in out
    assert [p.stem for p in scan_records.list_scans()] == [A]
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    code, out = _run(scanner, capsys, "--delete", A)
    assert code == 0 and f"✅ Deleted scan {A}." in out
    assert scan_records.list_scans() == []


def test_a_record_that_cant_be_removed_is_exit_code_2(scanner, capsys, nothing_sent,
                                                      monkeypatch):
    save(A)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    def busy(path):
        raise PermissionError("in use")

    monkeypatch.setattr(scan_records, "remove_file", busy)
    code, out = _run(scanner, capsys, "--delete", A)
    assert code == 2 and f"{A}.json could not be removed" in out
    assert scan_records.list_scans()


@pytest.mark.parametrize("argv", [["--show", "../x"], ["--show", A], ["--delete", A],
                                  ["--compare", A, A], ["--history", "--before", "x"]])
def test_a_bad_or_missing_id_is_exit_code_2(scanner, capsys, nothing_sent, argv):
    code, out = _run(scanner, capsys, *argv)
    assert code == 2 and out.strip()


def test_a_file_that_isnt_a_scan_is_left_out_and_said(scanner, capsys, nothing_sent):
    save(A)
    (scan_records.history_dir() / f"{B}.json").write_text("{}", encoding="utf-8")
    code, out = _run(scanner, capsys, "--history")
    assert code == 0 and "1 file(s) in instance/scan_history are not scans this toolkit can show" in out


@pytest.mark.parametrize("argv", [["10.0.0.5", "--history"], ["--history", "-p", "22"],
                                  ["--before", A], ["--show", A, "--export", "csv"],
                                  ["--history", "--show", A]])
def test_the_history_flags_go_alone(scanner, nothing_sent, argv):
    with pytest.raises(SystemExit) as stopped:
        scanner.main(argv)
    assert stopped.value.code == 2


def test_a_record_kept_from_before_f6b_shows_too(scanner, capsys, nothing_sent):
    old = record(A)
    del old["choices"]
    scan_records.write_scan(A, old)
    code, out = _run(scanner, capsys, "--show", A)
    assert code == 0 and "To scan it again: python -m toolkit scan 10.0.0.4/31 -p 22,80-82 " \
        "--protocol tcp" in out
