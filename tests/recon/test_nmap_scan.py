"""Step N4: the scan with ``--engine nmap``: the flags and questions, the stop before anything is
sent when nmap can't do the scan, the opening notice, nmap's product and version as the A1
names for NVD, nmap's OS match in the OS hint, and the report's engine and CPE columns. No
nmap is run: it is found and its XML read from the real fixtures of this PC."""

import builtins
import csv
import json
from pathlib import Path

import pytest

from toolkit.core.data.recon import scan_files, software
from toolkit.core.data.recon.nmap import NEEDS_PRIVILEGES, VERSION_SCRIPTS_NOTE, Nmap, NmapOptions
from toolkit.core.data.recon.nmap import NmapMissing
from toolkit.core.data.recon.nmap_run import NmapFailed
from toolkit.core.data.recon.scanner import PortResult
from toolkit.core.services.recon import engines, os_guess
from toolkit.core.services.recon.intel import Intel

NMAP = Nmap(Path("/opt/nmap/nmap"), (7, 992))
NO_KEYS = {"openrouter": "", "abuseipdb": "", "nvd": ""}


@pytest.mark.parametrize(("product", "version", "expect"), [
    ("OpenSSH", "for_Windows_8.1", "OpenSSH 8.1"),
    ("OpenSSH", "8.9p1 Ubuntu 3ubuntu0.10", "OpenSSH 8.9"),
    ("MariaDB", "5.5.5-10.6.12", "MariaDB 10.6.12"),
    ("Apache httpd", "2.4.49", "Apache 2.4.49"),
    ("Microsoft IIS httpd", "10.0", "Microsoft-IIS 10.0"),
    ("Redis key-value store", "7.2.4", "Redis 7.2.4"),
    ("MySQL", "5.0.51a-3ubuntu5", "MySQL 5.0.51"),
    ("Some Device ftpd", "1.2", "Some Device ftpd 1.2"),  # nmap's own name, short and plain
    ("Microsoft Windows RPC", "", None),  # no version: nothing to look up
    ("nginx", "192.168.1.1", None),  # an address is never sent
    ("evil;rm -rf", "1.0", None),
    ("x" * 41, "1.0", None),
])
def test_nmap_names_are_the_lookup_names(product, version, expect):
    assert software.nmap_software(product, version) == expect


@pytest.mark.parametrize("described", ["OpenSSH for_Windows_8.1 (protocol 2.0)",
                                       "Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)",
                                       "Microsoft IIS httpd 10.0"])
def test_nmaps_descriptions_name_windows(described):
    evidence = os_guess.banner_evidence([PortResult(80, "tcp", "open", "http", described)])
    assert [e.family for e in evidence] == ["Windows"]


def test_nmap_os_match_is_the_hint():
    hint = os_guess.os_hint(128, [], nmap_os=(("Microsoft Windows 10 1809 - 21H2", 100),
                                              ("Microsoft Windows 11", 90)))
    assert hint.verdict == "Microsoft Windows 10 1809 - 21H2"
    assert hint.evidence[-1] == os_guess.Evidence(
        "Windows", "Microsoft Windows 10 1809 - 21H2", "nmap OS detection, a 100% match")
    conflict = os_guess.os_hint(64, [], nmap_os=(("Microsoft Windows 10", 96),))
    assert conflict.verdict == os_guess.CONFLICT  # a Unix TTL and a Windows match disagree
    assert os_guess.os_hint(None, [], nmap_os=()) is None


@pytest.fixture
def nmap_here(monkeypatch):
    """nmap found (``NMAP``) and able to send raw packets or not (``privileged``)."""
    state = {"privileged": False}
    monkeypatch.setattr(engines, "find_nmap", lambda: NMAP)
    monkeypatch.setattr(engines, "has_privileges", lambda: state["privileged"])
    return state


def test_engine_problems(nmap_here):
    assert engines.engine_problem("builtin", NmapOptions(os_detect=True), ["udp"]) is None
    assert engines.engine_problem("nmap", NmapOptions(), ["tcp"]) is None
    assert engines.engine_problem("nmap", NmapOptions(), ["tcp", "udp"]) == NEEDS_PRIVILEGES
    assert engines.engine_problem("nmap", NmapOptions(os_detect=True), ["tcp"]) == \
        NEEDS_PRIVILEGES
    nmap_here["privileged"] = True
    assert engines.engine_problem("nmap", NmapOptions(os_detect=True), ["udp"]) is None


def test_no_nmap_is_a_problem_with_the_way_to_install_it():
    problem = engines.engine_problem("nmap", NmapOptions(), ["tcp"])
    assert "TOOLKIT_NMAP_PATH" in problem  # the tests' own: nmap is never found in a test


def test_the_notice_names_the_engine_and_nmaps_version_scripts(nmap_here):
    assert engines.describe("builtin", NmapOptions())[0].startswith("built-in")
    lines = engines.describe("nmap", NmapOptions("all", "polite", "safe", True))
    assert lines[0] == f"nmap 7.992 ({NMAP.path})"
    assert lines[1].startswith("raw packets: no")
    assert "--version-intensity 9" in lines[2] and "-T2" in lines[3]
    assert "the toolkit's 8 (banner," in lines[4] and lines[5] == "OS detection: on (-O)"
    assert lines[-1] == VERSION_SCRIPTS_NOTE and "iec61850-mms" in lines[-1]
    assert "intrusive" in lines[-1]
    nmap_here["privileged"] = True
    assert engines.describe("nmap", NmapOptions())[1] == "raw packets: yes (SYN scan)"


def test_the_notice_survives_nmap_going_away(monkeypatch):
    def gone():
        raise NmapMissing("nmap was not found.")

    monkeypatch.setattr(engines, "find_nmap", gone)
    assert engines.describe("nmap", NmapOptions()) == ["nmap: nmap was not found."]


class Recorder:
    """Every ``ScanView`` call, in order; no question is answered."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args))


@pytest.fixture
def nmap_xml(nmap_here, monkeypatch, scan_service, fixtures_dir):
    """The nmap engine reading ``nmap_<name>.xml``, its commands kept; ping answers."""
    runs = {"commands": [], "xml": "connect_scripts"}

    def run(command, progress, stop, time_limit):
        runs["commands"].append(command)
        return (fixtures_dir / f"nmap_{runs['xml']}.xml").read_bytes()

    monkeypatch.setattr(engines, "run_nmap", run)
    monkeypatch.setattr(scan_service, "ping_host", lambda host: (True, 128))
    monkeypatch.setattr(scan_service, "is_this_computer", lambda address: False)
    return runs


class NoLookups(Intel):
    """What would be looked up; nothing is sent."""

    def __init__(self):
        super().__init__({})
        self.asked = []

    def lookup_many(self, products):
        self.asked.extend(products)
        return {p: {"cves": [], "cve_source": "", "max_cvss": None, "severity": "UNKNOWN",
                    "ai_explanation": "", "errors": []} for p in products}


def test_nmap_products_go_to_nvd_as_the_a1_names(nmap_xml, scan_service):
    request = scan_service.ScanRequest("127.0.0.1", ["127.0.0.1"],
                                       [1, 135, 445, 2222, 3306, 5357, 11211], ["tcp"],
                                       engine="nmap", nmap_options=NmapOptions(scripts="safe"))
    intel, view = NoLookups(), Recorder()
    entries = scan_service.scan_target("127.0.0.1", request, intel, view)
    assert "--script" in nmap_xml["commands"][0]
    assert intel.asked == ["OpenSSH 8.1", "MariaDB 10.6.12", "Microsoft-HTTPAPI 2.0"]
    by_port = {e["port"]: e for e in entries}
    assert {e["engine"] for e in entries} == {"nmap"}
    assert by_port[2222]["software"] == "OpenSSH 8.1"
    assert by_port[2222]["cpe"] == ["cpe:/a:openbsd:openssh:for_windows_8.1"]
    assert by_port[445]["software"] == "" and by_port[445]["cpe"] == []


def test_nmaps_os_match_reaches_the_report(nmap_xml, scan_service):
    nmap_xml["xml"] = "syn_udp_os"
    request = scan_service.ScanRequest("127.0.0.1", ["127.0.0.1"], [1, 2, 3, 4, 5, 135, 2222],
                                       ["tcp"], engine="nmap")
    view = Recorder()
    entries = scan_service.scan_target("127.0.0.1", request, NoLookups(), view)
    hint = dict(view.calls)["os_hint"][1]
    assert hint.verdict == "Microsoft Windows 10 1809 - 21H2"
    assert entries[0]["os_hint"] == hint.summary()


def test_the_report_files_carry_the_engine_and_the_cpe(tmp_path):
    entry = {"target": "127.0.0.1", "port": 2222, "protocol": "tcp", "service": "ssh",
             "banner": "SSH-2.0-OpenSSH_for_Windows_8.1", "software": "OpenSSH 8.1",
             "cves": [], "cve_source": "", "max_cvss": None, "severity": "NO KNOWN CVE",
             "ai_explanation": "", "lookup_errors": [], "os_hint": "", "engine": "nmap",
             "cpe": ["cpe:/a:openbsd:openssh:for_windows_8.1", "cpe:/o:microsoft:windows"]}
    saved = {fmt: scan_files.export_results([entry], fmt, tmp_path)
             for fmt in ("json", "csv", "txt")}
    assert json.loads(saved["json"].read_text(encoding="utf-8"))[0]["cpe"] == entry["cpe"]
    header, row = list(csv.reader(saved["csv"].open(encoding="utf-8", newline="")))
    assert header[-2:] == ["Engine", "CPE (nmap)"]
    assert row[-2:] == ["nmap", " ".join(entry["cpe"])]
    txt = saved["txt"].read_text(encoding="utf-8")
    assert f"Engine: nmap\nCPE (nmap): {' '.join(entry['cpe'])}\n" in txt


def test_a_target_nmap_cant_scan_is_said_and_the_next_is_scanned(nmap_here, monkeypatch,
                                                                 scan_service):
    scanned = []

    def scan_target(target, request, intel, view):
        scanned.append(target)
        if target == "10.0.0.1":
            raise NmapFailed("nmap stopped with exit code 1: x")
        return []

    monkeypatch.setattr(scan_service, "scan_target", scan_target)
    request = scan_service.ScanRequest("10.0.0.0/31", ["10.0.0.1", "10.0.0.2"], [22], ["tcp"],
                                       engine="nmap")
    view = Recorder()
    assert scan_service.run_scan(request, NO_KEYS, 0, view) == 1
    assert scanned == ["10.0.0.1", "10.0.0.2"]
    names = [name for name, _ in view.calls]
    assert names[:2] == ["engine_notice", "lookups_notice"]
    assert ("scan_failed", ("10.0.0.1", "nmap stopped with exit code 1: x")) in view.calls


def test_the_console_shows_the_engine_and_the_failure(scan_screen, capsys):
    view = scan_screen.ConsoleScan()
    view.engine_notice(["nmap 7.992 (/opt/nmap/nmap)", "timing: normal (-T3)"])
    view.scan_failed("10.0.0.1", "nmap was not found.")
    assert capsys.readouterr().out == ("Scan engine:\n  nmap 7.992 (/opt/nmap/nmap)\n"
                                       "  timing: normal (-T3)\n"
                                       "Could not scan 10.0.0.1: nmap was not found.\n")


@pytest.fixture
def no_scan(scanner, scan_service, monkeypatch):
    """The command up to the scan: what reached ``run_scan``, every question asked, and no
    packet (the ping fails the test)."""
    seen = {"requests": [], "prompts": [], "answers": []}

    def fake_input(prompt=""):
        seen["prompts"].append(prompt)
        if not seen["answers"]:
            raise AssertionError(f"asked {prompt!r}")
        return seen["answers"].pop(0)

    def no_ping(host):
        raise AssertionError("pinged before the scan was allowed")

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(scan_service, "ping_host", no_ping)
    monkeypatch.setattr(scanner, "load_api_keys", lambda: dict(NO_KEYS))
    monkeypatch.setattr(scanner, "run_scan",
                        lambda request, keys, start, view: seen["requests"].append(request) or 0)
    return seen


@pytest.mark.parametrize("flag", [["--service-detection", "all"], ["--timing", "polite"],
                                  ["--scripts", "safe"], ["--os-detect"]])
def test_the_nmap_flags_need_the_nmap_engine(scanner, no_scan, capsys, flag):
    with pytest.raises(SystemExit) as stopped:
        scanner.main(["127.0.0.1", *flag, "--i-am-authorized"])
    assert stopped.value.code == 2
    assert f"{flag[0]}: only with --engine nmap" in capsys.readouterr().err


def test_no_nmap_stops_before_the_authorization(scanner, no_scan, capsys):
    assert scanner.main(["127.0.0.1", "--engine", "nmap"]) == 2
    out = capsys.readouterr().out
    assert "The nmap engine can't run this scan: " in out and "TOOLKIT_NMAP_PATH" in out
    assert no_scan["prompts"] == [] and no_scan["requests"] == []  # nothing asked, nothing sent


def test_udp_without_raw_packets_stops_before_the_authorization(scanner, no_scan, nmap_here,
                                                                capsys):
    assert scanner.main(["127.0.0.1", "--engine", "nmap", "--protocol", "both"]) == 2
    assert NEEDS_PRIVILEGES in capsys.readouterr().out
    assert no_scan["prompts"] == []


def test_the_flags_make_the_nmap_request(scanner, no_scan, nmap_here):
    nmap_here["privileged"] = True
    assert scanner.main(["127.0.0.1", "--engine", "NMAP", "--service-detection", "light",
                         "--timing", "aggressive", "--scripts", "safe", "--os-detect",
                         "--i-am-authorized"]) == 0
    request = no_scan["requests"][0]
    assert request.engine == "nmap"
    assert request.nmap_options == NmapOptions("light", "aggressive", "safe", True)


def test_the_flags_default_to_the_builtin_engine(scanner, no_scan):
    assert scanner.main(["127.0.0.1", "--i-am-authorized"]) == 0
    assert no_scan["requests"][0].engine == "builtin"
    assert no_scan["requests"][0].nmap_options == NmapOptions()


START = ["127.0.0.1", "22", "22", "tcp"]  # the target, the ports and the protocol
AFTER = ["yes", "n"]  # the authorization, then no progress line


def test_no_engine_question_without_nmap(scanner, no_scan):
    no_scan["answers"] = [*START, *AFTER]
    assert scanner.main([]) == 0
    assert not any("engine" in prompt for prompt in no_scan["prompts"])
    assert no_scan["requests"][0].engine == "builtin"


def test_the_questions_make_the_nmap_request(scanner, no_scan, monkeypatch):
    monkeypatch.setattr(engines.NmapEngine, "available", lambda self: None)
    monkeypatch.setattr(engines, "find_nmap", lambda: NMAP)
    monkeypatch.setattr(engines, "has_privileges", lambda: True)
    monkeypatch.setattr(scanner, "has_privileges", lambda: True)
    no_scan["answers"] = [*START, "maybe", "nmap", "", "y", "y", *AFTER]
    assert scanner.main([]) == 0
    request = no_scan["requests"][0]
    assert request.engine == "nmap" and request.interactive
    assert request.nmap_options == NmapOptions("normal", "normal", "safe", True)
    asked = no_scan["prompts"][4:9]
    assert asked[0] == asked[1] and "(builtin/nmap, Enter for builtin)" in asked[0]
    assert "(light/normal/all, Enter for normal)" in asked[2]
    assert "8 nmap scripts (banner," in asked[3] and "detect the OS" in asked[4]


def test_no_os_question_without_raw_packets(scanner, no_scan, monkeypatch):
    monkeypatch.setattr(engines.NmapEngine, "available", lambda self: None)
    monkeypatch.setattr(engines, "find_nmap", lambda: NMAP)
    monkeypatch.setattr(engines, "has_privileges", lambda: False)
    monkeypatch.setattr(scanner, "has_privileges", lambda: False)
    no_scan["answers"] = [*START, "nmap", "all", "n", *AFTER]
    assert scanner.main([]) == 0
    assert no_scan["requests"][0].nmap_options == NmapOptions("all", "normal", "", False)
    assert not any("detect the OS" in prompt for prompt in no_scan["prompts"])


def test_builtin_is_the_answer_to_enter(scanner, no_scan, monkeypatch):
    monkeypatch.setattr(engines.NmapEngine, "available", lambda self: None)
    no_scan["answers"] = [*START, "", *AFTER]
    assert scanner.main([]) == 0
    assert no_scan["requests"][0].engine == "builtin"


def test_print_entry_shows_the_cpe(scan_screen, capsys):
    port = PortResult(2222, "tcp", "open", "ssh", "x", cpe=("cpe:/a:openbsd:openssh:8.1",))
    from toolkit.core.services.recon import report
    entry = report.make_entry("127.0.0.1", port, None, {
        "cves": [], "cve_source": "", "max_cvss": None, "severity": "UNKNOWN",
        "ai_explanation": "", "errors": []}, engine="nmap")
    scan_screen.print_entry(entry)
    out = capsys.readouterr().out
    assert "CPE (nmap): cpe:/a:openbsd:openssh:8.1\n" in out
    assert "Software: not recognised by nmap or in the banner (no CVE lookup)" in out
