"""B6, step A3: the OS from the TTL, the banners and the open ports, with the reasons; a
conflict said, and this computer said (its ports may lead to a VM, a container or WSL)."""

import csv
import json

import pytest

from toolkit.core.data.recon import local, scan_files
from toolkit.core.data.recon.scanner import PortResult
from toolkit.core.services.recon import os_guess
from toolkit.core.services.recon.intel import Intel

UBUNTU_SSH = PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1")
UBUNTU_HTTP = PortResult(80, "tcp", "open", "http", "Apache/2.4.52 (Ubuntu)")
IIS = PortResult(80, "tcp", "open", "http", "Microsoft-IIS/10.0")
RPC = PortResult(135, "tcp", "open", "epmap?", "")
FREEBSD = PortResult(22, "tcp", "open", "ssh", "SSH-2.0-OpenSSH_9.6 FreeBSD-20240104")


def test_ttl_ranges():
    unix = os_guess.guess_os(64)[0]
    assert unix.startswith("Linux")
    assert os_guess.guess_os(52) == (unix, 12)  # 12 router hops away
    assert os_guess.guess_os(128) == ("Windows", 0)
    assert os_guess.guess_os(117) == ("Windows", 11)
    assert os_guess.guess_os(65)[0] == "Windows"
    assert "Cisco" in os_guess.guess_os(255)[0] and os_guess.guess_os(250)[1] == 5
    for impossible in (None, 0, -1, 256):
        assert os_guess.guess_os(impossible) is None
        assert os_guess.ttl_evidence(impossible) == []


def test_banner_evidence_names_the_ports():
    found = os_guess.banner_evidence([UBUNTU_SSH, UBUNTU_HTTP, IIS, RPC])
    assert found == [os_guess.Evidence("Linux", "Linux (Ubuntu)", "the banner of 22/tcp, 80/tcp"),
                     os_guess.Evidence("Windows", "Windows", "the banner of 80/tcp")]


@pytest.mark.parametrize(("banner", "says"), [
    ("SSH-2.0-OpenSSH_for_Windows_8.1", "Windows"),
    ("Microsoft-HTTPAPI/2.0", "Windows"),
    ("Apache/2.4.58 (Win64) OpenSSL/3.1.3", "Windows"),
    ("220 Microsoft FTP Service", "Windows"),
    ("SSH-2.0-OpenSSH_9.2p1 Debian-2+deb12u3", "Linux (Debian)"),
    ("Apache/2.4.6 (CentOS)", "Linux (CentOS)"),
    ("Apache/2.4.37 (Red Hat Enterprise Linux)", "Linux (Red Hat)"),
    ("SSH-2.0-OpenSSH_9.6 FreeBSD-20240104", "FreeBSD"),
    ("nginx/1.18.0", None),
    ("\x1b[31mUbuntu\x1b[0m <script>", "Linux (Ubuntu)"),  # says the table's words only
])
def test_banner_words(banner, says):
    found = os_guess.banner_evidence([PortResult(8000, "tcp", "open", "", banner)])
    assert [e.says for e in found] == ([says] if says else [])


def test_agree():
    assert os_guess.agree("Unix", "Linux") and os_guess.agree("BSD", "Unix")
    assert not os_guess.agree("Linux", "BSD") and not os_guess.agree("Windows", "Unix")


@pytest.mark.parametrize(("ttl", "ports", "verdict", "note"), [
    (64, [UBUNTU_SSH], "Linux (Ubuntu)", ""),
    (52, [UBUNTU_HTTP, UBUNTU_SSH], "Linux (Ubuntu)", ""),
    (128, [IIS, RPC], "Windows", ""),
    (128, [], "Windows", ""),
    (64, [], "Linux / macOS / BSD / other Unix", ""),
    (None, [FREEBSD], "FreeBSD", ""),
    (128, [UBUNTU_SSH], os_guess.CONFLICT, os_guess.CONFLICT_NOTE),
    (64, [RPC], os_guess.CONFLICT, os_guess.CONFLICT_NOTE),
    (None, [UBUNTU_SSH, FREEBSD], os_guess.CONFLICT, os_guess.CONFLICT_NOTE),
])
def test_os_hint(ttl, ports, verdict, note):
    hint = os_guess.os_hint(ttl, ports)
    assert (hint.verdict, hint.note, hint.this_computer) == (verdict, note, False)


def test_no_evidence_no_hint():
    assert os_guess.os_hint(None, [PortResult(8080, "tcp", "open", "http", "nginx")]) is None


def test_this_computer():
    hint = os_guess.os_hint(128, [UBUNTU_SSH], this_computer=True, local_os="Windows 10")
    assert (hint.verdict, hint.note, hint.this_computer) == (
        "this computer (Windows 10)", os_guess.THIS_COMPUTER_NOTE, True)
    plain = os_guess.os_hint(128, [RPC], this_computer=True, local_os="Windows 10")
    assert plain.note == "", "nothing disagrees with this computer's own OS"
    assert os_guess.os_hint(None, [], this_computer=True, local_os="Windows 10").evidence == ()


def test_summary_is_one_line():
    hint = os_guess.os_hint(128, [UBUNTU_SSH])
    assert hint.summary() == ("unsure: the evidence conflicts: Windows (TTL 128, about 0 hops "
                              "away); Linux (Ubuntu) (the banner of 22/tcp)")
    assert os_guess.OsHint("this computer (Windows 10)", ()).summary() == \
        "this computer (Windows 10)"


def test_this_computer_s_own_addresses():
    assert local.is_this_computer("127.0.0.1") and local.is_this_computer("::1")
    try:
        own = local._leaving_address("192.0.2.1")  # TEST-NET-1: never reached, nothing sent
    except OSError:
        pytest.skip("no route on this machine")
    assert local.is_this_computer(own)
    assert not local.is_this_computer("192.0.2.1")


def test_checking_an_address_sends_nothing(monkeypatch):
    calls = []

    class Recording:
        def __init__(self, *args):
            calls.append("socket")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def connect(self, address):
            calls.append(("connect", address))

        def getsockname(self):
            calls.append("getsockname")
            return ("192.168.1.11", 50000)

        def __getattr__(self, name):  # send, sendto, ...
            raise AssertionError(f"{name} called")

    monkeypatch.setattr(local.socket, "socket", Recording)
    assert local.is_this_computer("192.168.1.11")
    assert not local.is_this_computer("192.168.1.1")
    assert calls.count("getsockname") == 2 and ("connect", ("192.168.1.1", 9)) in calls


def test_this_os(monkeypatch):
    monkeypatch.setattr(local.platform, "system", lambda: "Windows")
    monkeypatch.setattr(local.platform, "release", lambda: "10")
    assert local.this_os() == "Windows 10"
    monkeypatch.setattr(local.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "arm64"))
    assert local.this_os() == "macOS 14.5"


class RecordingView:
    """Every ``ScanView`` call, in order."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args))


@pytest.mark.parametrize(("own", "expect"), [(False, os_guess.CONFLICT),
                                              (True, "this computer (Windows 10)")])
def test_the_scan_shows_the_hint_after_the_ports(scan_service, monkeypatch, own, expect):
    ports = [PortResult(80, "tcp", "open", "http", "Apache (Ubuntu)")]  # no version: no lookup
    monkeypatch.setattr(scan_service, "ping_host", lambda host: (True, 128))
    monkeypatch.setattr(scan_service, "scan_ports", lambda *args: list(ports))
    monkeypatch.setattr(scan_service, "is_this_computer", lambda address: own)
    monkeypatch.setattr(scan_service, "this_os", lambda: "Windows 10")
    request = scan_service.ScanRequest("10.0.0.5", ["10.0.0.5"], [80], ["tcp"])
    view = RecordingView()
    entries = scan_service.scan_target("10.0.0.5", request, Intel({}), view)
    names = [name for name, _ in view.calls]
    assert names.index("open_ports") < names.index("os_hint") < names.index("entry")
    hint = dict(view.calls)["os_hint"][1]
    assert hint.verdict == expect
    assert entries[0]["os_hint"] == hint.summary()


def test_no_open_ports_still_says_this_computer(scan_service, monkeypatch):
    monkeypatch.setattr(scan_service, "ping_host", lambda host: (True, 128))
    monkeypatch.setattr(scan_service, "scan_ports", lambda *args: [])
    monkeypatch.setattr(scan_service, "is_this_computer", lambda address: True)
    request = scan_service.ScanRequest("127.0.0.1", ["127.0.0.1"], [80], ["tcp"])
    view = RecordingView()
    assert scan_service.scan_target("127.0.0.1", request, Intel({}), view) == []
    assert [name for name, _ in view.calls][-2:] == ["no_open_ports", "os_hint"]


def test_the_report_files_carry_the_hint(tmp_path):
    entry = {"target": "10.0.0.5", "port": 22, "protocol": "tcp", "service": "ssh",
             "banner": "SSH-2.0-OpenSSH_8.9p1 Ubuntu", "software": "OpenSSH 8.9", "cves": [],
             "max_cvss": None, "severity": "NO KNOWN CVE", "ai_explanation": "",
             "lookup_errors": [], "os_hint": "Linux (Ubuntu): Linux (Ubuntu) (the banner of 22/tcp)"}
    saved = {fmt: scan_files.export_results([entry], fmt, tmp_path) for fmt in ("json", "csv", "txt")}
    assert json.loads(saved["json"].read_text(encoding="utf-8"))[0]["os_hint"] == entry["os_hint"]
    rows = list(csv.reader(saved["csv"].open(encoding="utf-8", newline="")))
    assert rows[0][-1] == "OS hint" and rows[1][-1] == entry["os_hint"]
    assert f"OS hint: {entry['os_hint']}\n" in saved["txt"].read_text(encoding="utf-8")
