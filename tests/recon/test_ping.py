"""B7: ping flags and timeouts per OS (all three OSes via the platform-patching helper);
ping by its full path, never from the current folder."""

import subprocess
from pathlib import Path

import pytest

from toolkit.core.data.recon import ping as recon_ping
from toolkit.core.services.recon import os_guess

# Windows lines: real output of `ping -n 1` on the project machine (Windows 10, 2026-10-20).
WINDOWS_V4 = b"Reply from 127.0.0.1: bytes=32 time<1ms TTL=128\r\n"
WINDOWS_V6 = b"Reply from ::1: time<1ms \r\n"
# Linux (iputils) and macOS (ping6) reply lines.
LINUX_V4 = b"64 bytes from 127.0.0.1: icmp_seq=1 ttl=64 time=0.045 ms\n"
MACOS_V6 = b"16 bytes from ::1, icmp_seq=0 hlim=64 time=0.1 ms\n"


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Windows' SystemRoot and the Unix folders in ``tmp_path``, with ping and ping6 in sbin."""
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    for name in ("ping", "ping6"):
        (tmp_path / "sbin").mkdir(exist_ok=True)
        (tmp_path / "sbin" / name).write_bytes(b"")
    monkeypatch.setattr(recon_ping, "UNIX_PLACES", (str(tmp_path / "bin"), str(tmp_path / "sbin")))
    return tmp_path


@pytest.mark.parametrize(("name", "program", "v4", "v6"), [
    ("windows", ("Windows", "System32", "PING.EXE"), ["-4", "-n", "1", "-w", "2000", "10.0.0.1"],
     ["-6", "-n", "1", "-w", "2000", "::1"]),
    ("linux", ("sbin", "ping"), ["-4", "-c", "1", "-W", "2", "10.0.0.1"],
     ["-6", "-c", "1", "-W", "2", "::1"]),
    ("darwin", ("sbin", "ping"), ["-c", "1", "-t", "2", "10.0.0.1"], ["-c", "1", "::1"]),
])
def test_ping_args_per_os(fake_platform, places, name, program, v4, v6):
    fake_platform(name)
    ping = str(places.joinpath(*program))
    assert recon_ping.ping_command("10.0.0.1", timeout=2) == [ping, *v4]
    v6_program = ping + "6" if name == "darwin" else ping
    assert recon_ping.ping_command("::1", timeout=2) == [v6_program, *v6]
    assert recon_ping.ping_command("10.0.0.1", timeout=0)[-2] in ("1", "1000")  # at least 1 s


def test_ping_is_never_looked_up_by_name(fake_platform, places, monkeypatch):
    """A planted ping in the current folder is never run: the command starts with the full
    path; with no ping in the known places there is no ping, not the planted one."""
    monkeypatch.chdir(places)
    (places / "ping").write_bytes(b"")
    (places / "PING.EXE").write_bytes(b"")
    for system in ("windows", "linux", "darwin"):
        fake_platform(system)
        program = Path(recon_ping.ping_command("10.0.0.1")[0])
        assert program.is_absolute() and program.parent != places
    fake_platform("linux")
    monkeypatch.setattr(recon_ping, "UNIX_PLACES", (str(places / "nothing"),))
    with pytest.raises(FileNotFoundError, match="no ping in"):
        recon_ping.ping_program()
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: ran.append(args))
    assert recon_ping.ping_host("10.0.0.1") == (False, None) and ran == []
    fake_platform("windows")
    monkeypatch.setenv("SystemRoot", "")
    with pytest.raises(FileNotFoundError, match="SystemRoot is not set"):
        recon_ping.ping_program()


def test_ttl_parsed_from_each_os():
    assert recon_ping.parse_ttl(WINDOWS_V4) == 128
    assert recon_ping.parse_ttl(WINDOWS_V6) is None  # Windows shows no TTL for IPv6
    assert recon_ping.parse_ttl(LINUX_V4) == 64
    assert recon_ping.parse_ttl(MACOS_V6) == 64


def test_ping_timeout_and_missing_program(monkeypatch):
    seen = {}

    def slow(command, **kwargs):
        seen.update(kwargs, command=command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", slow)
    assert recon_ping.ping_host("10.0.0.1", timeout=1) == (False, None)
    assert seen["timeout"] > 1, "the process gets time to start, beyond the ping wait"
    assert Path(seen["command"][0]).is_absolute() and seen["stdin"] == subprocess.DEVNULL

    def missing(command, **kwargs):
        raise FileNotFoundError("ping")

    monkeypatch.setattr(subprocess, "run", missing)
    assert recon_ping.ping_host("10.0.0.1") == (False, None)


def test_unreachable_reply_is_not_alive(monkeypatch):
    done = subprocess.CompletedProcess(
        [], 0, b"Reply from 10.0.0.9: Destination host unreachable.\r\n", b"")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: done)
    assert recon_ping.ping_host("10.0.0.1") == (False, None)


def _has_ping():
    try:
        return recon_ping.ping_program().is_file()
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _has_ping(), reason="no ping program")
def test_real_ping_of_this_computer():
    alive, ttl = recon_ping.ping_host("127.0.0.1")
    assert alive and os_guess.guess_os(ttl) is not None
