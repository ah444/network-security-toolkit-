"""B7: ping flags and timeouts per OS (all three OSes via the platform-patching helper)."""

import shutil
import subprocess

import pytest

# Windows lines: real output of `ping -n 1` on the project machine (Windows 10, 2026-10-20).
WINDOWS_V4 = b"Reply from 127.0.0.1: bytes=32 time<1ms TTL=128\r\n"
WINDOWS_V6 = b"Reply from ::1: time<1ms \r\n"
# Linux (iputils) and macOS (ping6) reply lines.
LINUX_V4 = b"64 bytes from 127.0.0.1: icmp_seq=1 ttl=64 time=0.045 ms\n"
MACOS_V6 = b"16 bytes from ::1, icmp_seq=0 hlim=64 time=0.1 ms\n"


@pytest.mark.parametrize(("name", "v4", "v6"), [
    ("windows", ["ping", "-4", "-n", "1", "-w", "2000", "10.0.0.1"],
     ["ping", "-6", "-n", "1", "-w", "2000", "::1"]),
    ("linux", ["ping", "-4", "-c", "1", "-W", "2", "10.0.0.1"],
     ["ping", "-6", "-c", "1", "-W", "2", "::1"]),
    ("darwin", ["ping", "-c", "1", "-t", "2", "10.0.0.1"], ["ping6", "-c", "1", "::1"]),
])
def test_ping_args_per_os(scanner, fake_platform, name, v4, v6):
    fake_platform(name)
    assert scanner.ping_command("10.0.0.1", timeout=2) == v4
    assert scanner.ping_command("::1", timeout=2) == v6
    assert scanner.ping_command("10.0.0.1", timeout=0)[-2] in ("1", "1000")  # at least 1 s


def test_ttl_parsed_from_each_os(scanner):
    assert scanner.parse_ttl(WINDOWS_V4) == 128
    assert scanner.parse_ttl(WINDOWS_V6) is None  # Windows shows no TTL for IPv6
    assert scanner.parse_ttl(LINUX_V4) == 64
    assert scanner.parse_ttl(MACOS_V6) == 64


def test_ping_timeout_and_missing_program(scanner, monkeypatch):
    seen = {}

    def slow(command, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", slow)
    assert scanner.ping_host("10.0.0.1", timeout=1) == (False, None)
    assert seen["timeout"] > 1, "the process gets time to start, beyond the ping wait"

    def missing(command, **kwargs):
        raise FileNotFoundError("ping")

    monkeypatch.setattr(subprocess, "run", missing)
    assert scanner.ping_host("10.0.0.1") == (False, None)


def test_unreachable_reply_is_not_alive(scanner, monkeypatch):
    done = subprocess.CompletedProcess(
        [], 0, b"Reply from 10.0.0.9: Destination host unreachable.\r\n", b"")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: done)
    assert scanner.ping_host("10.0.0.1") == (False, None)


@pytest.mark.skipif(shutil.which("ping") is None, reason="no ping program")
def test_real_ping_of_this_computer(scanner):
    alive, ttl = scanner.ping_host("127.0.0.1")
    assert alive and scanner.guess_os(ttl) is not None
