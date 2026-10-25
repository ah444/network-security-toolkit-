"""Ping a host and read the reply's TTL (B6).

The ping flags and timeouts follow the OS (``platform.system()``). What the TTL says
about the host is ``toolkit.core.services.recon.os_guess``.

ping is run by its full path (``ping_program``), never by its name: Windows looks for a bare
name in Python's folder and in the current folder before System32, so a ``ping.exe`` planted
there would run.
"""

import ipaddress
import os
import platform
import re
import subprocess
from pathlib import Path

TTL_PATTERN = re.compile(rb"(?:ttl|hlim)[=:]\s*(\d+)", re.IGNORECASE)
UNIX_PLACES = ("/bin", "/usr/bin", "/sbin", "/usr/sbin")  # Linux, macOS (/sbin), the BSDs


def ping_program(name: str = "ping") -> Path:
    """``name``'s full path: on Windows ``PING.EXE`` in System32 (from ``SystemRoot``), else the
    first of ``UNIX_PLACES`` that has it. FileNotFoundError when there is none."""
    if platform.system().lower() == "windows":
        root = Path(os.environ.get("SystemRoot", ""))  # any case on Windows
        if root.is_absolute():
            return root / "System32" / "PING.EXE"
        raise FileNotFoundError("SystemRoot is not set: no System32 to run ping from")
    for folder in UNIX_PLACES:
        if (Path(folder) / name).is_file():
            return Path(folder) / name
    raise FileNotFoundError(f"no {name} in {', '.join(UNIX_PLACES)}")


def ping_command(host: str, timeout: float = 1) -> list[str]:
    """One echo request with this OS's flags, waiting at most ``timeout`` seconds; the
    program by its full path (FileNotFoundError when there is none)."""
    system = platform.system().lower()
    ipv6 = ipaddress.ip_address(host).version == 6
    seconds = max(1, int(timeout))
    if system == "windows":
        return [str(ping_program()), "-6" if ipv6 else "-4", "-n", "1", "-w",
                str(seconds * 1000), host]
    if system == "darwin":
        if ipv6:  # macOS ping6 has no overall timeout flag; the subprocess timeout applies
            return [str(ping_program("ping6")), "-c", "1", host]
        return [str(ping_program()), "-c", "1", "-t", str(seconds), host]
    return [str(ping_program()), "-6" if ipv6 else "-4", "-c", "1", "-W", str(seconds), host]


def parse_ttl(output: bytes) -> int | None:
    """The TTL (or IPv6 hop limit) in ping output, or None."""
    match = TTL_PATTERN.search(output)
    return int(match.group(1)) if match else None


def ping_host(host: str, timeout: float = 1) -> tuple[bool, int | None]:
    """(alive, ttl) from one ping; ttl is None when the reply didn't show one."""
    try:
        done = subprocess.run(ping_command(host, timeout), capture_output=True,
                              stdin=subprocess.DEVNULL, timeout=timeout + 4, check=False)
    except (OSError, subprocess.TimeoutExpired):  # no ping program, or no answer in time
        return False, None
    output = done.stdout + done.stderr
    alive = done.returncode == 0 and b"unreachable" not in output.lower()
    return alive, parse_ttl(output)
