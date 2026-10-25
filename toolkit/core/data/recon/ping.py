"""Ping a host and read the reply's TTL (B6).

The ping flags and timeouts follow the OS (``platform.system()``). What the TTL says
about the host is ``toolkit.core.services.recon.os_guess``.
"""

import ipaddress
import platform
import re
import subprocess

TTL_PATTERN = re.compile(rb"(?:ttl|hlim)[=:]\s*(\d+)", re.IGNORECASE)

def ping_command(host: str, timeout: float = 1) -> list[str]:
    """One echo request with this OS's flags, waiting at most ``timeout`` seconds."""
    system = platform.system().lower()
    ipv6 = ipaddress.ip_address(host).version == 6
    seconds = max(1, int(timeout))
    if system == "windows":
        return ["ping", "-6" if ipv6 else "-4", "-n", "1", "-w", str(seconds * 1000), host]
    if system == "darwin":
        if ipv6:  # macOS ping6 has no overall timeout flag; the subprocess timeout applies
            return ["ping6", "-c", "1", host]
        return ["ping", "-c", "1", "-t", str(seconds), host]
    return ["ping", "-6" if ipv6 else "-4", "-c", "1", "-W", str(seconds), host]


def parse_ttl(output: bytes) -> int | None:
    """The TTL (or IPv6 hop limit) in ping output, or None."""
    match = TTL_PATTERN.search(output)
    return int(match.group(1)) if match else None


def ping_host(host: str, timeout: float = 1) -> tuple[bool, int | None]:
    """(alive, ttl) from one ping; ttl is None when the reply didn't show one."""
    try:
        done = subprocess.run(ping_command(host, timeout), capture_output=True,
                              timeout=timeout + 4, check=False)
    except (OSError, subprocess.TimeoutExpired):  # no ping program, or no answer in time
        return False, None
    output = done.stdout + done.stderr
    alive = done.returncode == 0 and b"unreachable" not in output.lower()
    return alive, parse_ttl(output)
