"""Ping a host and guess its OS family from the reply's TTL (B6, B7).

The ping flags and timeouts follow the OS (``platform.system()``); the guess is only
a guess, because TTLs can be changed.
"""

import ipaddress
import platform
import re
import subprocess

TTL_PATTERN = re.compile(rb"(?:ttl|hlim)[=:]\s*(\d+)", re.IGNORECASE)
# Common initial TTLs. Each router hop lowers the TTL by one, so a reply's TTL is at most
# its initial value: the guess is the smallest initial TTL that is not below it.
INITIAL_TTLS = (
    (64, "Linux / macOS / BSD / other Unix"),
    (128, "Windows"),
    (255, "Network device (e.g. Cisco) or Solaris/AIX"),
)


def ping_command(host, timeout=1):
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


def parse_ttl(output):
    """The TTL (or IPv6 hop limit) in ping output, or None."""
    match = TTL_PATTERN.search(output)
    return int(match.group(1)) if match else None


def ping_host(host, timeout=1):
    """(alive, ttl) from one ping; ttl is None when the reply didn't show one."""
    try:
        done = subprocess.run(ping_command(host, timeout), capture_output=True,
                              timeout=timeout + 4, check=False)
    except (OSError, subprocess.TimeoutExpired):  # no ping program, or no answer in time
        return False, None
    output = done.stdout + done.stderr
    alive = done.returncode == 0 and b"unreachable" not in output.lower()
    return alive, parse_ttl(output)


def guess_os(ttl):
    """(OS family, hops away) from a reply's TTL, or None. A guess only: TTLs can be set."""
    if ttl is None or not 1 <= ttl <= 255:
        return None
    for initial, name in INITIAL_TTLS:
        if ttl <= initial:
            return name, initial - ttl
    return None
