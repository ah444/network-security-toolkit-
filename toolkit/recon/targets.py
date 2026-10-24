"""Scan targets, port ranges and limits (B11, B13, S16)."""

import ipaddress

from ..data.settings import ScanConfig
from ..data.recon.resolve import first_address


def parse_target(text: str, max_hosts: int) -> list[str]:
    """The IP addresses to scan: one IPv4/IPv6 address, a network (CIDR) or a host name."""
    text = text.strip()
    if not text:
        raise ValueError("No target given.")
    if "/" in text:
        try:
            network = ipaddress.ip_network(text, strict=False)
        except ValueError:
            raise ValueError(f"{text!r} is not a valid network "
                             "(example: 192.168.1.0/24).") from None
        if network.num_addresses > max_hosts:
            raise ValueError(f"{text} has {network.num_addresses:,} addresses; the limit is "
                             f"{max_hosts:,} hosts (TOOLKIT_SCAN_MAX_HOSTS).")
        return [str(ip) for ip in network.hosts()] or [str(network.network_address)]
    literal = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    try:
        return [str(ipaddress.ip_address(literal))]
    except ValueError:
        pass
    try:
        return [first_address(text)]
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Could not resolve {text!r}: {error}") from None


def parse_port_range(start_text: str, end_text: str) -> range:
    """``range`` of ports from the two answers; ValueError with a message otherwise."""
    try:
        start, end = int(start_text), int(end_text)
    except ValueError:
        raise ValueError("Ports must be whole numbers.") from None
    if not (1 <= start <= 65535 and 1 <= end <= 65535):
        raise ValueError("Ports must be between 1 and 65535.")
    if start > end:
        raise ValueError(f"The start port ({start}) is after the end port ({end}).")
    return range(start, end + 1)


def parse_ports(spec: str) -> list[int]:
    """Ports from ``80``, ``1-1024`` or a list such as ``22,80,8000-8100``: sorted, each
    port once; ValueError with a message otherwise."""
    parts = [part.strip() for part in spec.split(",")]
    if not any(parts):
        raise ValueError("No ports given (examples: 80, 1-1024, 22,80,8000-8100).")
    ports: set[int] = set()
    for part in parts:
        if not part:
            raise ValueError(f"{spec!r} has an empty item between commas.")
        start, dash, end = part.partition("-")
        try:
            ports.update(parse_port_range(start, end if dash else start))
        except ValueError as error:
            raise ValueError(f"{part!r}: {error}") from None
    return sorted(ports)


def check_limits(host_count: int, probe_count: int, limits: ScanConfig) -> str | None:
    """A message when a scan is above the limits, else None."""
    if host_count > limits.max_hosts:
        return (f"{host_count:,} hosts is above the limit of {limits.max_hosts:,} "
                "(TOOLKIT_SCAN_MAX_HOSTS).")
    if probe_count > limits.max_probes:
        return (f"{probe_count:,} host x port checks is above the limit of "
                f"{limits.max_probes:,} (TOOLKIT_SCAN_MAX_PROBES). Scan fewer ports or hosts.")
    return None


def is_ip_literal(text: str) -> bool:
    try:
        ipaddress.ip_address(text.strip().strip("[]"))
    except ValueError:
        return False
    return True


def is_public_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False
