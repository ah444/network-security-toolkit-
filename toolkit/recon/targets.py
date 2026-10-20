"""Scan targets, port ranges and limits (B11, B13, S16)."""

import ipaddress
import socket


def parse_target(text, max_hosts):
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
        infos = socket.getaddrinfo(text, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Could not resolve {text!r}: {error}") from None
    return [infos[0][4][0]]


def parse_port_range(start_text, end_text):
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


def check_limits(host_count, probe_count, limits):
    """A message when a scan is above the limits, else None."""
    if host_count > limits.max_hosts:
        return (f"{host_count:,} hosts is above the limit of {limits.max_hosts:,} "
                "(TOOLKIT_SCAN_MAX_HOSTS).")
    if probe_count > limits.max_probes:
        return (f"{probe_count:,} host x port checks is above the limit of "
                f"{limits.max_probes:,} (TOOLKIT_SCAN_MAX_PROBES). Scan fewer ports or hosts.")
    return None


def is_ip_literal(text):
    try:
        ipaddress.ip_address(text.strip().strip("[]"))
    except ValueError:
        return False
    return True


def is_public_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False
