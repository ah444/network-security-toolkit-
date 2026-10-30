"""This computer: whether an address is one of its own, and its OS (step A3). Nothing is sent:
"connecting" a UDP socket only picks the route and the address it would leave from."""

import ipaddress
import platform
import socket


def _leaving_address(address: str) -> str:
    """The address of this computer that traffic to ``address`` would leave from (a UDP socket
    "connected" to it: no packet is sent); OSError if there is no route."""
    family = socket.AF_INET6 if ipaddress.ip_address(address).version == 6 else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.connect((address, 9))
        return str(sock.getsockname()[0]).split("%", 1)[0]  # an IPv6 scope ("%12") dropped


def is_this_computer(address: str) -> bool:
    """True when ``address`` is one of this computer's own: a loopback address, or the one
    traffic to it would leave from; False if there is no route."""
    ip = ipaddress.ip_address(address)
    if ip.is_loopback:
        return True
    try:
        return ipaddress.ip_address(_leaving_address(address)) == ip
    except OSError:
        return False


def this_os() -> str:
    """This computer's OS, such as "Windows 10", "Linux 6.8.0" or "macOS 14.5"."""
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}".strip()
    return f"{system} {platform.release()}".strip()
