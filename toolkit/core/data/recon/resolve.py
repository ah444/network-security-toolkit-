"""Host names to addresses (the system resolver)."""

import socket


def first_address(name: str) -> str:
    """The first address ``name`` resolves to; OSError or UnicodeError if none."""
    infos = socket.getaddrinfo(name, None, type=socket.SOCK_STREAM)
    return str(infos[0][4][0])
