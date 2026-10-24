"""B13: IPv6 targets."""

import socket

import pytest

from toolkit.recon import targets
from toolkit.data.recon import scanner as recon_scanner


def _has_ipv6_loopback():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


def test_ipv6_target(monkeypatch):
    assert targets.parse_target("::1", 256) == ["::1"]
    assert targets.parse_target("[2001:DB8::1]", 256) == ["2001:db8::1"]
    assert targets.parse_target("2001:db8::/126", 256) == ["2001:db8::1", "2001:db8::2",
                                                           "2001:db8::3"]
    assert not targets.is_public_ip("::1") and targets.is_public_ip("2606:4700::1111")

    def ipv6_only(host, port, *args, **kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::5", 0, 0, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", ipv6_only)
    assert targets.parse_target("ipv6-only.example", 256) == ["2001:db8::5"]


@pytest.mark.skipif(not _has_ipv6_loopback(), reason="no IPv6 loopback on this machine")
def test_ipv6_scan_of_this_computer(tcp_server):
    port = tcp_server(b"SSH-2.0-OpenSSH_9.6\r\n", host="::1", family=socket.AF_INET6)
    result = recon_scanner.scan_tcp_port("::1", port)
    assert (result.state, result.banner) == ("open", "SSH-2.0-OpenSSH_9.6")
    with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as silent:
        silent.bind(("::1", 0))
        udp = recon_scanner.scan_udp_port("::1", silent.getsockname()[1], timeout=0.5)
    assert udp.state == "open|filtered"
