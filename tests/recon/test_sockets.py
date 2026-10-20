"""R1, R2: one connection per port, always closed."""

import socket

import pytest

from toolkit.recon import probes
from toolkit.recon import scanner as recon_scanner


@pytest.fixture
def recorded(monkeypatch):
    """Every socket the scanner opens with create_connection."""
    opened = []
    real = socket.create_connection

    def record(*args, **kwargs):
        sock = real(*args, **kwargs)
        opened.append(sock)
        return sock

    monkeypatch.setattr(socket, "create_connection", record)
    return opened


def test_no_socket_leak_on_errors(tcp_server, recorded, monkeypatch):
    port = tcp_server(b"SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1\r\n")
    result = recon_scanner.scan_tcp_port("127.0.0.1", port)
    assert (result.state, result.banner) == ("open", "SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1")
    assert len(recorded) == 1, "the banner is read over the scan's own connection (R2)"

    def broken(sock, host, port):
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(recon_scanner, "read_banner", broken)
    result = recon_scanner.scan_tcp_port("127.0.0.1", port)
    assert (result.state, result.banner) == ("open", "")

    def crash(sock, host, port):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(recon_scanner, "read_banner", crash)
    with pytest.raises(RuntimeError):
        recon_scanner.scan_tcp_port("127.0.0.1", port)
    assert len(recorded) == 3
    assert all(sock.fileno() == -1 for sock in recorded), "every socket is closed"


def test_tls_handshake_failure_closes_socket(tcp_server, recorded, monkeypatch):
    port = tcp_server(b"not TLS at all\r\n")
    monkeypatch.setitem(probes.PROBES, port, probes.Probe(tls=True))
    result = recon_scanner.scan_tcp_port("127.0.0.1", port)
    assert (result.state, result.banner) == ("open", "")
    assert len(recorded) == 1 and recorded[0].fileno() == -1


def test_closed_port(free_port):
    assert recon_scanner.scan_tcp_port("127.0.0.1", free_port).state == "closed"


def test_http_banner_from_server_header(tcp_server, monkeypatch):
    port = tcp_server(b"HTTP/1.0 200 OK\r\nServer: Apache/2.4.49 (Unix)\r\n\r\n",
                      read_first=True)
    monkeypatch.setitem(probes.PROBES, port, probes.Probe("http"))
    assert recon_scanner.scan_tcp_port("127.0.0.1", port).banner == "Apache/2.4.49 (Unix)"
