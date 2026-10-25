"""Step A2: the service is the one its answer showed, never the port number alone; a port's
usual name is shown with a "?" (not confirmed); no reader invents a banner."""

import contextlib
import socket

import pytest

from toolkit.cli.ui import scan as screen
from toolkit.core.data.recon import probes, services
from toolkit.core.data.recon import scanner as recon_scanner
from toolkit.core.data.recon.scanner import PortResult

SILENT = object()  # recv times out


class FakeSocket:
    """Answers each recv with the next reply (SILENT: a timeout, then nothing more); records what
    is sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = b""

    def sendall(self, data):
        self.sent += data

    def recv(self, limit):
        reply = self.replies.pop(0) if self.replies else SILENT
        if reply is SILENT:
            raise TimeoutError
        return reply[:limit]


@pytest.fixture
def plain_tls(monkeypatch):
    """TLS that wraps nothing, so a TLS probe reads the fake socket as it is."""
    class Context:
        def wrap_socket(self, sock):
            return contextlib.nullcontext(sock)

    monkeypatch.setattr(probes, "banner_tls_context", Context)


@pytest.mark.parametrize(("text", "service"), [
    ("SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1", "ssh"),
    ("HTTP/1.1 400 Bad Request", "http"),
    ("RFB 003.008", "vnc"),
    ("+OK Dovecot ready.", "pop3"),
    ("* OK [CAPABILITY IMAP4rev1] Dovecot ready.", "imap"),
    ("220 (vsFTPd 3.0.3)", "ftp"),
    ("220-FileZilla Server 1.7.0", "ftp"),
    ("220 Microsoft FTP Service", "ftp"),
    ("220 ProFTPD 1.3.5 Server (Debian)", "ftp"),
    ("220 mail.example.com ESMTP Postfix (Ubuntu)", "smtp"),
    ("220 mx ESMTP Exim 4.92", "smtp"),
    ("220 ready", ""),  # FTP or SMTP: the answer doesn't say
    ("$3456\r\n# Server\r\nredis_version:7.0.11", "redis"),
    ("-NOAUTH Authentication required.", "redis"),
    ("VERSION 1.6.21", "memcached"),
    ("MySQL 8.0.33", "mysql"),
    ("Host '192.168.1.11' is not allowed to connect to this MariaDB server", "mysql"),
    ("It looks like you are trying to access MongoDB over HTTP on the native driver port.",
     "mongodb"),
    ("AMQP\x00\x00\x09\x01", "amqp"),
    ("-ERR unknown command", ""),  # POP3 or Redis
    ("", ""),
])
def test_identify(text, service):
    assert services.identify(text) == service


def test_the_usual_name_has_a_question_mark(monkeypatch):
    assert services.service_label(5432, "") == "postgresql?"
    assert services.service_label(5432, "http") == "http", "the answer wins over the port"
    assert services.service_label(47822, "ssh") == "ssh"
    monkeypatch.setattr(socket, "getservbyport", lambda port, protocol: "epmap")
    assert services.service_label(135, "") == "epmap?"


def test_usual_names_are_the_same_on_every_os(monkeypatch):
    def no_list(port, protocol):
        raise OSError("not in the services list")

    monkeypatch.setattr(socket, "getservbyport", no_list)
    assert services.usual_name(3306) == "mysql" and services.usual_name(27017) == "mongodb"
    assert services.service_label(47822, "") == "", "no usual name: nothing is made up"
    assert services.usual_name(3306, "udp") == "", "the table is for TCP"


def test_over_tls():
    assert [services.over_tls(name) for name in ("http", "imap", "ssh", "")] == [
        "https", "imaps", "ssh", ""]


def test_no_reader_invents_a_banner(plain_tls):
    for port in probes.PROBES:
        reply = probes.read_banner(FakeSocket(SILENT, SILENT, SILENT), "127.0.0.1", port)
        assert reply == probes.Reply(), (port, reply)  # no answer: no text, no service
    assert probes.read_banner(FakeSocket(SILENT, SILENT), "127.0.0.1", 47822) == probes.Reply()


def test_a_silent_port_is_asked_get():
    sock = FakeSocket(SILENT, b"HTTP/1.1 503 Service Unavailable\r\nServer: Microsoft-HTTPAPI/2.0"
                              b"\r\n\r\n")
    reply = probes.read_banner(sock, "127.0.0.1", 5357)
    assert reply == probes.Reply("Microsoft-HTTPAPI/2.0", "http")
    assert sock.sent == b"GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"


def test_a_service_that_speaks_first_gets_nothing_sent():
    sock = FakeSocket(b"SSH-2.0-OpenSSH_9.6\r\n")
    assert probes.read_banner(sock, "127.0.0.1", 2222) == probes.Reply("SSH-2.0-OpenSSH_9.6",
                                                                        "ssh")
    assert sock.sent == b""


def test_a_printer_gets_nothing_sent():
    for port in range(9100, 9108):
        sock = FakeSocket(SILENT)
        assert probes.read_banner(sock, "127.0.0.1", port) == probes.Reply()
        assert sock.sent == b"", port


@pytest.mark.parametrize(("answer", "reply"), [
    (b"HTTP/1.1 200 OK\r\nServer: nginx/1.18.0\r\n\r\n<html>", ("nginx/1.18.0", "http")),
    (b"HTTP/1.0 200 OK\r\nContent-Type: text/html\r\n\r\n<html>", ("HTTP/1.0 200 OK", "http")),
    (b'HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{"name" : "node-1", '
     b'"version" : {"number" : "8.11.1", "build_flavor" : "default"}, '
     b'"tagline" : "You Know, for Search"}', ("Elasticsearch 8.11.1", "elasticsearch")),
    (b"HTTP/1.0 200 OK\r\nConnection: close\r\nContent-Length: 85\r\n\r\nIt looks like you are "
     b"trying to access MongoDB over HTTP on the native driver port.\r\n",
     ("It looks like you are trying to access MongoDB over HTTP on the native driver port.",
      "mongodb")),
    (b"SSH-2.0-OpenSSH_9.6\r\n", ("SSH-2.0-OpenSSH_9.6", "ssh")),  # not HTTP: as it is
])
def test_http_answers(answer, reply):
    sock = FakeSocket(answer)
    assert probes.read_banner(sock, "127.0.0.1", 8080) == probes.Reply(*reply)
    assert sock.sent.startswith(b"GET / HTTP/1.0\r\n")


def test_https_is_named_https(plain_tls):
    reply = probes.read_banner(FakeSocket(b"HTTP/1.1 200 OK\r\nServer: IIS\r\n\r\n"),
                               "127.0.0.1", 443)
    assert reply == probes.Reply("IIS", "https")


@pytest.mark.parametrize(("answer", "reply"), [
    (b"N", ("PostgreSQL (no SSL)", "postgresql")),
    (b"S", ("PostgreSQL (SSL offered)", "postgresql")),
    (b"HTTP/1.1 400 Bad Request\r\n", ("HTTP/1.1 400 Bad Request", "http")),
])
def test_postgresql(answer, reply):
    sock = FakeSocket(answer[:1], answer[1:]) if len(answer) > 1 else FakeSocket(answer)
    assert probes.read_banner(sock, "127.0.0.1", 5432) == probes.Reply(*reply)
    assert sock.sent == bytes.fromhex("0000000804d2162f"), "an SSL request only, no login"


def _rdp_confirm(neg=b""):
    x224 = bytes([6 + len(neg), 0xD0, 0, 0, 0x12, 0x34, 0]) + neg
    return bytes([3, 0]) + (4 + len(x224)).to_bytes(2, "big") + x224


@pytest.mark.parametrize(("answer", "reply"), [
    (_rdp_confirm(bytes.fromhex("020008000200 0000".replace(" ", ""))),
     ("RDP (security: CredSSP (NLA))", "rdp")),
    (_rdp_confirm(bytes.fromhex("0200080001000000")), ("RDP (security: TLS)", "rdp")),
    (_rdp_confirm(bytes.fromhex("0300080005000000")), ("RDP (refused the security asked)", "rdp")),
    (_rdp_confirm(), ("RDP", "rdp")),
    (b"SSH-2.0-OpenSSH_9.6", ("SSH-2.0-OpenSSH_9.6", "ssh")),
])
def test_rdp(answer, reply):
    sock = FakeSocket(answer)
    assert probes.read_banner(sock, "127.0.0.1", 3389) == probes.Reply(*reply)
    assert sock.sent == probes.RDP_REQUEST and len(sock.sent) == 19


def test_scan_takes_the_service_from_the_answer(tcp_server):
    ssh = tcp_server(b"SSH-2.0-OpenSSH_9.6\r\n")
    http = tcp_server(b"HTTP/1.0 200 OK\r\nServer: Apache/2.4.49 (Unix)\r\n\r\n", read_first=True)
    assert recon_scanner.scan_tcp_port("127.0.0.1", ssh).service == "ssh"
    result = recon_scanner.scan_tcp_port("127.0.0.1", http)
    assert (result.service, result.banner) == ("http", "Apache/2.4.49 (Unix)")


def test_a_failed_read_keeps_the_usual_name(tcp_server, monkeypatch):
    port = tcp_server(b"anything")
    monkeypatch.setitem(services.PORT_NAMES, port, "postgresql")

    def broken(sock, host, port):
        raise ConnectionResetError

    monkeypatch.setattr(recon_scanner, "read_banner", broken)
    result = recon_scanner.scan_tcp_port("127.0.0.1", port)
    assert (result.state, result.service, result.banner) == ("open", "postgresql?", "")


def test_the_table_fits_80_columns(capsys):
    screen.print_table([PortResult(9300, "tcp", "open", "elasticsearch?", "x" * 100),
                        PortResult(5432, "tcp", "open", "postgresql", "PostgreSQL (no SSL)")],
                       "OPEN PORTS")
    lines = capsys.readouterr().out.splitlines()[2:]  # after the empty line and the title
    assert {len(line) for line in lines} == {79}
    assert "| elasticsearch? |" in lines[3] and "|   postgresql   |" in lines[4]
