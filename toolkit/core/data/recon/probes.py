"""Banner probes: how each port's banner is read, one registry entry per port (D13).

``PROBES`` maps a port to a ``Probe``: which reader to use (a key of ``READERS``), whether
to wrap the connection in TLS first, what to send for a request/answer service, and the
UDP payload. Ports without an entry get ``DEFAULT_PROBE`` (read what the service says, or
ask it ``GET /``: many services answer that, HTTP on any port among them). Adding a probe =
adding one entry.

Readers get the scan's own connection (R2) and return a ``Reply``: the banner and the
service the answer showed (``toolkit.core.data.recon.services``; step A2: from the answer,
never from the port number). TLS is used for banners only, without certificate checks (B9).
"""

import re
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass

from .services import identify, over_tls


@dataclass(frozen=True)
class Probe:
    """How to read one port's banner over TCP, and what to send to it over UDP."""

    read: str = "nudge"  # a key of READERS
    tls: bool = False  # wrap the connection in TLS before reading
    send: bytes = b""  # read "request": what to send first
    udp_payload: bytes | None = None  # None: DEFAULT_UDP_PAYLOAD


@dataclass(frozen=True)
class Reply:
    """What a port answered: the banner (text from the network) and the service the answer
    showed ("" when it didn't)."""

    text: str = ""
    service: str = ""


DEFAULT_PROBE = Probe()
DEFAULT_UDP_PAYLOAD = b"\r\n"
PRINTER = Probe("greeting")  # a printer prints what it gets: nothing is sent to it
PROBES: dict[int, Probe] = {
    21: Probe("greeting"),  # FTP: greeting services speak first
    22: Probe("greeting"),  # SSH
    25: Probe("greeting"),  # SMTP
    # DNS "version.bind" TXT/CH query (the v4 hex string mixed in plain letters and never ran)
    53: Probe(udp_payload=bytes.fromhex("00ff01000001000000000000")
              + b"\x07version\x04bind\x00\x00\x10\x00\x03"),
    80: Probe("http"),
    110: Probe("greeting"),  # POP3
    123: Probe(udp_payload=bytes.fromhex("1b0304fa0001000000000000")),  # NTP
    143: Probe("greeting"),  # IMAP
    161: Probe(udp_payload=bytes.fromhex(  # SNMP
        "302602010004067075626c6963a0190201000201000201003011300f060b2b06010201"
        "0101000500")),
    443: Probe("http", tls=True),
    465: Probe("greeting", tls=True),  # SMTPS
    587: Probe("greeting"),  # SMTP submission
    636: Probe(tls=True),  # LDAPS
    993: Probe("greeting", tls=True),  # IMAPS
    995: Probe("greeting", tls=True),  # POP3S
    3306: Probe("mysql"),
    3389: Probe("rdp"),
    5432: Probe("postgresql"),
    6379: Probe("request", send=b"INFO server\r\n"),  # Redis
    8000: Probe("http"),
    8008: Probe("http"),
    8080: Probe("http"),
    8443: Probe("http", tls=True),
    8888: Probe("http"),
    **{port: PRINTER for port in range(9100, 9108)},  # raw printing (JetDirect)
    9200: Probe("http"),  # Elasticsearch REST
    11211: Probe("request", send=b"version\r\n"),  # Memcached
}
MYSQL_GREETING = re.compile(rb"^.{4}\x0a([0-9][\w.\-]*)\x00", re.DOTALL)
ELASTICSEARCH = re.compile(r'"number"\s*:\s*"(\d+(?:\.\d+)+)".*"tagline"\s*:\s*"You Know, for '
                           r'Search"', re.DOTALL)
POSTGRESQL_SSL_REQUEST = (8).to_bytes(4, "big") + (80877103).to_bytes(4, "big")
POSTGRESQL_SSL = {b"S": "PostgreSQL (SSL offered)", b"N": "PostgreSQL (no SSL)",
                  b"E": "PostgreSQL (refused the SSL request)"}
# X.224 Connection Request with an RDP negotiation request (TLS and CredSSP), [MS-RDPBCGR]
RDP_REQUEST = bytes.fromhex("030000130ee000000000000100080003000000")
RDP_PROTOCOLS = {0: "standard RDP security", 1: "TLS", 2: "CredSSP (NLA)", 4: "RDSTLS",
                 8: "CredSSP with early user authorization"}


def probe_for(port: int) -> Probe:
    """The probe of ``port``: its entry in ``PROBES``, else ``DEFAULT_PROBE``."""
    return PROBES.get(port, DEFAULT_PROBE)


def udp_payload(port: int) -> bytes:
    """What to send to a UDP port to get an answer."""
    return probe_for(port).udp_payload or DEFAULT_UDP_PAYLOAD


def banner_tls_context() -> ssl.SSLContext:
    """TLS for reading a banner only: no certificate check, so a service with a self-signed
    certificate still shows its banner (B9). The web lookups (NVD, AbuseIPDB, OpenRouter)
    never use it: they keep full certificate verification."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _receive_once(sock: socket.socket, limit: int = 1024) -> bytes:
    """One read of at most ``limit`` bytes; empty if nothing came within the timeout."""
    try:
        return sock.recv(limit)
    except TimeoutError:
        return b""


def _receive_all(sock: socket.socket, limit: int = 8192) -> bytes:
    """Reads until the service closes the connection, ``limit`` bytes arrived or the timeout passed;
    what came."""
    data = b""
    try:
        while len(data) < limit:
            chunk = sock.recv(limit - len(data))
            if not chunk:
                break
            data += chunk
    except TimeoutError:
        pass
    return data


def banner_text(data: bytes) -> str:
    """Bytes from the network as text (invalid UTF-8 replaced), trimmed."""
    return data.decode("utf-8", errors="replace").strip()


def _text_reply(data: bytes) -> Reply:
    """What a port sent, and the service it shows."""
    text = banner_text(data)
    return Reply(text, identify(text))


def _host_header(host: str) -> str:
    """``host`` for an HTTP Host header: an IPv6 address in brackets."""
    return f"[{host}]" if ":" in host else host


def _get(sock: socket.socket, host: str) -> bytes:
    """Sends ``GET /`` and returns the answer (the service closes the connection after it)."""
    sock.sendall(f"GET / HTTP/1.0\r\nHost: {_host_header(host)}\r\n\r\n".encode())
    return _receive_all(sock)


def http_reply(data: bytes) -> Reply:
    """An answer to ``GET /``: an HTTP one is the Server header (else its status line), or
    Elasticsearch and its version; any other answer is taken as it is."""
    text = banner_text(data)
    if not text.startswith("HTTP/"):
        return Reply(text, identify(text))
    head, _, body = text.replace("\r\n", "\n").partition("\n\n")
    found = ELASTICSEARCH.search(body)
    if found:
        return Reply(f"Elasticsearch {found.group(1)}", "elasticsearch")
    other = identify(body)  # a service that answers HTTP with its own words (MongoDB)
    if other:
        return Reply(body.strip().splitlines()[0], other)
    lines = head.splitlines()
    for line in lines[1:]:
        if line.lower().startswith("server:"):
            return Reply(line.split(":", 1)[1].strip(), "http")
    return Reply(lines[0], "http")


# A reader gets the open connection, the host and the port's probe; it returns the reply.
Reader = Callable[[socket.socket, str, Probe], Reply]


def _read_http(sock: socket.socket, host: str, probe: Probe) -> Reply:
    """Sends ``GET /``: the HTTP answer (``http_reply``)."""
    return http_reply(_get(sock, host))


def _read_mysql(sock: socket.socket, host: str, probe: Probe) -> Reply:
    """The version from the MySQL/MariaDB greeting packet, else what was received."""
    data = _receive_once(sock)
    match = MYSQL_GREETING.match(data)
    return Reply(f"MySQL {match.group(1).decode()}", "mysql") if match else _text_reply(data)


def _read_postgresql(sock: socket.socket, host: str, probe: Probe) -> Reply:
    """PostgreSQL's one-byte answer to an SSL request (no login is tried); any other answer
    is taken as it is."""
    sock.sendall(POSTGRESQL_SSL_REQUEST)
    data = _receive_once(sock, 1)
    if data in POSTGRESQL_SSL:
        return Reply(POSTGRESQL_SSL[data], "postgresql")
    return _text_reply(data + _receive_once(sock))


def _read_rdp(sock: socket.socket, host: str, probe: Probe) -> Reply:
    """An RDP connection request (no login): an X.224 Connection Confirm is RDP, with the
    security it chose; any other answer is taken as it is."""
    sock.sendall(RDP_REQUEST)
    data = _receive_once(sock)
    if len(data) < 11 or data[0] != 3 or data[5] != 0xD0:
        return _text_reply(data)
    if len(data) >= 19 and data[11] == 0x02:  # RDP_NEG_RSP: the chosen protocol
        chosen = int.from_bytes(data[15:19], "little")
        return Reply(f"RDP (security: {RDP_PROTOCOLS.get(chosen, chosen)})", "rdp")
    if len(data) >= 19 and data[11] == 0x03:  # RDP_NEG_FAILURE
        return Reply("RDP (refused the security asked)", "rdp")
    return Reply("RDP", "rdp")


def _read_request(sock: socket.socket, host: str, probe: Probe) -> Reply:
    """Sends ``probe.send`` and returns the answer."""
    sock.sendall(probe.send)
    return _text_reply(_receive_once(sock))


def _read_greeting(sock: socket.socket, host: str, probe: Probe) -> Reply:
    """The service speaks first: read what it says; nothing is sent."""
    return _text_reply(_receive_once(sock))


def _read_nudge(sock: socket.socket, host: str, probe: Probe) -> Reply:
    """Read what the service says; if it stays silent, ask it ``GET /`` (``http_reply``)."""
    data = _receive_once(sock)
    return _text_reply(data) if data else http_reply(_get(sock, host))


READERS: dict[str, Reader] = {
    "http": _read_http, "mysql": _read_mysql, "postgresql": _read_postgresql,
    "rdp": _read_rdp, "request": _read_request, "greeting": _read_greeting,
    "nudge": _read_nudge}


def read_banner(sock: socket.socket, host: str, port: int) -> Reply:
    """The banner of an open port and the service it showed, read over the scan's own
    connection (R2)."""
    probe = probe_for(port)
    read = READERS[probe.read]
    if probe.tls:
        with banner_tls_context().wrap_socket(sock) as tls:
            reply = read(tls, host, probe)
        return Reply(reply.text, over_tls(reply.service))
    return read(sock, host, probe)
