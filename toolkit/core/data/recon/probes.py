"""Banner probes: how each port's banner is read, one registry entry per port (D13).

``PROBES`` maps a port to a ``Probe``: which reader to use (a key of ``READERS``), whether
to wrap the connection in TLS first, what to send for a request/answer service, and the
UDP payload. Ports without an entry get ``DEFAULT_PROBE`` (read what the service says,
or nudge it with an empty line). Adding a probe = adding one entry.

Readers get the scan's own connection (R2) and return the banner text; TLS is used for
banners only, without certificate checks (B9).
"""

import re
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass



@dataclass(frozen=True)
class Probe:
    """How to read one port's banner over TCP, and what to send to it over UDP."""

    read: str = "nudge"  # a key of READERS
    tls: bool = False  # wrap the connection in TLS before reading
    send: bytes = b""  # read "request": what to send first
    label: str = ""  # read "detect": the service shown (no banner is read)
    udp_payload: bytes | None = None  # None: DEFAULT_UDP_PAYLOAD


DEFAULT_PROBE = Probe()
DEFAULT_UDP_PAYLOAD = b"\r\n"
PROBES: dict[int, Probe] = {
    21: Probe("greeting"),  # FTP: greeting services speak first
    22: Probe("greeting"),  # SSH
    25: Probe("greeting"),  # SMTP
    # DNS "version.bind" TXT/CH query (the v4 hex string mixed in plain letters and never ran)
    53: Probe("detect", label="DNS", udp_payload=bytes.fromhex("00ff01000001000000000000")
              + b"\x07version\x04bind\x00\x00\x10\x00\x03"),
    80: Probe("http"),
    110: Probe("greeting"),  # POP3
    123: Probe(udp_payload=bytes.fromhex("1b0304fa0001000000000000")),  # NTP
    143: Probe("greeting"),  # IMAP
    161: Probe(udp_payload=bytes.fromhex(  # SNMP
        "302602010004067075626c6963a0190201000201000201003011300f060b2b06010201"
        "0101000500")),
    389: Probe("detect", label="LDAP"),
    443: Probe("http", tls=True),
    465: Probe("greeting", tls=True),  # SMTPS
    587: Probe("greeting"),  # SMTP submission
    636: Probe("detect", tls=True, label="LDAPS"),
    993: Probe("greeting", tls=True),  # IMAPS
    995: Probe("greeting", tls=True),  # POP3S
    1433: Probe("detect", label="MSSQL"),
    3306: Probe("mysql"),
    3389: Probe("detect", label="RDP"),
    5432: Probe("detect", label="PostgreSQL"),
    6379: Probe("request", send=b"INFO server\r\n"),  # Redis
    8000: Probe("http"),
    8008: Probe("http"),
    8080: Probe("http"),
    8443: Probe("http", tls=True),
    8888: Probe("http"),
    9200: Probe("http"),  # Elasticsearch REST
    9300: Probe("detect", label="Elasticsearch transport"),
    11211: Probe("request", send=b"version\r\n"),  # Memcached
    27017: Probe("detect", label="MongoDB"),
}
MYSQL_GREETING = re.compile(rb"^.{4}\x0a([0-9][\w.\-]*)\x00", re.DOTALL)


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


def _host_header(host: str) -> str:
    """``host`` for an HTTP Host header: an IPv6 address in brackets."""
    return f"[{host}]" if ":" in host else host


# A reader gets the open connection, the host and the port's probe; it returns the banner.
Reader = Callable[[socket.socket, str, Probe], str]


def _read_detect(sock: socket.socket, host: str, probe: Probe) -> str:
    """Reads nothing: the port is shown as its known service (``probe.label``)."""
    return f"{probe.label} service detected"


def _read_http(sock: socket.socket, host: str, probe: Probe) -> str:
    """Sends ``HEAD /`` and returns the Server header, or says the service is HTTP without one;
    anything else that answered is returned as it is."""
    sock.sendall(f"HEAD / HTTP/1.0\r\nHost: {_host_header(host)}\r\n\r\n".encode())
    reply = banner_text(_receive_all(sock))
    for line in reply.splitlines():
        if line.lower().startswith("server:"):
            return line.split(":", 1)[1].strip()
    return "HTTP service (no Server header)" if reply.startswith("HTTP/") else reply


def _read_mysql(sock: socket.socket, host: str, probe: Probe) -> str:
    """The version from MySQL's greeting packet, else what was received."""
    data = _receive_once(sock)
    match = MYSQL_GREETING.match(data)
    return f"MySQL {match.group(1).decode()}" if match else banner_text(data)


def _read_request(sock: socket.socket, host: str, probe: Probe) -> str:
    """Sends ``probe.send`` and returns the answer."""
    sock.sendall(probe.send)
    return banner_text(_receive_once(sock))


def _read_greeting(sock: socket.socket, host: str, probe: Probe) -> str:
    """The service speaks first: read what it says."""
    return banner_text(_receive_once(sock))


def _read_nudge(sock: socket.socket, host: str, probe: Probe) -> str:
    """Read what the service says; if it stays silent, give it a line to answer."""
    data = _receive_once(sock)
    if not data:
        sock.sendall(b"\r\n")
        data = _receive_once(sock)
    return banner_text(data)


READERS: dict[str, Reader] = {
    "detect": _read_detect, "http": _read_http, "mysql": _read_mysql,
    "request": _read_request, "greeting": _read_greeting, "nudge": _read_nudge}


def read_banner(sock: socket.socket, host: str, port: int) -> str:
    """The banner of an open port, read over the scan's own connection (R2)."""
    probe = probe_for(port)
    read = READERS[probe.read]
    if probe.tls:
        with banner_tls_context().wrap_socket(sock) as tls:
            return read(tls, host, probe)
    return read(sock, host, probe)
