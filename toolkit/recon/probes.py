"""Banner probes: how each port's banner is read, one registry entry per port (D13).

``PROBES`` maps a port to a ``Probe``: which reader to use (a key of ``READERS``), whether
to wrap the connection in TLS first, what to send for a request/answer service, and the
UDP payload. Ports without an entry get ``DEFAULT_PROBE`` (read what the service says,
or nudge it with an empty line). Adding a probe = adding one entry.

Readers get the scan's own connection (R2) and return the banner text; TLS is used for
banners only, without certificate checks (B9).
"""

import re
import ssl
from dataclasses import dataclass

from .text import clean_text


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
PROBES = {
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


def probe_for(port):
    return PROBES.get(port, DEFAULT_PROBE)


def udp_payload(port):
    """What to send to a UDP port to get an answer."""
    return probe_for(port).udp_payload or DEFAULT_UDP_PAYLOAD


def banner_tls_context():
    """TLS for reading a banner only: no certificate check, so a service with a self-signed
    certificate still shows its banner (B9). The web lookups (NVD, AbuseIPDB, OpenRouter)
    never use it: they keep full certificate verification."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _receive_once(sock, limit=1024):
    try:
        return sock.recv(limit)
    except TimeoutError:
        return b""


def _receive_all(sock, limit=8192):
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


def banner_text(data):
    return data.decode("utf-8", errors="replace").strip()


def _host_header(host):
    return f"[{host}]" if ":" in host else host


def _read_detect(sock, host, probe):
    return f"{probe.label} service detected"


def _read_http(sock, host, probe):
    sock.sendall(f"HEAD / HTTP/1.0\r\nHost: {_host_header(host)}\r\n\r\n".encode())
    reply = banner_text(_receive_all(sock))
    for line in reply.splitlines():
        if line.lower().startswith("server:"):
            return line.split(":", 1)[1].strip()
    return "HTTP service (no Server header)" if reply.startswith("HTTP/") else reply


def _read_mysql(sock, host, probe):
    data = _receive_once(sock)
    match = MYSQL_GREETING.match(data)
    return f"MySQL {match.group(1).decode()}" if match else banner_text(data)


def _read_request(sock, host, probe):
    sock.sendall(probe.send)
    return banner_text(_receive_once(sock))


def _read_greeting(sock, host, probe):
    """The service speaks first: read what it says."""
    return banner_text(_receive_once(sock))


def _read_nudge(sock, host, probe):
    """Read what the service says; if it stays silent, give it a line to answer."""
    data = _receive_once(sock)
    if not data:
        sock.sendall(b"\r\n")
        data = _receive_once(sock)
    return banner_text(data)


READERS = {"detect": _read_detect, "http": _read_http, "mysql": _read_mysql,
           "request": _read_request, "greeting": _read_greeting, "nudge": _read_nudge}


def read_banner(sock, host, port):
    """The banner of an open port, read over the scan's own connection (R2)."""
    probe = probe_for(port)
    read = READERS[probe.read]
    if probe.tls:
        with banner_tls_context().wrap_socket(sock) as tls:
            return read(tls, host, probe)
    return read(sock, host, probe)


# --- Software in a banner ---------------------------------------------------------------------

PRODUCT_VERSION = re.compile(r"([A-Za-z][A-Za-z0-9\-]{1,30})[/_ ]v?(\d+(?:\.\d+)+)")
REDIS_VERSION = re.compile(r"redis_version:(\d+(?:\.\d+)+)")
NOT_PRODUCTS = {"http", "https", "ssh", "rtsp", "sip"}  # protocol names, not software


def product_version(banner):
    """"product version" from a banner (e.g. "OpenSSH 8.9"), or None. This short,
    allow-listed text is the only part of a banner that ever leaves the computer (S12)."""
    text = clean_text(banner)
    match = REDIS_VERSION.search(text)
    if match:
        return f"Redis {match.group(1)}"
    for match in PRODUCT_VERSION.finditer(text):
        if match.group(1).lower() not in NOT_PRODUCTS:
            return f"{match.group(1)} {match.group(2)}"[:60]
    return None
