"""Which service answered on a port (step A2): from what it said, never from the port number
alone.

``RULES`` name the service an answer shows (an SSH greeting is SSH on any port). When the
answer doesn't show it, the port's usual name is given with a "?" (``postgresql?``): a
service usually found there, not confirmed. The usual names come from ``PORT_NAMES`` (the
same on every OS) and then the system's services list.
"""

import re
import socket

RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^SSH-\d"), "ssh"),
    (re.compile(r"^HTTP/\d"), "http"),
    (re.compile(r"^RFB \d{3}\.\d{3}"), "vnc"),
    (re.compile(r"^\+OK"), "pop3"),
    (re.compile(r"^\* (?:OK|PREAUTH)"), "imap"),
    (re.compile(r"^220[ -].*(?:ftp|filezilla)", re.IGNORECASE), "ftp"),
    (re.compile(r"^220[ -].*(?:smtp|postfix|exim|sendmail)", re.IGNORECASE), "smtp"),
    (re.compile(r"redis_version:|^-NOAUTH |^-DENIED Redis"), "redis"),
    (re.compile(r"^VERSION \d"), "memcached"),
    (re.compile(r"^MySQL \d|is not allowed to connect to this (?:MySQL|MariaDB) server"),
     "mysql"),
    (re.compile(r"trying to access MongoDB over HTTP"), "mongodb"),
    (re.compile(r"^AMQP"), "amqp"),
)
# Usual services the system's list lacks (Windows has none of these) or names differently
PORT_NAMES = {
    465: "smtps", 587: "submission", 3306: "mysql", 5432: "postgresql", 5672: "amqp",
    5900: "vnc", 6379: "redis", 8000: "http-alt", 8008: "http-alt", 8080: "http-alt",
    8443: "https-alt", 8888: "http-alt", 9100: "jetdirect", 9200: "elasticsearch",
    9300: "elasticsearch", 11211: "memcached", 27017: "mongodb",
}
TLS_NAMES = {"http": "https", "imap": "imaps", "pop3": "pop3s", "smtp": "smtps",
             "ftp": "ftps"}


def identify(text: str) -> str:
    """The service ``text`` (what a port sent) shows, or an empty string."""
    for pattern, name in RULES:
        if pattern.search(text):
            return name
    return ""


def over_tls(service: str) -> str:
    """The name of ``service`` spoken inside TLS (``http`` is ``https``)."""
    return TLS_NAMES.get(service, service)


def usual_name(port: int, protocol: str = "tcp") -> str:
    """The service usually on ``port``: ``PORT_NAMES`` (TCP), else the system's list, else ""."""
    if protocol == "tcp" and port in PORT_NAMES:
        return PORT_NAMES[port]
    try:
        return socket.getservbyport(port, protocol)
    except OSError:
        return ""


def service_label(port: int, confirmed: str) -> str:
    """What the report shows for an open TCP port: the service its answer showed, else its
    usual name with a "?" (not confirmed), else ""."""
    if confirmed:
        return confirmed
    usual = usual_name(port)
    return f"{usual}?" if usual else ""
