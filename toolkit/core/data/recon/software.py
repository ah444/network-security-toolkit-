"""The software in a banner: "product version" (e.g. "OpenSSH 8.9"), looked up online (S12).

``RULES`` are tried in order; each is a pattern and how its match is named. A banner no rule
fits goes to the generic "Name/1.2" or "Name 1.2" pattern. The result is short and
allow-listed: the only part of a banner that ever leaves the computer, so it never carries
an address (an IPv4-like version is refused).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from .text import clean_text

VERSION = r"(\d+(?:\.\d+)+)"
GENERIC = re.compile(r"([A-Za-z][A-Za-z0-9\-]{1,30}(?: Server)?)[/_ ]v?" + VERSION)
SSH_SOFTWARE = re.compile(r"^SSH-\d+(?:\.\d+)?-(\S+)")
SSH_NAME_VERSION = re.compile(r"^([A-Za-z][A-Za-z0-9]*)[_\-]v?" + VERSION)
IPV4_LIKE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
NOT_PRODUCTS = {  # protocol names and words around a number, not software
    "http", "https", "ssh", "rtsp", "sip", "version", "release", "build", "revision", "rev",
    "server", "host", "to", "at", "on", "of", "is", "for", "from"}
MAX_LENGTH = 60


@dataclass(frozen=True)
class Rule:
    """A banner pattern and the name of what it matched (from the match's groups)."""

    pattern: re.Pattern[str]
    name: Callable[[re.Match[str]], str | None]


def _ssh(match: re.Match[str]) -> str | None:
    """The SSH software (``SSH-2.0-OpenSSH_8.9p1``): only the token after the protocol, so a
    comment after it (``Ubuntu-3ubuntu0.1``) or Windows in ``OpenSSH_for_Windows_8.1`` is never
    taken for the product."""
    token = match.group(1).replace("OpenSSH_for_Windows_", "OpenSSH_")
    found = SSH_NAME_VERSION.match(token)
    return f"{found.group(1)} {found.group(2)}" if found else None


RULES: tuple[Rule, ...] = (
    Rule(SSH_SOFTWARE, _ssh),
    # MariaDB's greeting puts "5.5.5-" (for old MySQL clients) before its own version
    Rule(re.compile(r"^MySQL (?:5\.5\.5-)?" + VERSION + r"-MariaDB"),
         lambda m: f"MariaDB {m.group(1)}"),
    Rule(re.compile(r"redis_version:" + VERSION), lambda m: f"Redis {m.group(1)}"),
    Rule(re.compile(r"^VERSION " + VERSION + r"$"),  # Memcached's answer to "version"
         lambda m: f"Memcached {m.group(1)}"),
)


def _generic(text: str) -> str | None:
    """The first "Name/1.2" or "Name 1.2" that is a product with a version, not an address."""
    for match in GENERIC.finditer(text):
        name, version = match.groups()
        if name.lower() not in NOT_PRODUCTS and not IPV4_LIKE.match(version):
            return f"{name} {version}"
    return None


def product_version(banner: str) -> str | None:
    """"product version" from a banner (e.g. "OpenSSH 8.9"), or None. This short,
    allow-listed text is the only part of a banner that ever leaves the computer (S12)."""
    text = clean_text(banner).strip()
    for rule in RULES:
        match = rule.pattern.search(text)
        if match:
            found = rule.name(match)
            return found[:MAX_LENGTH] if found else None
    found = _generic(text)
    return found[:MAX_LENGTH] if found else None
