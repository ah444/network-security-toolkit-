"""The access token (v5.2 step 2.4f): when the web app is opened to the network (``--host``),
anyone not on this computer types it once on the unlock page; this computer's own browser
(a loopback address) never needs it.

A new token is made at every start and shown only in the terminal (never logged: it is a
registered secret, ``toolkit.core.data.logs``). It is 16 characters from an alphabet without
the look-alikes 0, O, 1 and I (80 bits), grouped in fours to be typed. An unlocked session
holds the start's ``run_id``, so a token or an unlock of an earlier start never works
again. Five wrong tokens from one address within a minute lock that address out for the
rest of the minute.

A new token made on this computer (``lock_out``, HS1, RD1) changes the token and the
``run_id`` at once, so every session unlocked before is locked again; the jobs those
sessions asked for that still wait are cancelled and never start, a running one goes on.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import secrets
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from ...core.data.logs import add_secrets
from .errors import TOO_MANY_TRIES, WRONG_TOKEN, ServiceError
from .jobs import THIS_COMPUTER, Job, JobRunner

log = logging.getLogger(__name__)

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 32 symbols: no 0, O, 1 or I
TOKEN_LENGTH = 16  # 80 bits
MAX_TRIES = 5  # wrong tokens from one address ...
TRY_WINDOW = 60.0  # ... within this many seconds lock it out until the oldest is a minute old


def new_token() -> str:
    """A random access token, in groups of four: ``K7QX-M2PD-9RTA-W4HE``."""
    chars = "".join(secrets.choice(ALPHABET) for _ in range(TOKEN_LENGTH))
    return "-".join(chars[i:i + 4] for i in range(0, TOKEN_LENGTH, 4))


def normalized(text: str) -> str:
    """A typed token as it is compared: capitals, without spaces and dashes."""
    return "".join(ch for ch in text.upper() if ch not in " -\t")


def is_this_computer(address: str | None) -> bool:
    """True for a loopback address (IPv4, IPv6, or IPv4 inside IPv6): a request from this
    computer's own browser."""
    try:
        ip = ipaddress.ip_address(address or "")
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


@dataclass
class Access:
    """The token of this start and the wrong tries per address."""

    token: str = field(default_factory=new_token)
    run_id: str = field(default_factory=lambda: secrets.token_hex(16))
    _failures: dict[str, deque[float]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        add_secrets(self.token, normalized(self.token))

    def renew(self) -> str:
        """A new token and a new ``run_id``: every session unlocked before is locked again at
        once. The new token is a registered secret before it is used."""
        token = new_token()
        add_secrets(token, normalized(token))
        with self._lock:
            self.token, self.run_id = token, secrets.token_hex(16)
        return token

    def refusal(self, job: Job) -> str | None:
        """Why ``job`` may not start: it was asked for under an unlock older than the last new
        token (None: this computer's job, or one of the current unlock)."""
        if job.origin.unlock in (None, self.run_id):
            return None
        return (f"{job.kind.title} cancelled before it started: the access token was changed "
                "on the toolkit's computer.")

    def unlock(self, typed: str, address: str) -> str:
        """The ``run_id`` an unlocked session keeps, when ``typed`` is the token;
        WRONG_TOKEN (401) otherwise, TOO_MANY_TRIES (429) after MAX_TRIES wrong ones."""
        now = time.monotonic()
        with self._lock:
            tries = self._failures.setdefault(address, deque())
            while tries and now - tries[0] > TRY_WINDOW:
                tries.popleft()
            if len(tries) >= MAX_TRIES:
                raise ServiceError("Too many wrong access tokens from this computer: wait a "
                                   "minute, then try again.", TOO_MANY_TRIES)
            if hmac.compare_digest(normalized(typed).encode(), normalized(self.token).encode()):
                tries.clear()
                log.info("Unlocked for %s", address)
                return self.run_id
            tries.append(now)
        log.warning("A wrong access token from %s", address)
        raise ServiceError("That isn't the access token. It is shown in the terminal where "
                           "the toolkit was started.", WRONG_TOKEN)


def lock_out(access: Access, runners: Iterable[JobRunner]) -> tuple[str, int]:
    """A new token (``Access.renew``), then the jobs other computers asked for under an older
    unlock that still wait are cancelled (a running one goes on): the new token and how many
    were cancelled."""
    token = access.renew()
    cancelled = 0
    for runner in runners:
        for job in runner.active():
            reason = access.refusal(job)
            if reason is not None and runner.cancel_waiting(job, reason):
                cancelled += 1
    log.info("A new access token was made on this computer; waiting jobs of other computers "
             "cancelled: %d", cancelled)
    return token, cancelled


def remote_jobs(runners: Iterable[JobRunner]) -> list[tuple[JobRunner, Job]]:
    """The jobs other computers asked for that wait or run, with their runner."""
    return [(runner, job) for runner in runners for job in runner.active()
            if job.origin.source != THIS_COMPUTER]
