"""What a scan may send to each online service, or why it sends nothing (step A5, HS4): the
lines of the scan's notice, from its request and the keys that are set. In a module of its own
since F6a (``scan`` re-exports it), so the scan stays under 400 lines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...data.keys import KEY_VARIABLES
from ...data.recon.nvd import NVD_INTERVAL
from .targets import is_public_ip

if TYPE_CHECKING:
    from .scan import ScanRequest


@dataclass(frozen=True)
class OnlinePlan:
    """What a scan may send to each online service, or why it sends nothing (step A5)."""

    nvd: str
    ai: str
    abuseipdb: str


def _turned_off(request: ScanRequest, flag: str) -> str:
    """Why a service with its key is off: the answer no, its switch on the web page, or its
    ``--no-`` flag (HS4)."""
    if request.interactive:
        return "off: you said no"
    return "off: turned off on the page" if request.from_page else f"off: turned off ({flag})"


def _ai_plan(request: ScanRequest, keys: Mapping[str, str]) -> str:
    """The AI explanation: on, or why not (no key, turned off, the answer no)."""
    if request.use_ai:
        return "on, the software name and version only"
    if not keys.get("openrouter"):
        return f"off: no {KEY_VARIABLES['openrouter']}"
    return _turned_off(request, "--no-ai")


def _abuse_plan(request: ScanRequest, keys: Mapping[str, str]) -> str:
    """The AbuseIPDB check: for the public addresses only, or why not."""
    if not any(is_public_ip(target) for target in request.targets):
        return "off: no public address (a private one is never sent)"
    if not keys.get("abuseipdb"):
        return f"off: no {KEY_VARIABLES['abuseipdb']}"
    if request.interactive:
        return "asked before each public address"
    return "on, the public addresses only" if request.abuseipdb else \
        _turned_off(request, "--no-abuseipdb")


def online_plan(request: ScanRequest, keys: Mapping[str, str]) -> OnlinePlan:
    """What this scan may send online and why not, from the request (the flags or the answers)
    and the keys that are set."""
    pace = "" if keys.get("nvd") else (
        f"; no {KEY_VARIABLES['nvd']}, so one request every {NVD_INTERVAL[False]:g} s")
    nvd = f"on, the software name and version only{pace}" if request.nvd else \
        _turned_off(request, "--no-nvd")
    return OnlinePlan(nvd, _ai_plan(request, keys), _abuse_plan(request, keys))
