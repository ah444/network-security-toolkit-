"""How long one lookup may take, which failures are expected, and how they are named
(R3): a failed lookup is recorded and the scan goes on. The scan's HTTP session counts what it
sends (step A5).
"""

import threading
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

import requests

INTEL_TIMEOUT = 15  # seconds per lookup: a slow service can't stop the scan
INTEL_WORKERS = 4  # lookups at a time (step 1.4e); more wouldn't help: NVD limits the rate


def describe(error: BaseException) -> str:
    """A short name for a failed lookup (``timed out``, ``HTTP 404``, ``no connection``, or the
    error's type), without the URL, so a key in it never reaches a report."""
    if isinstance(error, requests.Timeout):
        return "timed out"
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}"
    if isinstance(error, requests.ConnectionError):
        return "no connection"
    return type(error).__name__


def host_of(url: str) -> str:
    """The host name in ``url`` ("" if none)."""
    return urlsplit(url).hostname or ""


class Session(requests.Session):
    """The scan's HTTP session, one per scan, shared by the gateways; it counts the requests it
    sends to each host (``sent``), so the scan can say what went online (step A5)."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: Counter[str] = Counter()
        self._counting = threading.Lock()

    def request(self, method: str | bytes, url: str | bytes, *args: Any,
                **kwargs: Any) -> requests.Response:
        """Counts the request for its host (also one that then fails), then sends it."""
        host = host_of(url.decode() if isinstance(url, bytes) else url)
        with self._counting:
            self.sent[host] += 1
        return super().request(method, url, *args, **kwargs)

LOOKUP_ERRORS = (requests.RequestException, ValueError, KeyError, IndexError, TypeError)
