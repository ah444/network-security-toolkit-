"""Reputation of a public IP address from AbuseIPDB (asked only after consent)."""

import requests

from .findings import IntelFinding
from .lookup import INTEL_TIMEOUT

ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"


class AbuseIpdbProvider:
    """Reputation of a public IP address (IntelProvider, kind "ip_reputation")."""

    name = "AbuseIPDB"
    kind = "ip_reputation"

    def __init__(self, session: requests.Session, key: str, timeout: float = INTEL_TIMEOUT,
                 max_age: int = 90) -> None:
        self.session = session
        self.key = key
        self.timeout = timeout
        self.max_age = max_age

    def lookup(self, subject: str) -> IntelFinding:
        """AbuseIPDB's ``data`` for the address ``subject`` (reports from the last ``max_age``
        days). The caller sends only a public address, after consent."""
        response = self.session.get(
            ABUSEIPDB_URL, params={"ipAddress": subject, "maxAgeInDays": str(self.max_age)},
            headers={"Accept": "application/json", "Key": self.key}, timeout=self.timeout)
        response.raise_for_status()
        return IntelFinding(data=response.json().get("data", {}))
