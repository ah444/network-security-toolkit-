"""How long one lookup may take, which failures are expected, and how they are named
(R3): a failed lookup is recorded and the scan goes on.
"""

import requests

INTEL_TIMEOUT = 15  # seconds per lookup: a slow service can't stop the scan
INTEL_WORKERS = 4  # lookups at a time (step 1.4e); more wouldn't help: NVD limits the rate


def describe(error: BaseException) -> str:
    if isinstance(error, requests.Timeout):
        return "timed out"
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}"
    if isinstance(error, requests.ConnectionError):
        return "no connection"
    return type(error).__name__


LOOKUP_ERRORS = (requests.RequestException, ValueError, KeyError, IndexError, TypeError)
