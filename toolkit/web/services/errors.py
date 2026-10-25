"""What the web trainer's services refuse or fail with: a code and a message written for
the user (S8). The routes answer each with the HTTP status it has always had
(``toolkit.web.controllers.api.status_of``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Error codes (bundle refusals keep the code of their BundleError, e.g. BAD_SIGNATURE).
BAD_REQUEST = "BAD_REQUEST"  # the request body or a value in it is invalid
NO_DATASET = "NO_DATASET"
UNKNOWN_COLUMN = "UNKNOWN_COLUMN"
NOT_TRAINED = "NOT_TRAINED"
NO_MODEL_LOADED = "NO_MODEL_LOADED"
BUSY = "BUSY"  # 409: a training run is already in progress in this workspace
TOO_LARGE = "TOO_LARGE"  # 413
DATA_LIMIT = "DATA_LIMIT"  # 413: the datasets in memory would go over TOOLKIT_MAX_DATA_MB
UNREADABLE_FILE = "UNREADABLE_FILE"
TRAINING_FAILED = "TRAINING_FAILED"
CANCELLED = "CANCELLED"  # a job cancelled by the user (in the job, not an HTTP answer)
TIME_LIMIT = "TIME_LIMIT"  # a job stopped by TOOLKIT_JOB_MAX_MIN
BAD_HOST = "BAD_HOST"  # 400: the Host header isn't this app's (an IP address or localhost)
CROSS_SITE = "CROSS_SITE"  # 403
CSRF_TOKEN = "CSRF_TOKEN"  # 403: the X-CSRF-Token header is missing or not this session's
NOT_FOUND = "NOT_FOUND"  # 404
METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"  # 405
INTERNAL = "INTERNAL"  # 500: anything unexpected; details only in the server log



class ServiceError(Exception):
    """A request the web trainer can't do now; ``message`` is safe to show to the user."""

    def __init__(self, message: str, code: str = BAD_REQUEST,
                 details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = dict(details or {})


class Unavailable(ServiceError):
    """Not now, for the whole server (every workspace is busy training): try again later."""


class Failed(ServiceError):
    """Something that should have worked didn't; the details are in the server log."""
