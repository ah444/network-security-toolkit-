"""Request and response helpers shared by the routes (S8, S9, S10 interim).

Every error answer has one shape: ``{"error": {"code", "message", "details"?}}``, with
the HTTP status the route always used. Messages are written for the user and never
contain exception text, tracebacks or server paths (S8).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

from ..services.errors import (
    BAD_REQUEST,
    BUSY,
    DATA_LIMIT,
    INTERNAL,
    METHOD_NOT_ALLOWED,
    NOT_FOUND,
    TOO_LARGE,
    Failed,
    ServiceError,
    Unavailable,
)
from .state import state

log = logging.getLogger(__name__)

Answer = Response | tuple[Response, int]


class ApiError(ServiceError):
    """An error answer raised from anywhere in a route, with its HTTP status."""

    def __init__(self, message: str, code: str = BAD_REQUEST, status: int = 400,
                 details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message, code, details)
        self.status = status


def error_response(message: str, status: int = 400, code: str = BAD_REQUEST,
                   **details: Any) -> tuple[Response, int]:
    """The error answer: ``{"error": {"code", "message", "details"}}`` (details if any)."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return jsonify({"error": error}), status


def json_body() -> dict[str, Any]:
    """Return the request's JSON object, or raise ApiError (S9, S10 interim).

    Requiring a JSON object (Content-Type: application/json) also means a plain
    HTML form on another website can't trigger these routes.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object (Content-Type: application/json).")
    return data


def query() -> dict[str, str]:
    """The query string of a GET request, one value per name."""
    return request.args.to_dict()


STATUS = {NOT_FOUND: 404, BUSY: 409, DATA_LIMIT: 413, TOO_LARGE: 413}  # else 400


def status_of(error: ServiceError) -> int:
    """The HTTP status of a refusal: a route's own, else by its kind or code."""
    if isinstance(error, ApiError):
        return error.status
    if isinstance(error, Unavailable):
        return 503
    if isinstance(error, Failed):
        return 500
    return STATUS.get(error.code, 400)


def _api_error(error: ServiceError) -> tuple[Response, int]:
    return error_response(error.message, status_of(error), error.code, **error.details)


def _upload_too_large(_error: HTTPException) -> tuple[Response, int]:
    return error_response(f"Upload too large (limit: {state().config.max_upload_mb} MB).", 413,
                          TOO_LARGE)


HTTP_CODES = {404: NOT_FOUND, 405: METHOD_NOT_ALLOWED}


def _http_error(error: HTTPException) -> HTTPException | tuple[Response, int]:
    """JSON for the API's own 404/405; pages keep Flask's HTML answers."""
    if request.path.startswith("/api/") and error.code in HTTP_CODES:
        message = ("No such API route." if error.code == 404
                   else f"{request.method} is not allowed on this API route.")
        return error_response(message, error.code, HTTP_CODES[error.code])
    return error


def _unexpected(_error: Exception) -> tuple[Response, int]:
    """Anything unexpected: logged with its traceback, answered without details (S8)."""
    log.exception("Unexpected error in %s %s", request.method, request.path)
    return error_response("Something went wrong on the server. Details are in the server log.",
                          500, INTERNAL)


def register_error_handlers(app: Flask) -> None:
    """One JSON error shape for API errors, too-large uploads and unexpected errors."""
    app.register_error_handler(ServiceError, _api_error)
    app.register_error_handler(413, _upload_too_large)
    app.register_error_handler(HTTPException, _http_error)
    app.register_error_handler(Exception, _unexpected)
