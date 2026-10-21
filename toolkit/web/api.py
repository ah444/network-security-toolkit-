"""Request and response helpers shared by the routes (S8, S9, S10 interim)."""

from flask import jsonify, request

from ..config import SplitConfig
from .state import state

class BadRequestBody(Exception):
    """The request body or one of its values is invalid (answered with HTTP 400)."""


def json_body():
    """Return the request's JSON object, or raise BadRequestBody (S9, S10 interim).

    Requiring a JSON object (Content-Type: application/json) also means a plain
    HTML form on another website can't trigger these routes.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequestBody("Request body must be a JSON object (Content-Type: application/json).")
    return data


def int_param(data, key, default, low, high):
    """Read an integer value from a JSON body, clamped to [low, high]."""
    value = data.get(key, default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise BadRequestBody(f"'{key}' must be a whole number.") from None
    return max(low, min(high, number))


def test_size_param(data):
    """Share of rows held out for the test split (clamped to 10-50 %)."""
    value = data.get("test_size", SplitConfig().test_size)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BadRequestBody("'test_size' must be a number between 0.1 and 0.5.") from None
    return max(0.1, min(0.5, number))


def error_response(message, status=400, **extra):
    return jsonify({"error": message, **extra}), status


def _bad_request_body(error):
    return jsonify({"error": str(error)}), 400


def _upload_too_large(_error):
    return jsonify({"error": f"Upload too large (limit: {state().config.max_upload_mb} MB)."}), 413


def register_error_handlers(app):
    """JSON answers for invalid request bodies (400) and too-large uploads (413)."""
    app.register_error_handler(BadRequestBody, _bad_request_body)
    app.register_error_handler(413, _upload_too_large)
