"""Security headers, the CSRF token and the cross-site check on every request (S5, S10),
and safe names for uploaded and exported files (S2-S4)."""

from __future__ import annotations

import hmac
import re
import secrets
from pathlib import PureWindowsPath
from uuid import uuid4

from flask import Flask, Response, request, session
from werkzeug.utils import secure_filename

from ..services.errors import CROSS_SITE, CSRF_TOKEN
from .api import error_response

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",                  # no inline scripts, no eval (S5 second layer)
    "style-src 'self'",                   # app.css only: no inline styles (v5.1 step 1.3b)
    "img-src 'self' data:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

# Browser features the app never uses, switched off for this site and anything it embeds.
PERMISSIONS_POLICY = ", ".join(f"{feature}=()" for feature in (
    "accelerometer", "camera", "geolocation", "gyroscope", "magnetometer", "microphone",
    "payment", "usb",
))

# Pages kept out of the browser cache: the trainer page carries the session's CSRF token.
NO_STORE_ENDPOINTS = frozenset({"pages.dynamic_ids_page"})


def _security_headers(response: Response) -> Response:
    """Security headers on every response (S5, S10)."""
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = PERMISSIONS_POLICY
    # API answers hold dataset values, results and models: never cached, on disk or shared.
    if request.path.startswith("/api/") or request.endpoint in NO_STORE_ENDPOINTS:
        response.headers["Cache-Control"] = "no-store"
    return response


CSRF_HEADER = "X-CSRF-Token"
STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def csrf_token() -> str:
    """The session's CSRF token (S10), made on first use and kept for the whole session,
    so tabs of the same browser share it. Pages put it in <meta name="csrf-token">."""
    token = session.get("csrf")
    if not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        session["csrf"] = token
    return token


def _check_state_changing_request() -> tuple[Response, int] | None:
    """CSRF protection (S10) for POST/PUT/PATCH/DELETE: the Origin header, if sent, must be
    this site, and the X-CSRF-Token header must be the session's token. JSON-only bodies
    and the SameSite=Strict cookie are further layers."""
    if request.method not in STATE_CHANGING_METHODS:
        return None
    origin = request.headers.get("Origin")
    if origin is not None and origin.rstrip("/") != request.host_url.rstrip("/"):
        return error_response("Cross-site request blocked.", 403, CROSS_SITE)
    expected = session.get("csrf")
    sent = request.headers.get(CSRF_HEADER, "")
    # A server restart changes the signing key, so the old page's token stops matching.
    if not isinstance(expected, str) or not hmac.compare_digest(sent.encode(), expected.encode()):
        return error_response("Missing or expired security token. Reload the page.", 403,
                              CSRF_TOKEN)
    return None


def register_security(app: Flask) -> None:
    """Adds the security headers to every answer and the origin and CSRF check to every
    state-changing request; templates get ``csrf_token``."""
    app.after_request(_security_headers)
    app.before_request(_check_state_changing_request)
    app.jinja_env.globals["csrf_token"] = csrf_token


_EXPORT_NAME = re.compile(r"[A-Za-z0-9_-]{1,80}")


def safe_upload_name(original: str | None) -> str:
    """Unique, sanitised file name for an upload (no path parts, no '..'), S2/S3."""
    cleaned = secure_filename(original or "") or "upload"
    return f"{uuid4().hex}_{cleaned}"


def safe_model_basename(name: object) -> str:
    """Return the export base name if it only uses letters, digits, '_' and '-' (1-80 chars).

    Raises ValueError otherwise, so a download name can never carry a path (S4).
    """
    if not isinstance(name, str) or not _EXPORT_NAME.fullmatch(name):
        raise ValueError("Model name may only use letters, digits, '_' and '-' (max 80).")
    return name


def display_name(original: str | None) -> str:
    """The uploaded file's own name for messages and the run record: no folder parts
    (from either OS), no control characters. Only safe_upload_name() names files on disk."""
    name = PureWindowsPath(original or "").name  # splits on both '\' and '/'
    name = "".join(ch for ch in name if ch.isprintable()).strip()[:120]
    return name or "upload"
