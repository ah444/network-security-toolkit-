"""Security headers, the Host check, the CSRF token and the cross-site check on every request
(S5, S10, v5.2 step 2.4a), and safe names for uploaded and exported files (S2-S4)."""

from __future__ import annotations

import hmac
import ipaddress
import logging
import re
import secrets
from collections.abc import Callable
from pathlib import PureWindowsPath
from uuid import uuid4

from flask import Flask, Response, request, session
from werkzeug.utils import secure_filename

from ..services.errors import BAD_HOST, CROSS_SITE, CSRF_TOKEN
from .api import error_response

log = logging.getLogger(__name__)

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


DEFAULT_PORTS = {"http": 80, "https": 443}  # a Host without a port names the scheme's own
MAX_HOSTS_LOGGED = 20  # each refused Host is logged once, up to this many


def is_own_host(host: str, port: int, scheme: str = "http") -> bool:
    """Whether a request's Host header names this app (step 2.4a): an IP address (IPv6 in
    brackets) or ``localhost``, on the app's ``port``. Any other name is refused: a page on
    another site whose name was made to point at 127.0.0.1 (DNS rebinding) would be the same
    site as this app to the browser, so the Origin check and the CSRF token alone can't stop
    it; its requests carry that name."""
    if host.startswith("["):  # [IPv6] or [IPv6]:port
        name, bracket, rest = host[1:].partition("]")
        if not bracket or (rest and not rest.startswith(":")):
            return False
        given, versions = rest[1:], (6,)
    else:  # IPv4, localhost or a name, with or without :port
        name, _, given = host.partition(":")
        versions = (4,)
    if not given:
        given = str(DEFAULT_PORTS.get(scheme, 0))
    if given != str(port):  # also a second ":" (an IPv6 address without brackets)
        return False
    if versions == (4,) and name.lower() == "localhost":
        return True
    if "%" in name:  # an IPv6 zone: never this app's address
        return False
    try:
        return ipaddress.ip_address(name).version in versions
    except ValueError:
        return False


def _host_check(port: int) -> Callable[[], Response | tuple[Response, int] | None]:
    """The before-request check of the Host header (``is_own_host``): 400 ``BAD_HOST``,
    as JSON for the API, else as a line of text with the address to open."""
    logged: set[str] = set()

    def check() -> Response | tuple[Response, int] | None:
        host = request.host
        if is_own_host(host, port, request.scheme):
            return None
        if host not in logged and len(logged) < MAX_HOSTS_LOGGED:
            logged.add(host)
            log.warning("Refused a request for the host %r: not this app's address", host[:100])
        message = (f"This address isn't the toolkit's: open http://127.0.0.1:{port}/ "
                   "instead.")
        if request.path.startswith("/api/"):
            return error_response(message, 400, BAD_HOST)
        return Response(message + "\n", 400, mimetype="text/plain")

    return check


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


def register_security(app: Flask, port: int) -> None:
    """Adds the security headers to every answer, the Host check (the app on ``port``) to
    every request, before anything else, and the origin and CSRF check to every
    state-changing request; templates get ``csrf_token``."""
    app.after_request(_security_headers)
    app.before_request(_host_check(port))
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
