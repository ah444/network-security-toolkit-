"""Security headers and the cross-site check on every request (S5, S10), and safe names
for uploaded and exported files (S2-S4)."""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from uuid import uuid4

from flask import Flask, Response, request
from werkzeug.utils import secure_filename

from .api import CROSS_SITE, error_response

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",                  # no inline scripts, no eval (S5 second layer)
    "style-src 'self' 'unsafe-inline'",   # inline style="" is still used; tightened in v5.1
    "img-src 'self' data:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


def _security_headers(response: Response) -> Response:
    """Security headers on every response (S5, S10)."""
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _block_cross_site_requests() -> tuple[Response, int] | None:
    """Interim CSRF protection (S10) until the v5.1 token: reject state-changing
    requests whose Origin header names another site."""
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        origin = request.headers.get("Origin")
        if origin is not None and origin.rstrip("/") != request.host_url.rstrip("/"):
            return error_response("Cross-site request blocked.", 403, CROSS_SITE)
    return None


def register_security(app: Flask) -> None:
    app.after_request(_security_headers)
    app.before_request(_block_cross_site_requests)


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
