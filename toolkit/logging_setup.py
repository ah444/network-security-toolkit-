"""Logging for the tools: a rotating file per tool, console output, secrets redacted (D17).

Each tool calls ``setup_logging(tool)`` once when it starts (never on import):

- file ``instance/logs/<tool>.log``, UTF-8, rotated at ``max_bytes`` with ``backups`` old
  files; its level comes from ``TOOLKIT_LOG_LEVEL`` (default INFO);
- console (stderr) at ``console_level``;
- one plain-text line per record: ISO time, level, logger name, message;
- library warnings (``warnings.warn``) are logged too, so they stay visible (B19);
- third-party HTTP libraries are capped at WARNING, so DEBUG shows our own lines only.

Redaction happens in the formatter, so it also covers tracebacks and third-party records:
the exact values of known secrets (the API keys and ``TOOLKIT_SECRET_KEY`` from the
environment, plus any registered with ``add_secrets``) and the values after
``Authorization``, ``Bearer``, ``api_key`` / ``apiKey`` / ``Key`` / ``token`` / ``secret`` /
``password``. Dataset values are never logged: log counts, names and reasons instead.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import instance_dir, log_level

REDACTED = "[REDACTED]"
SECRET_VARIABLES = ("OPENROUTER_API_KEY", "ABUSEIPDB_API_KEY", "NVD_API_KEY", "TOOLKIT_SECRET_KEY")
MIN_SECRET_LENGTH = 8  # shorter values would redact ordinary words
QUIET_LIBRARIES = ("urllib3", "requests", "charset_normalizer")
LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

_NAMED_VALUE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|x-api-key|key|token|secret|password)"
    r"(['\"]?\s*[:=]\s*['\"]?)(?:bearer\s+)?[^\s'\",;&}]+")
_BEARER = re.compile(r"(?i)\b(?:bearer)\s+[A-Za-z0-9._~+/=-]{8,}")

_secrets: set[str] = set()
_handlers: list[logging.Handler] = []


def add_secrets(*values: str) -> None:
    """Values that must never appear in a log line (e.g. keys read from a file)."""
    _secrets.update(v for v in (value.strip() for value in values if value)
                    if len(v) >= MIN_SECRET_LENGTH)


def forget_secrets() -> None:
    _secrets.clear()


def redact(text: str) -> str:
    for secret in sorted(_secrets, key=len, reverse=True):  # longest first
        text = text.replace(secret, REDACTED)
    text = _BEARER.sub(f"Bearer {REDACTED}", text)
    return _NAMED_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(tool: str, *, console_level: int = logging.WARNING,
                  environ: Mapping[str, str] | None = None,
                  max_bytes: int = 1_000_000, backups: int = 3) -> Path:
    """Configure the root logger for one tool; calling it again replaces the handlers.

    Returns the path of the log file.
    """
    environ = os.environ if environ is None else environ
    file_level = log_level(environ)
    add_secrets(*(environ.get(name, "") for name in SECRET_VARIABLES))

    path = instance_dir(environ) / "logs" / f"{tool}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = RedactingFormatter(LINE_FORMAT, TIME_FORMAT)
    file_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups,
                                       encoding="utf-8")
    file_handler.setLevel(file_level)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)

    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()
    _handlers[:] = [file_handler, console]
    for handler in _handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(min(file_level, console_level))
    for name in QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(max(logging.WARNING, file_level))
    logging.captureWarnings(True)
    return path
