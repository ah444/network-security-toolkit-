"""Text that came from the network, made safe to show and export (S13, S14)."""

import re

# CSI sequences, OSC sequences (ended by BEL or ESC + backslash) and 2-byte escapes.
ANSI_ESCAPE = re.compile("\x1b(?:\\[[0-?]*[ -/]*[@-~]|\\][^\x07\x1b]*(?:\x07|\x1b\\\\)?|[@-Z\\\\-_])")
CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # keeps \t and \n
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def clean_text(text, limit=None):
    """Printable text: terminal escape sequences and control characters removed."""
    text = CONTROL.sub("", ANSI_ESCAPE.sub("", str(text)))
    if limit and len(text) > limit:
        text = text[:limit - 3] + "..."
    return text


def one_line(text, limit=None):
    return clean_text(" ".join(str(text).split()), limit)


def csv_cell(value):
    """A spreadsheet-safe cell: text that starts like a formula gets a leading ' (S14)."""
    if isinstance(value, str) and value.startswith(FORMULA_START):
        return "'" + value
    return value


AI_MAX_CHARS = 2000
CVE_ANYWHERE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
HTML_TAG = re.compile(r"<[^<>]{0,200}>")


def untrusted_text(reply):
    """An AI reply as inert plain text (S13): no escapes, control characters or HTML tags,
    no CVE IDs (CVEs come only from NVD), length-capped."""
    text = HTML_TAG.sub("", clean_text(reply if isinstance(reply, str) else ""))
    text = CVE_ANYWHERE.sub("[CVE ID removed: see the NVD list]", text)
    return clean_text(text, AI_MAX_CHARS).strip()
