"""The scan report's HTML (v5.4 F5), the only module outside the web that uses Jinja2.

1. reads the page (scan_report.html) and its look (scan_report.css), files of this package;
2. fills the page with the report's data (services/recon/scan_report.py), every value escaped;
3. makes a file that stands alone: the CSS inside, a policy that allows that style only, so
   the browser runs no script and loads nothing.

Jinja2 does this job and nothing else (DECISIONS F5): the template is ours, never made from
data; the data is plain text, numbers, lists and dicts; autoescape is on; a name the data
lacks is an error; the sandbox refuses unsafe attributes. See also:
tests/recon/test_scan_report.py (XSS in every field, no |safe in the template),
tests/test_architecture.py (JINJA2_OUTSIDE_WEB: this module only).
"""

import base64
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment
from markupsafe import Markup

TEMPLATE = Path(__file__).with_name("scan_report.html")
STYLESHEET = Path(__file__).with_name("scan_report.css")


def stylesheet() -> str:
    """The report's CSS."""
    return STYLESHEET.read_text(encoding="utf-8")


def report_html(report: Mapping[str, Any]) -> str:
    """The report page as one file: the CSS inside, allowed by its hash; no script."""
    css = stylesheet()
    digest = base64.b64encode(hashlib.sha256(css.encode("utf-8")).digest()).decode("ascii")
    pages = ImmutableSandboxedEnvironment(autoescape=True, undefined=StrictUndefined)
    page = pages.from_string(TEMPLATE.read_text(encoding="utf-8"))
    # Markup: our own CSS file put in as it is (escaping would break it); never data
    return page.render(report=report, css=Markup(css), style_hash=f"sha256-{digest}")
