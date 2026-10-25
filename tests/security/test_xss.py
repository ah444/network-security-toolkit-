"""S5: dataset text (CSV headers, cells, class names, errors, predictions) must never be
inserted into the page as HTML.

Since v5.1 step 1.3a the page's ES modules build every element with dom.js (``el()``,
``setContent()``), which only ever sets text, so the rule is simple: no raw-HTML sink in
any module, and no HTML written as a string. The browser check with
tests/fixtures/xss_header.csv is part of each step's real check.
"""

import re

import pytest

from web_js import JS_DIR, js_sources

SINKS = ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "DOMParser",
         "createContextualFragment", "srcdoc", "eval(", "new Function"]


@pytest.mark.parametrize("sink", SINKS)
def test_no_raw_html_sinks(sink):
    offenders = [name for name, code in js_sources().items() if sink in code]
    assert offenders == [], f"{sink} used in {offenders}"


def test_no_html_written_as_strings():
    """A string that starts a tag ('<td', "<p>", `<div ...`) would be HTML built by hand."""
    tag_string = re.compile(r"""["'`]\s*<\s*/?[a-zA-Z]""")
    offenders = [f"{name}: {line.strip()[:80]}" for name, code in js_sources().items()
                 for line in code.splitlines() if tag_string.search(line)]
    assert offenders == []


def test_no_inline_styles_or_handlers_from_js():
    """The CSP blocks inline styles and handlers, so they could only break the page."""
    for name, code in js_sources().items():
        assert ".style." not in code and ".style =" not in code, name
        assert not re.search(r"""setAttribute\(\s*["'](style|on\w+)""", code), name
        assert not re.search(r"\.on[a-z]+\s*=", code), name  # element.onclick = ...


def test_el_refuses_style_and_handlers():
    dom = (JS_DIR / "dom.js").read_text(encoding="utf-8")
    refused = re.search(r"const REFUSED = /(.+)/i;", dom).group(1)
    for prop in ("onclick", "onerror", "style", "innerHTML", "outerHTML", "srcdoc"):
        assert re.match(refused, prop, re.I), prop
    for prop in ("class", "title", "value", "htmlFor", "colspan"):
        assert not re.match(refused, prop, re.I), prop


def test_csv_header_rendered_as_text():
    """The preview table's header cells come from the CSV's column names (named in the 5.0
    section 8 audit): they go through dom.js table(), which makes each one a text child."""
    sources = js_sources()
    assert 'table("preview-table", columns,' in sources["data.js"]
    assert 'el("thead", {}, el("tr", {}, headers.map(text => el("th", {}, text))))' in sources["dom.js"]


def test_text_is_set_with_text_content():
    dom = js_sources()["dom.js"]
    assert "node.textContent = value" in dom  # the "text" prop
    assert "node.append(child instanceof Node ? child : String(child))" in dom  # text nodes
