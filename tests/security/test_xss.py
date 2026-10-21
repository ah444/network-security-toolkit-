r"""S5: dataset text (CSV headers, cells, class names, errors, predictions) must never
be inserted into the page as HTML.

Static rules for the v5.0 in-place fix of static/js/main.js (the module rewrite with
a DOM helper is v5.1):
1. Every ``${...}`` in a template literal that builds HTML goes through ``escapeHtml(...)``.
   Exempt: lines that assign to a safe text sink (``.textContent =`` / ``.text =``).
2. On lines that build HTML (``.innerHTML =``/``+=`` or an HTML string literal), every
   piece joined with ``+`` is a string literal, an ``escapeHtml(...)`` call, a template
   literal (checked by rule 1) or a known HTML-builder variable that is itself built
   under these rules.
3. No other raw-HTML sinks (insertAdjacentHTML, outerHTML, document.write).
The browser check with tests/fixtures/xss_header.csv is done by hand (checklist 0.3).
"""

import re

import pytest

MAIN_JS = "toolkit/web/static/js/main.js"
SAFE_SINK = re.compile(r"\.(textContent|text)\s*=")
HTML_LINE = re.compile(r"\.innerHTML\s*\+?=|['\"]\s*<|<[a-zA-Z/][^'\"]*['\"]")
HTML_BUILDERS = {"accHTML", "distHTML", "recHTML", "configHTML", "html", "tableHTML"}


@pytest.fixture(scope="module")
def js_lines(request):
    path = request.config.rootpath / MAIN_JS
    return path.read_text(encoding="utf-8").splitlines()


def _split_top_level_plus(expr: str) -> list[str]:
    parts, depth, quote, current = [], 0, None, ""
    for ch in expr:
        if quote:
            current += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"`":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "+" and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += ch
    parts.append(current.strip())
    return [p for p in parts if p]


def _operand_ok(operand: str) -> bool:
    operand = operand.rstrip(";").strip()
    return (
        bool(re.fullmatch(r"'[^']*'|\"[^\"]*\"|`[^`]*`", operand))
        or operand.startswith("escapeHtml(")
        or operand in HTML_BUILDERS
    )


def test_escape_helper_exists(js_lines):
    source = "\n".join(js_lines)
    assert "function escapeHtml(" in source
    for char in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
        assert char in source


def test_template_interpolations_are_escaped(js_lines):
    offenders = []
    for number, line in enumerate(js_lines, 1):
        if SAFE_SINK.search(line):
            continue
        for match in re.finditer(r"\$\{", line):
            if not line.startswith("escapeHtml(", match.end()):
                offenders.append(f"{number}: {line.strip()[:100]}")
                break
    assert offenders == [], "unescaped ${...} in HTML:\n" + "\n".join(offenders)


def test_concatenated_html_is_escaped(js_lines):
    offenders = []
    for number, line in enumerate(js_lines, 1):
        if SAFE_SINK.search(line) or not HTML_LINE.search(line):
            continue
        # ${...} interpolations are covered by rule 1; drop them before splitting on '+'.
        line = re.sub(r"\$\{[^{}]*\}", "X", line)
        if "+" not in line.replace("+=", ""):
            continue
        rhs = re.split(r"\+?=(?!=)", line, maxsplit=1)[-1] if "=" in line else line
        for operand in _split_top_level_plus(rhs):
            if not _operand_ok(operand):
                offenders.append(f"{number}: {operand[:60]!r} in {line.strip()[:80]}")
    assert offenders == [], "unescaped value joined into HTML:\n" + "\n".join(offenders)


def test_no_other_raw_html_sinks(js_lines):
    source = "\n".join(js_lines)
    for sink in ("insertAdjacentHTML", "outerHTML", "document.write"):
        assert sink not in source


def test_csv_header_rendered_as_text(js_lines):
    """The preview table builds header cells from column names: they must be escaped."""
    header_lines = [line for line in js_lines if "<th>" in line and "${" in line]
    assert header_lines, "preview header code not found"
    assert all("${escapeHtml(" in line for line in header_lines)
