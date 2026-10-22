"""CSS that keeps both themes readable (step 0.7: U1, U2, U3, D2) and, since v5.1 step 1.3b,
one stylesheet with colour tokens for both pages (no inline styles, CSP style-src 'self')."""

import re

import pytest

from web_js import JS_DIR, TEMPLATE, js_sources

CSS_DIR = JS_DIR.parent / "css"
APP_CSS = CSS_DIR / "app.css"
TEMPLATES = sorted(TEMPLATE.parent.glob("*.html"))
DARK = "@media (prefers-color-scheme: dark)"
COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|(?<![\w-])(?:white|black)(?![\w-])")
# class names the modules build at runtime: "status-" + kind, "severity-" + level, ...
BUILT_PREFIXES = ("status-", "severity-", "cm-correct-", "cm-wrong-")


def _without_comments(text):
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


@pytest.fixture(scope="module")
def css():
    return _without_comments(APP_CSS.read_text(encoding="utf-8"))


def _block(text, start):
    """The body of the block whose opening brace ends at `start`."""
    depth = 1
    for index in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[start:index]
    raise AssertionError("unclosed block")


def _token_blocks(css):
    light = re.search(r"^:root\s*\{", css, re.M)
    dark = re.search(re.escape(DARK) + r"\s*\{\s*:root\s*\{", css)
    assert light and dark
    return _block(css, light.end()), _block(css, dark.end())


def test_one_stylesheet_for_both_pages():
    assert sorted(path.name for path in CSS_DIR.iterdir()) == ["app.css"]
    for template in TEMPLATES:
        links = re.findall(r"<link[^>]*>", template.read_text(encoding="utf-8"))
        assert len(links) == 1 and "filename='css/app.css'" in links[0], template.name


def test_no_inline_styles_in_templates():
    for template in TEMPLATES:
        text = template.read_text(encoding="utf-8")
        assert not re.search(r"\sstyle\s*=", text), template.name
        assert "<style" not in text, template.name


def test_colours_only_in_tokens(css):
    light, dark = _token_blocks(css)
    rest = css.replace(light, "").replace(dark, "")
    assert COLOUR.findall(rest) == []


def test_dark_theme_only_changes_tokens(css):
    """One dark block, and it only sets custom properties: no rule is written twice."""
    assert css.count(DARK) == 1
    media = re.search(re.escape(DARK) + r"\s*\{", css)
    _, dark = _token_blocks(css)
    assert re.sub(r"\s", "", _block(css, media.end()).replace(dark, "")) == ":root{}"
    declarations = [part.strip() for part in dark.split(";") if part.strip()]
    assert declarations and all(part.startswith("--") for part in declarations)


def test_every_token_is_defined(css):
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    assert set(re.findall(r"var\((--[\w-]+)\)", css)) <= defined


def test_important_only_on_hidden(css):
    selectors = re.findall(r"([^{};]+)\{[^{}]*!important", css)
    assert [selector.strip() for selector in selectors] == ["[hidden]"]
    assert css.count("!important") == 1


def test_every_class_is_used(css):
    """No rules for markup that no longer exists (.workflow-steps, .model-checkbox-*, .alert ...)."""
    used = " ".join([path.read_text(encoding="utf-8") for path in TEMPLATES]
                    + list(js_sources().values()))
    classes = set(re.findall(r"\.([a-zA-Z][\w-]*)", re.sub(r"\{[^{}]*\}", "{}", css)))
    unused = [name for name in sorted(classes)
              if name not in used and not name.startswith(BUILT_PREFIXES)]
    assert unused == []


def test_no_global_white_text_override(css):
    """U1: `*:not(h1):not(h2) { color: white !important }` made light mode white-on-white."""
    assert "*:not(h1)" not in css


def test_braces_balanced(css):
    """U2: an unclosed @media block once made half of style2.css dark-mode-only."""
    depth = 0
    for char in css:
        depth += {"{": 1, "}": -1}.get(char, 0)
        assert depth >= 0, "extra closing brace"
    assert depth == 0, f"{depth} unclosed block(s)"


def test_no_web_font_dependency(css):
    """U3: 'Inter' was named but never loaded; use the system font stack (works offline)."""
    assert "Inter" not in css
    assert "@import" not in css and "url(" not in css


def test_focus_is_visible(css):
    assert ":focus-visible" in css
    assert not re.search(r"outline\s*:\s*none", css)
