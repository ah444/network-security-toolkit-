"""Step 0.7 (U1, U2, U3, D2): CSS that keeps both themes readable."""

import re

import pytest

CSS_DIR = "tools/IDS/IDS_Web_CIC_Dynamic/static/css"
FILES = ["style.css", "style2.css"]


@pytest.fixture(params=FILES)
def css(request, repo_root):
    return request.param, (repo_root / CSS_DIR / request.param).read_text(encoding="utf-8")


def _without_comments(text):
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def test_no_global_white_text_override(css):
    """U1: `*:not(h1):not(h2) { color: white !important }` made light mode white-on-white."""
    name, text = css
    assert "*:not(h1)" not in _without_comments(text), name


def test_sections_do_not_force_white_text(css):
    name, text = css
    for selector in (".container", ".section"):
        for block in re.findall(re.escape(selector) + r"(?::hover)?\s*\{([^}]*)\}",
                                _without_comments(text)):
            assert not re.search(r"(^|;|\s)color\s*:\s*white", block), f"{name} {selector}"


def test_braces_balanced(css):
    """U2: an unclosed @media block made half of style2.css dark-mode-only."""
    name, text = css
    depth = 0
    for char in _without_comments(text):
        depth += {"{": 1, "}": -1}.get(char, 0)
        assert depth >= 0, f"{name}: extra closing brace"
    assert depth == 0, f"{name}: {depth} unclosed block(s)"


def test_dark_media_block_ends_before_shared_rules(repo_root):
    """Rules after the dark-mode block (tables, confusion matrix, messages) apply in both themes."""
    text = _without_comments((repo_root / CSS_DIR / "style2.css").read_text(encoding="utf-8"))
    media = re.search(r"@media \(prefers-color-scheme: dark\)\s*\{", text)
    depth, end = 1, None
    for index in range(media.end(), len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            end = index
            break
    assert end is not None
    inside = text[media.end():end]
    for selector in (".preview-table", ".confusion-matrix", ".success-message", ".stats-grid"):
        assert selector not in inside, f"{selector} must not be dark-mode-only"


def test_no_web_font_dependency(css):
    """U3: 'Inter' was named but never loaded; use the system font stack (works offline)."""
    name, text = css
    assert "Inter" not in text, name
    assert "@import" not in text and "fonts.googleapis" not in text, name


def test_focus_is_visible(css):
    name, text = css
    clean = _without_comments(text)
    assert ":focus-visible" in clean, f"{name}: no visible focus style"
    assert not re.search(r"outline\s*:\s*none", clean), f"{name}: focus outline removed"


def test_stat_values_not_forced_dark(repo_root):
    """Seen in the 0.3 browser check: #2d3748 !important made them invisible in dark mode."""
    text = (repo_root / CSS_DIR / "style2.css").read_text(encoding="utf-8")
    block = re.search(r"\.stat-item\s*\{([^}]*)\}", text).group(1)
    assert "!important" not in block
