"""Test helpers for colour contrast (v5.2): the stylesheet's tokens in each theme, a token's
colour (its var() chain followed), and the WCAG 2.1 contrast ratio of two colours."""

import re

from web_js import JS_DIR

APP_CSS = JS_DIR.parent / "css" / "app.css"
LIGHT = ":root {"
DARK = ':root[data-theme="dark"] {'  # the same values as the system's dark (test_css.py)


def stylesheet():
    """app.css without its comments."""
    return re.sub(r"/\*.*?\*/", "", APP_CSS.read_text(encoding="utf-8"), flags=re.S)


def tokens(css, block_start):
    """The custom properties of the block that starts with ``block_start``."""
    start = css.index(block_start) + len(block_start)
    depth, end = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(css[end], 0)
        end += 1
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", css[start:end]))


def colour(name, theme, light):
    """A token's colour in ``theme`` (its var() chain followed; the light block holds what a
    dark block doesn't change)."""
    value = theme.get(name, light.get(name))
    while value.startswith("var("):
        inner = value[4:-1]
        value = theme.get(inner, light.get(inner))
    return value


def _luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(one, two):
    """The WCAG 2.1 contrast ratio of two ``#rrggbb`` colours."""
    high, low = sorted((_luminance(one), _luminance(two)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def themes():
    """(light tokens, dark tokens)."""
    css = stylesheet()
    return tokens(css, LIGHT), tokens(css, DARK)
