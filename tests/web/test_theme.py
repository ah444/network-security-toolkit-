"""v5.2 step 2.1: the theme button. Auto (the default) follows the system, Light and Dark
override it; the choice is kept in the browser. ``theme.js`` is a classic script in the
<head> of both pages, so a page is drawn in its theme from the start."""

import re

import pytest

from web_js import JS_DIR, TEMPLATE, js_sources

TEMPLATES = sorted(TEMPLATE.parent.glob("*.html"))
THEME_SCRIPT = "<script src=\"{{ url_for('static', filename='js/theme.js') }}\"></script>"


@pytest.fixture(scope="module")
def theme():
    return js_sources()["theme.js"]


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_both_pages_load_the_theme_before_the_stylesheet(template):
    """A classic script (not a module, not deferred) runs before the page is drawn."""
    html = template.read_text(encoding="utf-8")
    head = html[:html.index("</head>")]
    assert head.count(THEME_SCRIPT) == 1
    assert head.index(THEME_SCRIPT) < head.index("<link")


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_both_pages_have_the_button_in_the_navbar(template):
    html = template.read_text(encoding="utf-8")
    nav = re.search(r'<nav class="navbar">.*?</nav>', html, re.S).group(0)
    buttons = re.findall(r"<button[^>]*>[^<]*</button>", nav)
    # the Menu button (v5.2 step 2.2a) opens the links on small screens: tests/web/test_responsive.py
    assert buttons[1:] == ['<button type="button" id="themeToggle" class="nav-link theme-toggle" '
                           'hidden>Theme: Auto</button>']
    assert len(buttons) == 2 and 'id="navToggle"' in buttons[0]


def test_no_inline_script_anywhere():
    """The CSP allows scripts from the site's own files only (script-src 'self')."""
    for template in TEMPLATES:
        for tag in re.findall(r"<script[^>]*>", template.read_text(encoding="utf-8")):
            assert " src=" in tag, (template.name, tag)


def test_the_three_choices_and_auto_by_default(theme):
    assert 'const CHOICES = ["auto", "light", "dark"];' in theme
    # anything else in storage (an old or edited value) means Auto
    assert 'return CHOICES.includes(value) ? value : "auto";' in theme


def test_only_light_and_dark_set_the_attribute(theme):
    """Auto leaves ``data-theme`` off, so the CSS follows the system (and its changes)."""
    apply = re.search(r"function apply\(\) \{.*?\n  \}", theme, re.S).group(0)
    assert 'if (choice === "auto") {\n      root.removeAttribute("data-theme");' in apply
    assert 'root.setAttribute("data-theme", choice);' in apply


def test_storage_refused_is_not_an_error(theme):
    """A private window or blocked site data: the button still works on the page."""
    assert theme.count("localStorage") == 2
    assert re.search(r"try \{\s*const value = localStorage\.getItem\(STORAGE_KEY\);", theme)
    assert re.search(r"try \{\s*localStorage\.setItem\(STORAGE_KEY, choice\);\s*\} catch", theme)


def test_the_button_says_its_state_and_appears_only_with_javascript(theme):
    assert "button.textContent = `Theme: ${LABELS[choice]}`;" in theme
    # the accessible name starts with the visible text (WCAG 2.5.3)
    assert re.search(r'button\.setAttribute\("aria-label", `Theme: \$\{LABELS\[choice\]\}', theme)
    assert "button.hidden = false;" in theme


def test_the_click_cycles_auto_light_dark(theme):
    assert "choice = CHOICES[(CHOICES.indexOf(choice) + 1) % CHOICES.length];" in theme


def test_no_globals(theme):
    """A classic script shares one global scope with every other script: keep it all inside."""
    code = theme.strip()
    assert code.startswith("(() => {") and code.endswith("})();")


def test_theme_js_is_served(make_app):
    response = make_app().test_client().get("/static/js/theme.js")
    assert response.status_code == 200
    assert response.mimetype == "text/javascript"
    assert (JS_DIR / "theme.js").is_file()
