"""v5.2 step 2.2a: the pages on phones and tablets. At 768 px and less the navbar is one
row (the logo and a Menu button that opens the links and the Theme button), the padding is
smaller under 600 px, and the step bar is one row that scrolls sideways. ``navbar.js`` is a
classic script right after the navbar; without JavaScript the links show as before."""

import re

import pytest

from web_js import JS_DIR, REPO_ROOT, TEMPLATE, js_sources

TEMPLATES = sorted(TEMPLATE.parent.glob("*.html"))
APP_CSS = JS_DIR.parent / "css" / "app.css"
NAVBAR_SCRIPT = "<script src=\"{{ url_for('static', filename='js/navbar.js') }}\"></script>"
MENU_BUTTON = ('<button type="button" id="navToggle" class="nav-link nav-toggle" '
               'aria-expanded="false" aria-controls="navMenu" hidden>Menu</button>')


@pytest.fixture(scope="module")
def navbar():
    return js_sources()["navbar.js"]


@pytest.fixture(scope="module")
def css():
    return re.sub(r"/\*.*?\*/", "", APP_CSS.read_text(encoding="utf-8"), flags=re.S)


def _rules(text):
    """Plain rules (no nested blocks) as {selector: {property: value}}, later ones winning."""
    rules = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", text):
        parts = [part.strip() for part in body.split(";") if part.strip()]
        declarations = dict(re.split(r"\s*:\s*", part, maxsplit=1) for part in parts)
        for name in selector.split(","):
            rules.setdefault(name.strip(), {}).update(declarations)
    return rules


def _media_blocks(css, query):
    """(start, end) of the body of every ``@media (<query>)`` block."""
    for match in re.finditer(r"@media \(" + re.escape(query) + r"\)\s*\{", css):
        depth, start = 1, match.end()
        for end in range(start, len(css)):
            depth += {"{": 1, "}": -1}.get(css[end], 0)
            if depth == 0:
                yield start, end
                break


def _media(css, query):
    return _rules("".join(css[start:end] + "\n" for start, end in _media_blocks(css, query)))


def _outside_media(css):
    return re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)


def _top_level(css, selector):
    return _rules(_outside_media(css))[selector]


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_both_pages_have_the_menu_button_and_its_menu(template):
    html = template.read_text(encoding="utf-8")
    nav = re.search(r'<nav class="navbar">.*?</nav>', html, re.S).group(0)
    assert nav.count(MENU_BUTTON) == 1
    assert nav.index("nav-logo") < nav.index(MENU_BUTTON) < nav.index('<ul class="nav-menu" id="navMenu">')


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_navbar_js_runs_right_after_the_navbar(template):
    """A classic script after </nav>: the menu is closed before the page is drawn."""
    html = template.read_text(encoding="utf-8")
    assert html.count(NAVBAR_SCRIPT) == 1
    assert re.search(r"</nav>\s*" + re.escape(NAVBAR_SCRIPT), html)


def test_the_menu_button_appears_only_with_javascript(navbar):
    assert "button.hidden = false;" in navbar
    assert 'setOpen(false);\n  button.hidden = false;' in navbar


def test_the_button_says_whether_the_menu_is_open(navbar):
    assert 'nav.setAttribute("data-menu", open ? "open" : "closed");' in navbar
    assert 'button.setAttribute("aria-expanded", String(open));' in navbar


def test_escape_closes_and_returns_focus_to_the_button(navbar):
    escape = re.search(r'if \(event\.key === "Escape".*?\n    \}', navbar, re.S).group(0)
    assert "setOpen(false);" in escape and "button.focus();" in escape


def test_choosing_a_link_closes_the_menu(navbar):
    assert re.search(r'querySelectorAll\("\.nav-menu a"\)\.forEach\(link => \{\s*'
                     r'link\.addEventListener\("click", \(\) => setOpen\(false\)\);', navbar)


def test_no_globals(navbar):
    code = navbar.strip()
    assert code.startswith("(() => {") and code.endswith("})();")


def test_navbar_js_is_served(make_app):
    response = make_app().test_client().get("/static/js/navbar.js")
    assert response.status_code == 200
    assert response.mimetype == "text/javascript"


def test_the_menu_button_is_drawn_on_small_screens_only(css):
    assert _top_level(css, ".nav-toggle")["display"] == "none"
    assert _media(css, "max-width: 768px")[".nav-toggle"]["display"] == "inline-flex"


def test_small_screens_one_navbar_row_and_a_closed_menu(css):
    small = _media(css, "max-width: 768px")
    assert small[".navbar"].get("flex-direction", "row") == "row"
    assert small[".navbar"]["flex-wrap"] == "wrap"
    assert small[".nav-logo"]["font-size"] == "1.2rem"
    assert small[".nav-logo"]["flex"] == "1"  # beside the button, not on a row of its own
    assert small['.navbar[data-menu="closed"] .nav-menu'] == {"display": "none"}
    # an open menu (or no JavaScript) takes the row under the logo
    assert small[".nav-menu"]["flex-basis"] == "100%"


def test_closed_menu_is_hidden_only_on_small_screens(css):
    """On wider screens the links always show, whatever the menu's state."""
    assert "data-menu" not in _outside_media(css)
    assert "data-menu" not in _media(css, "max-width: 600px")


def test_less_padding_on_phones(css):
    """About 288 of 320 px for the content (it was 206)."""
    phone = _media(css, "max-width: 600px")
    assert phone[".container"]["padding"] == "0 0.5rem"
    assert phone[".container"]["margin"] == "1rem 0"
    assert phone[".section"]["padding"] == "1rem 0.5rem"


def test_full_width_buttons_fit_inside_their_margins(css):
    """width: 100% plus a 0.5rem margin on each side ran 1rem past the card's content."""
    assert _top_level(css, ".btn")["margin"] == "0.5rem"
    assert _top_level(css, ".btn")["width"] == "calc(100% - 1rem)"
    assert _top_level(css, ".btn-sm")["width"] == "auto"


def test_step_bar_is_one_row_that_scrolls_sideways(css):
    small = _media(css, "max-width: 768px")
    assert small[".stepper-list"]["flex-wrap"] == "nowrap"
    assert small[".stepper-list"]["overflow-x"] == "auto"
    # room for the focus ring (3 px and 2 px away) inside the scrolling box
    assert small[".stepper-list"]["padding"] == "5px"
    assert small[".stepper-list > li"] == {"flex": "none"}
    assert _top_level(css, ".step-link")["min-height"] == "44px"


def test_the_current_step_is_kept_in_view():
    """Only the step bar scrolls (scrollLeft), never the page (no scrollIntoView)."""
    steps = js_sources()["steps.js"]
    assert "scrollIntoView" not in steps
    keep = re.search(r"function keepInView\(link\) \{.*?\n\}", steps, re.S).group(0)
    assert "list.scrollLeft +=" in keep
    assert 'if (step === current) {\n      link.setAttribute("aria-current", "step");\n      keepInView(link);' in steps


def test_model_choices_are_short():
    """The numbers stay in the leaderboard; on a phone the choice was cut off."""
    predict = js_sources()["predict.js"]
    assert "value: row.model, text: row.title, selected: row.selected" in predict
    assert "CV macro-F1" not in predict


def test_layout_audit_script_exists():
    audit = (REPO_ROOT / "scripts" / "layout_audit.js").read_text(encoding="utf-8")
    for measure in ("pageScrollX", "outOfScreen", "smallTargets", "tinyText", "stickyShare",
                    "sectionContent", "stepper", "screens"):
        assert measure in audit, measure
