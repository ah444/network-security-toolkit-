"""v5.2 step 2.2a: the pages on phones and tablets. At 768 px and less the navbar is one
row (the logo and a Menu button that opens the links and the Theme button), the padding is
smaller under 600 px, and the step bar is one row that scrolls sideways. ``navbar.js`` is a
classic script right after the navbar; without JavaScript the links show as before.

v5.2 step 2.2b: large screens and long content. The trainer is 1200 px wide with text lines
and fields kept to a readable size, checkbox lists and the one-row form are columns, a long
one-row form starts closed, the confusion matrix scrolls inside its own box with the class
names in view, every scroll box can be reached with the keyboard, and the home page's four
cards are 1, 2 or 4 columns."""

import re

import pytest

from web_js import JS_DIR, REPO_ROOT, TEMPLATE, TEMPLATES, js_sources

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


# ---------- v5.2 step 2.2b: large screens and long content ----------

def _form(name):
    html = TEMPLATE.read_text(encoding="utf-8")
    return re.search(r'<form id="' + name + r'".*?</form>', html, re.S).group(0)


def test_the_trainer_is_as_wide_as_the_home_page(css):
    """It stopped at 670 px of content from 1024 px up (35 % of a 1920 px screen)."""
    assert _top_level(css, ".container")["max-width"] == "1200px"
    assert "max-width" not in _top_level(css, ".container-wide")


def test_text_lines_and_fields_keep_a_readable_size(css):
    for text in (".section-description", ".balancing-hint"):
        assert _top_level(css, text)["max-width"] == "75ch", text
    assert _top_level(css, ".form-group")["max-width"] == "40rem"
    assert _top_level(css, ".training-progress")["max-width"] == "40rem"
    # a full-width button ends where the fields do (its margins are 0.5rem)
    assert _top_level(css, ".btn")["max-width"] == "39rem"


def test_checkbox_lists_are_columns_where_there_is_room(css):
    lists = _top_level(css, ".checkbox-list")
    assert lists["display"] == "grid"
    assert lists["grid-template-columns"] == "repeat(auto-fill, minmax(14rem, 1fr))"
    assert lists["max-height"] == "240px"


@pytest.mark.parametrize("form, select, inputs", [
    ("predictionForm", "algorithmSelect", "dynamicInputs"),
    ("predictLoadedForm", "loadedAlgorithmSelect", "loadedDynamicInputs"),
])
def test_the_one_row_form_is_in_a_details_element(form, select, inputs):
    """The model choice stays outside: the CSV file prediction uses it too."""
    html = _form(form)
    details = re.search(r'<details class="row-entry">.*?</details>', html, re.S).group(0)
    assert re.match(r'<details class="row-entry">\s*<summary>Enter one row by hand</summary>', details)
    assert f'<div id="{inputs}" class="field-grid"></div>' in details
    assert '<button type="submit" class="btn">Predict</button>' in details
    assert html.index(f'id="{select}"') < html.index("<details")


def test_a_long_one_row_form_starts_closed():
    predict = js_sources()["predict.js"]
    assert "const OPEN_UP_TO = 12;" in predict
    build = re.search(r"function buildPredictionForm\(.*?\n\}", predict, re.S).group(0)
    assert 'const details = container.closest("details");' in build
    assert "details.open = featuresInfo.length <= OPEN_UP_TO;" in build
    assert "`Enter one row by hand (${featuresInfo.length} features)`" in build


def test_the_one_row_form_is_columns_and_its_summary_easy_to_tap(css):
    grid = _top_level(css, ".field-grid")
    assert grid["display"] == "grid"
    assert grid["grid-template-columns"] == "repeat(auto-fill, minmax(14rem, 1fr))"
    assert _top_level(css, ".row-entry > summary")["min-height"] == "44px"


def test_a_long_file_name_wraps_on_saved_runs(css):
    """Gate 5.3: "<run id>.importance.decision_tree.json" in the Delete panel ran 34 px past a
    320 px screen; the Storage leftovers are names like it."""
    for box in ("#deleteView li", "#storageView li"):
        assert _top_level(css, box)["overflow-wrap"] == "anywhere", box


def test_why_is_easy_to_tap(css):
    """F3g: "Why?" under a prediction was 26 px high, measured at the 8 sizes."""
    assert _top_level(css, ".why > summary")["min-height"] == "44px"


def test_every_scroll_box_can_be_reached_with_the_keyboard(css):
    """A box that scrolls needs focus to scroll with the arrow keys, and a name to be told apart."""
    dom = js_sources()["dom.js"]
    helper = re.search(r"export function scrollBox\(label, \.\.\.children\) \{.*?\n\}", dom, re.S).group(0)
    assert ('el("div", { class: "table-container", tabindex: "0", role: "region", '
            '"aria-label": label }, children)') in helper
    for name, code in js_sources().items():
        if name != "dom.js":
            assert '"table-container"' not in code, name
    html = TEMPLATE.read_text(encoding="utf-8")
    assert ('<div id="dataPreviewTable" class="table-container" tabindex="0" role="region" '
            'aria-label="Dataset preview"></div>') in html
    assert "class=\"table-container\"" not in html.replace('id="dataPreviewTable" class="table-container"', "")
    focus = re.search(r"([^{}]*)\{\s*outline: 3px solid var\(--focus\);\s*outline-offset: 2px;", css).group(1)
    assert '[tabindex="0"]:focus-visible' in focus and "summary:focus-visible" in focus


def test_the_matrix_scrolls_inside_its_box_with_the_names_in_view(css):
    train = js_sources()["train.js"]
    assert 'scrollBox("Confusion matrix of " + modelTitle(modelName),' in train
    assert re.search(r'el\("thead", \{\}, el\("tr", \{\}, el\("th", \{ scope: "col" \}, '
                     r'"Actual ↓", el\("br"\), "Predicted →"\)', train)
    assert 'el("th", { scope: "row", title: names[i] }, shortLabel(names[i]))' in train
    assert "overflow-x" not in _top_level(css, ".confusion-matrix-container")
    assert _top_level(css, ".confusion-matrix-container .table-container")["max-height"] == "70vh"
    header = _top_level(css, ".confusion-matrix thead th")
    names = _top_level(css, ".confusion-matrix tbody th")
    corner = _top_level(css, ".confusion-matrix thead th:first-child")
    assert (header["position"], header["top"]) == ("sticky", "0")
    assert (names["position"], names["left"]) == ("sticky", "0")
    assert corner["left"] == "0" and int(corner["z-index"]) > int(header["z-index"])


def test_a_big_matrix_has_smaller_cells(css):
    train = js_sources()["train.js"]
    assert "const COMPACT_ABOVE = 15;" in train
    assert 'matrix.length > COMPACT_ABOVE ? " cm-compact" : ""' in train
    compact = _top_level(css, ".cm-compact td")
    assert compact == _top_level(css, ".cm-compact th")
    assert compact["min-width"] == "2.5rem" and compact["padding"] == "4px 6px"
    assert compact["font-size"] == "0.8rem"  # 12.8 px: no text under 12 px


def test_the_dashboard_is_one_column_then_two_side_by_side(css):
    """F10: the scans and the models one under the other, side by side where there is room."""
    assert _top_level(css, ".dashboard-grid")["grid-template-columns"] == "1fr"
    assert _media(css, "min-width: 1000px")[".dashboard-grid"]["grid-template-columns"] == \
        "repeat(2, minmax(0, 1fr))"
    assert ".features-grid" not in css and ".feature-card" not in css, "the old cards are gone"
