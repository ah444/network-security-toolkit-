"""v5.1 step 1.3a (D1): the page's script is a set of native ES modules (no build step)."""

import mimetypes
import re

from web_js import ENTRY_MODULES, JS_DIR, SCAN_TEMPLATE, TEMPLATE, all_js, js_sources

IMPORT = re.compile(r"""^import\s+[^;]*?\s+from\s+["']([^"']+)["'];""", re.M)


def _imports(name):
    return IMPORT.findall(js_sources()[name])


def test_template_loads_main_as_a_module():
    html = TEMPLATE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script[^>]*>", html)
    # two classic scripts: theme.js in the <head> (v5.2 step 2.1, tests/web/test_theme.py) and
    # navbar.js right after the navbar (step 2.2a, tests/web/test_responsive.py)
    assert scripts == ['<script src="{{ url_for(\'static\', filename=\'js/theme.js\') }}">',
                       '<script src="{{ url_for(\'static\', filename=\'js/navbar.js\') }}">',
                       '<script type="module" src="{{ url_for(\'static\', filename=\'js/main.js\') }}">']
    scan_scripts = re.findall(r"<script[^>]*>", SCAN_TEMPLATE.read_text(encoding="utf-8"))
    assert scan_scripts == scripts[:2] + [
        '<script type="module" src="{{ url_for(\'static\', filename=\'js/scan.js\') }}">']


def test_imports_are_relative_and_exist():
    for name in js_sources():
        for target in _imports(name):
            assert re.fullmatch(r"\./[a-z]+\.js", target), f"{name}: {target}"
            assert (JS_DIR / target[2:]).is_file(), f"{name}: {target}"


def _reached(entry):
    """The modules a page's entry module loads, itself included."""
    reached, todo = set(), [entry]
    while todo:
        name = todo.pop()
        if name not in reached:
            reached.add(name)
            todo += [target[2:] for target in _imports(name)]
    return reached


def test_every_module_is_used():
    reached = set().union(*(_reached(entry) for entry in ENTRY_MODULES))
    assert reached == set(js_sources()) - {"theme.js", "navbar.js"}  # classic scripts, no imports
    shared = _reached("main.js") & _reached("scan.js")
    assert shared == {"api.js", "dom.js", "summary.js"}, "the pages share only the helpers"


def test_only_api_js_talks_to_the_server():
    for name, code in js_sources().items():
        if name != "api.js":
            assert "fetch(" not in code and "XMLHttpRequest" not in code, name


def test_no_var_and_no_globals():
    for name, code in js_sources().items():
        assert not re.search(r"^\s*var\s", code, re.M), name
        assert "window." not in code.replace("window.confirm", "").replace("window.location", ""), name


def test_every_element_id_exists_in_the_template():
    """A misspelt id would only fail in the browser, when that button is used: each page's
    modules use only the ids of their own page."""
    sources = js_sources()
    for entry, template in ENTRY_MODULES.items():
        html = template.read_text(encoding="utf-8")
        template_ids = set(re.findall(r'\sid="([^"]+)"', html))
        code = "\n".join(sources[name] for name in _reached(entry))
        used = set(re.findall(r"""byId\(\s*["']([^"']+)["']\s*\)""", code))
        used |= set(re.findall(r'data-list="([^"]+)"', html))
        assert used and used <= template_ids, (entry, sorted(used - template_ids))


def test_every_event_has_a_sender_and_a_listener():
    code = all_js()
    sent = set(re.findall(r"""emit\(\s*["']([a-z:]+)["']""", code))
    heard = set(re.findall(r"""\bon\(\s*["']([a-z:]+)["']""", code))
    assert sent == heard == {"dataset:loaded", "dataset:changed", "run:ready", "bundle:loaded",
                             "target:chosen", "path:changed", "workspace:resumed", "steps:restore"}


def test_js_is_served_as_javascript_even_if_the_os_says_otherwise(make_app):
    """Some Windows installs map .js to text/plain in the registry; a module script served
    like that (with nosniff) would not run, and the page would do nothing."""
    mimetypes.add_type("text/plain", ".js")
    try:
        response = make_app().test_client().get("/static/js/main.js")
        assert response.status_code == 200
        assert response.mimetype == "text/javascript"
    finally:
        mimetypes.add_type("text/javascript", ".js")
