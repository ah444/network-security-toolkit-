"""Step 0.7 (U4, D2 accessibility basics): viewport, lang and a label for every input."""

import re
from html.parser import HTMLParser

import pytest

TEMPLATES = "tools/IDS/IDS_Web_CIC_Dynamic/templates"
MAIN_JS = "tools/IDS/IDS_Web_CIC_Dynamic/static/js/main.js"


class _FormScanner(HTMLParser):
    """Collect controls, <label for=...> targets and controls wrapped in a <label>."""

    def __init__(self):
        super().__init__()
        self.controls, self.label_for, self.wrapped = [], set(), set()
        self._label_depth = 0
        self.html_lang = None
        self.viewport = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "html":
            self.html_lang = attrs.get("lang")
        if tag == "meta" and attrs.get("name") == "viewport":
            self.viewport = attrs.get("content", "")
        if tag == "label":
            self._label_depth += 1
            if attrs.get("for"):
                self.label_for.add(attrs["for"])
        if tag in ("input", "select", "textarea") and attrs.get("type") != "hidden":
            self.controls.append(attrs.get("id"))
            if self._label_depth:
                self.wrapped.add(attrs.get("id"))

    def handle_endtag(self, tag):
        if tag == "label" and self._label_depth:
            self._label_depth -= 1


def _scan(repo_root, name):
    scanner = _FormScanner()
    scanner.feed((repo_root / TEMPLATES / name).read_text(encoding="utf-8"))
    return scanner


@pytest.mark.parametrize("name", ["home.html", "dynamic_ids.html"])
def test_viewport_lang_and_labels(repo_root, name):
    page = _scan(repo_root, name)
    assert page.html_lang == "en", f"{name}: <html lang> missing"
    assert page.viewport and "width=device-width" in page.viewport, f"{name}: viewport missing"
    unlabelled = [c for c in page.controls if c not in page.label_for and c not in page.wrapped]
    assert unlabelled == [], f"{name}: inputs without a label: {unlabelled}"


def test_home_page_does_not_load_trainer_script(repo_root):
    html = (repo_root / TEMPLATES / "home.html").read_text(encoding="utf-8")
    assert "main.js" not in html


def test_dynamic_form_labels_are_linked(repo_root):
    """Labels created by JS for the prediction form must point at their input."""
    js = (repo_root / MAIN_JS).read_text(encoding="utf-8")
    builders = re.findall(r"function build\w*PredictionForm\(featuresInfo\) \{.*?\n    \}", js,
                          flags=re.S)
    assert len(builders) == 2
    for body in builders:
        assert "label.htmlFor = input.id" in body
