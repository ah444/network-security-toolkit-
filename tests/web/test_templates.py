"""Step 0.7 (U4, D2 accessibility basics): viewport, lang and a label for every input."""

import re
from html.parser import HTMLParser

import pytest

from web_js import js_sources

TEMPLATES = "toolkit/web/ui/templates"


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


@pytest.mark.parametrize("name", ["home.html", "dynamic_ids.html", "scan.html", "runs.html",
                                  "scans.html"])
def test_viewport_lang_and_labels(repo_root, name):
    page = _scan(repo_root, name)
    assert page.html_lang == "en", f"{name}: <html lang> missing"
    assert page.viewport and "width=device-width" in page.viewport, f"{name}: viewport missing"
    unlabelled = [c for c in page.controls if c not in page.label_for and c not in page.wrapped]
    assert unlabelled == [], f"{name}: inputs without a label: {unlabelled}"


def test_home_page_does_not_load_trainer_script(repo_root):
    html = (repo_root / TEMPLATES / "home.html").read_text(encoding="utf-8")
    assert "main.js" not in html


def test_dynamic_form_labels_are_linked():
    """Labels created by JS for the prediction forms must point at their input (one builder
    for both forms since step 1.3a, with a different id prefix for each)."""
    predict = js_sources()["predict.js"]
    builder = re.search(r"function buildPredictionForm\(.*?\n\}", predict, flags=re.S).group(0)
    assert 'el("label", { htmlFor: input.id }' in builder
    assert "input.id = idPrefix + index" in builder
    prefixes = re.findall(r'buildPredictionForm\("\w+", "([\w-]+)"', predict)
    assert sorted(prefixes) == ["loaded-feature-", "session-feature-"]


def test_home_page_describes_signed_bundles(repo_root):
    """Step 1.3d: nothing on the home page says PKL files load or a model retrains. (Its cards
    describing the signed bundles gave way to the dashboard in v5.4 F10b; the trainer's page
    says it.)"""
    html = (repo_root / TEMPLATES / "home.html").read_text(encoding="utf-8")
    assert "PKL" not in html.upper() and "continue training" not in html


def test_status_messages_are_announced(repo_root):
    """Step 1.3e (plan §8, accessibility): every message area is a polite live region."""
    html = (repo_root / TEMPLATES / "dynamic_ids.html").read_text(encoding="utf-8")
    areas = re.findall(r'<div id="(\w+)" class="message"([^>]*)>', html)
    assert len(areas) == 14  # + the two "Predict a CSV file" results (step 1.4f), the importance
    # (v5.3 F3e)
    assert [name for name, attrs in areas if 'role="status"' not in attrs] == []
    assert re.search(r'id="trainingProgress"[^>]*aria-live="polite"', html)
