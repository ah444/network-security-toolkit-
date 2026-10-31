"""v5.3 F2d: the charts of a run (charts.js), drawn as SVG by the page's own code: every number
also as text (beside its mark and in the chart's name for screen readers), the colours from
the stylesheet's tokens, text never parsed. Rendered by the page's module in Node
(``charts.mjs``), skipped where Node is not installed; the drawing itself is checked in the
browser (F2g)."""

import json
import re
import shutil
import subprocess

import pytest

from web_js import JS_DIR, REPO_ROOT, js_sources

HERE = REPO_ROOT / "tests" / "web"
APP_CSS = JS_DIR.parent / "css" / "app.css"
XSS = "<img src=x onerror=alert(1)>"
CAPTIONS = ["CV macro-F1, fold by fold", "CV against test macro-F1", "Test recall per class",
            "Fit time (log scale)"]


@pytest.fixture(scope="module")
def rendered():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run([node, str(HERE / "charts.mjs"), str(JS_DIR)],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _figure(rendered, caption):
    return rendered["figures"][CAPTIONS.index(caption)]


def _marks(figure, cls):
    return [mark for mark in figure["marks"] if mark["class"] == cls]


def test_the_section_and_its_four_charts(rendered):
    assert rendered["section"]["tag"] == "DETAILS"
    assert rendered["section"]["class"] == "run-charts"
    assert "open" in rendered["section"]["attributes"]
    assert rendered["summary"] == "Charts"
    assert [figure["caption"] for figure in rendered["figures"]] == CAPTIONS


def test_each_chart_is_a_named_box_the_keyboard_reaches(rendered):
    for figure in rendered["figures"]:
        box = figure["box"]["attributes"]
        assert (box["role"], box["tabindex"], box["aria-label"]) == \
            ("region", "0", figure["caption"])


def test_the_drawings_are_named_with_every_number(rendered):
    for caption in (CAPTIONS[0], CAPTIONS[1], CAPTIONS[3]):
        figure = _figure(rendered, caption)
        drawing = figure["drawing"]["attributes"]
        assert drawing["role"] == "img" and drawing["width"] == "600"
        assert drawing["viewBox"] == f"0 0 600 {drawing['height']}"
        values = [mark["text"] for mark in _marks(figure, "chart-value")]
        names = [mark["text"] for mark in _marks(figure, "chart-label")]
        assert names == ["Random Forest", "Decision Tree", XSS]
        assert len(values) == 3
        for name, value in zip(names, values, strict=True):
            assert f"{name}: " in drawing["aria-label"]
            for number in re.findall(r"[\d,]+\.?\d*", value):
                assert number in drawing["aria-label"], (caption, number)


def test_the_folds_chart(rendered):
    figure = _figure(rendered, CAPTIONS[0])
    assert len(_marks(figure, "chart-fold")) == 3 * 3 + 1  # every fold, and the legend's
    assert len(_marks(figure, "chart-mean")) == 3 + 1
    assert [m["text"] for m in _marks(figure, "chart-value")] == \
        ["0.912 ± 0.004", "0.881 ± 0.012", "0.550 ± 0.030"]
    assert "folds 0.908, 0.913, 0.915; mean 0.912 ± 0.004" in \
        figure["drawing"]["attributes"]["aria-label"]
    assert [m["text"] for m in _marks(figure, "chart-tick")][-3:] == ["0.50", "0.75", "1.00"]
    assert "folds" not in rendered["without_folds"].split(". ", 1)[1], "no empty fold list"


def test_cv_and_test_have_different_shapes(rendered):
    """Not the colour only: CV is a circle, test a square, and the legend says so."""
    figure = _figure(rendered, CAPTIONS[1])
    assert {m["tag"] for m in _marks(figure, "chart-cv")} == {"CIRCLE"}
    assert {m["tag"] for m in _marks(figure, "chart-test")} == {"RECT"}
    assert len(_marks(figure, "chart-cv")) == len(_marks(figure, "chart-test")) == 3 + 1
    ticks = [m["text"] for m in _marks(figure, "chart-tick")]
    assert ticks[:2] == ["CV (training split)", "test (scored once)"]
    assert ticks[2:] == ["0.40", "0.70", "1.00"], "the axis starts at the tenth below the lowest"


def test_the_time_chart_is_a_log_scale_with_few_ticks(rendered):
    figure = _figure(rendered, CAPTIONS[3])
    ticks = [m["text"] for m in _marks(figure, "chart-tick")]
    assert ticks == ["0.001 s", "0.1 s", "10 s", "1,000 s", "100,000 s"]
    assert [m["text"] for m in _marks(figure, "chart-value")] == ["4,414 s", "13 s", "0.0042 s"]
    assert rendered["same_speed"] == ["10 s", "100 s"], "one model's time still has an axis"
    assert rendered["seconds"] == ["0.0042 s", "0.5 s", "1 s", "10 s", "13 s", "4,414 s"]


def test_every_mark_is_inside_the_drawing(rendered):
    for figure in rendered["figures"]:
        if not figure["drawing"]:
            continue
        height = float(figure["drawing"]["attributes"]["height"])
        for mark in figure["marks"]:
            a = mark["attributes"]
            for key in ("x", "x1", "x2", "cx"):
                if key in a:
                    assert 0 <= float(a[key]) <= 600, (figure["caption"], mark)
            for key in ("y", "y1", "y2", "cy"):
                if key in a:
                    assert 0 <= float(a[key]) <= height, (figure["caption"], mark)


def test_the_recall_table(rendered):
    figure = _figure(rendered, CAPTIONS[2])
    assert figure["drawing"] is None and figure["note"].startswith("Shaded: under 0.99")
    cells = [(c["tag"], c["class"], c["text"]) for c in figure["cells"]]
    assert cells[:4] == [("TH", "", "Class ↓ / Model →"), ("TH", "", "Random Forest"),
                         ("TH", "", "Decision Tree"), ("TH", "", XSS)]
    assert cells[4:8] == [("TH", "", "BENIGN"), ("TD", "", "0.994"), ("TD", "", "0.991"),
                          ("TD", "cm-wrong-4", "0.700")]
    assert cells[-4:] == [("TH", "", XSS), ("TD", "cm-wrong-5", "0.300"),
                          ("TD", "cm-wrong-4", "0.620"), ("TD", "cm-wrong-5", "0.000")]
    assert rendered["shades"] == [None, None, "cm-wrong-1", "cm-wrong-1", "cm-wrong-2",
                                  "cm-wrong-2", "cm-wrong-3", "cm-wrong-3", "cm-wrong-4",
                                  "cm-wrong-4", "cm-wrong-5", "cm-wrong-5"]


def test_text_from_the_run_stays_text(rendered):
    text = json.dumps(rendered)
    assert '"tag": "IMG"' not in text
    for figure in rendered["figures"]:
        for mark in figure["marks"] + figure["cells"]:
            assert not any(re.match(r"on|style$", key) for key in mark["attributes"])


def test_the_charts_have_no_inline_style_and_set_text_as_text():
    code = js_sources()["charts.js"]
    assert "innerHTML" not in code and ".style" not in code and "style:" not in code
    assert "node.textContent = text" in code


def test_the_charts_styles():
    css = APP_CSS.read_text(encoding="utf-8")
    rule = css[css.index(".chart-drawing text {"):]
    assert "font-size: 13px;" in rule[:rule.index("}")], "the text never shrinks with the box"
    for cls in ("chart-grid", "chart-spread", "chart-mean", "chart-fold", "chart-cv",
                "chart-test", "chart-note", "run-charts"):
        assert f".{cls}" in css, cls


def test_the_trainer_shows_the_charts_under_the_leaderboard():
    code = js_sources()["train.js"]
    assert 'import { leaderboardTable, runCharts } from "./charts.js";' in code
    assert code.index("leaderboardTable(result.leaderboard)") < \
        code.index("runCharts(result.leaderboard, result.folds)")
