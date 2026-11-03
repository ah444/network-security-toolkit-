"""v5.3 step F3e: what a model relies on, on the pages.

- The trainer (Train step): a model chosen, **Calculate** (nothing starts on its own), the
  job's progress and Cancel, then the chart of the 15 features with the biggest drop, the
  concentration warning (HC2) and what the numbers are.
- The saved runs' page: a run's saved results, read only; none calculated says where to.
Rendered by the page's modules in Node (``importance.mjs``), skipped where Node is not
installed; the page itself is checked in the browser (F3g)."""

import json
import shutil
import subprocess

import pytest

from web_js import JS_DIR, REPO_ROOT, TEMPLATE, js_sources

HERE = REPO_ROOT / "tests" / "web"
APP_CSS = JS_DIR.parent / "css" / "app.css"
XSS = "<img src=x onerror=alert(1)>"
LONG = "Total Length of Bwd Packets Per Second Averaged"


@pytest.fixture(scope="module")
def rendered():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run([node, str(HERE / "importance.mjs"), str(JS_DIR)],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _marks(view, cls):
    return [mark for mark in view["marks"] if mark["class"] == cls]


def test_the_15_features_with_the_biggest_drop(rendered):
    plain = rendered["plain"]
    assert plain["caption"] == "What Random Forest relies on"
    labels = [mark["text"] for mark in _marks(plain, "chart-label")]
    assert labels[0] == XSS and labels[2:] == [f"feature {i}" for i in range(2, 15)]
    assert labels[1] == "Total Length of Bwd Packets…", "a long name is cut, never overlaps"
    assert [m["text"] for m in _marks(plain, "chart-value")][:2] == \
        ["0.2000 ± 0.0040", "0.1840 ± 0.0040"]
    assert len(_marks(plain, "chart-drop")) == 15
    assert len(_marks(rendered["few"], "chart-label")) == 3


def test_every_number_is_in_the_drawings_name(rendered):
    name = rendered["plain"]["drawing"]["attributes"]["aria-label"]
    assert name.startswith("The drop in macro-F1 when each feature's values are shuffled")
    assert f"{LONG}: 0.1840 ± 0.0040" in name and f"{XSS}: 0.2000 ± 0.0040" in name
    assert "feature 15" not in name


def test_a_drop_below_zero_goes_left_of_zero(rendered):
    """Shuffling can help a little (a negative drop): its bar ends at 0 instead of starting."""
    zero = float(_marks(rendered["plain"], "chart-zero")[0]["attributes"]["x1"])
    bars = [m["attributes"] for m in _marks(rendered["plain"], "chart-drop")]
    assert all(float(b["width"]) >= 0 for b in bars)
    assert float(bars[0]["x"]) == zero, "0.2: from 0 to the right"
    assert float(bars[14]["x"]) + float(bars[14]["width"]) == pytest.approx(zero),         "-0.024: from the left up to 0"
    assert [m["text"] for m in _marks(rendered["plain"], "chart-value")][14] == "-0.0240 ± 0.0040"


def test_every_mark_is_inside_the_drawing(rendered):
    for key in ("plain", "few"):
        view = rendered[key]
        height = float(view["drawing"]["attributes"]["height"])
        for mark in view["marks"]:
            a = mark["attributes"]
            for name in ("x", "x1", "x2"):
                if name in a:
                    assert 0 <= float(a[name]) <= 600, mark
            for name in ("y", "y1", "y2"):
                if name in a:
                    assert 0 <= float(a[name]) <= height, mark
            if "width" in a and mark["class"] == "chart-drop":
                assert float(a["x"]) + float(a["width"]) <= 600


def test_what_the_numbers_are(rendered):
    texts = [t["text"] for t in rendered["plain"]["texts"]]
    assert texts[0] == ("Scored on 5,000 of 22,544 test rows, 5 repeats, seed 42; the 15 of "
                        "20 features with the biggest drop are shown.")
    assert texts[1] == ("A diagnosis of this run on its test rows, not a way to choose features "
                        "or models; it shows what the model relies on, not causes.")
    assert len(texts) == 2, "no warning: no feature carries most of the effect, and it was saved"


def test_the_concentration_warning(rendered):
    warning = [t for t in rendered["concentrated"]["texts"] if t["class"] == "warning-message"]
    assert [w["text"] for w in warning] == [
        f"⚠️ {XSS} carries more than half of the effect. Check that it is not a copy of the "
        "label (one possible cause) before trusting the scores."]


def test_a_result_not_saved_says_why(rendered):
    warning = [t for t in rendered["not_saved"]["texts"] if t["class"] == "warning-message"]
    assert [w["text"] for w in warning] == ["Not saved: only 10 MB free"]


def test_text_from_the_run_stays_text(rendered):
    for key in ("plain", "concentrated"):
        assert all(item["tag"] != "IMG" for item in rendered[key]["all"])
        for mark in rendered[key]["all"]:
            assert not any(k.startswith("on") or k == "style" for k in mark["attributes"])


def test_a_saved_run_shows_its_results(rendered):
    assert "What the models rely on" in rendered["run_with"]
    assert "What Random Forest relies on" in rendered["run_with"]
    assert "feature 3 carries more than half of the effect" in rendered["run_with"]
    assert "Not saved" not in rendered["run_with"], "a saved file has no saved flag"


def test_a_saved_run_without_results_says_where_to_calculate(rendered):
    text = rendered["run_without"]
    assert ("Not calculated for this run. Calculate it on the Dynamic IDS page right after a "
            "training: a saved run's test rows are not kept.") in text
    assert "1 file under this run's name could not be read and was left out." in text


def test_the_trainer_has_calculate_and_nothing_starts_on_its_own():
    html = TEMPLATE.read_text(encoding="utf-8")
    section = html[html.index('id="importanceSection"'):html.index("<!-- Step 6")]
    assert "Nothing starts until you click Calculate." in section
    assert '<button type="button" id="importanceBtn" class="btn btn-sm">Calculate</button>' in section
    assert 'id="cancelImportanceBtn"' in section and 'aria-live="polite"' in section
    code = js_sources()["explain.js"]
    assert 'send("/api/v1/jobs/importance", "POST", { model: ' in code
    assert 'send(jobUrl, "DELETE")' in code
    assert code.count('send("/api/v1/jobs/importance"') == 1, "only from Calculate's click"
    assert 'addEventListener("click", calculate)' in code


def test_the_styles():
    css = APP_CSS.read_text(encoding="utf-8")
    for cls in ("chart-drop", "chart-zero"):
        assert f".{cls} {{" in css, cls
