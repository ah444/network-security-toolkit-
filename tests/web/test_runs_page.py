"""v5.3 F2e: the saved runs' page, /runs. It reads the runs' records through /api/v1/runs only
(nothing uploaded, trained or loaded); a difference only when both runs scored the same test
rows, otherwise the numbers side by side and why. What runsview.js shows is rendered by the
page's module in Node (``runs.mjs``), skipped where Node is not installed; the page itself is
checked in the browser (F2g)."""

import json
import re
import shutil
import subprocess

import pytest

from web_js import JS_DIR, REPO_ROOT, RUNS_TEMPLATE, TEMPLATES, js_sources

HERE = REPO_ROOT / "tests" / "web"
XSS = "<img src=x onerror=alert(1)>"
REASON_CODES = ["DIFFERENT_DATASET", "DIFFERENT_TEST_FILE", "DIFFERENT_LABEL",
                "DIFFERENT_SPLIT", "DIFFERENT_ROWS"]


@pytest.fixture(scope="module")
def rendered():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run([node, str(HERE / "runs.mjs"), str(JS_DIR)],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _texts(elements, tag):
    return [item["text"] for item in elements if item["tag"] == tag]


def test_the_page_is_served(client):
    response = client.get("/runs")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert '<a href="/runs" class="nav-link" aria-current="page">Saved Runs</a>' in html
    assert "js/runs.js" in html


def test_every_page_links_to_it(client):
    for url in ("/", "/dynamic_ids", "/scan", "/runs"):
        assert 'href="/runs" class="nav-link"' in client.get(url).get_data(as_text=True), url
    for template in TEMPLATES:
        assert 'href="/runs"' in template.read_text(encoding="utf-8"), template.name


def test_the_export_links_to_the_saved_run():
    code = js_sources()["predict.js"]
    assert ('el("a", { href: "/runs#" + result.run_id, class: "btn btn-secondary btn-sm" }, '
            '"See it in Saved Runs")') in code, "a 44 px button-like link next to Download"


def test_the_page_reads_the_records_only():
    """No upload, no training, no bundle: GETs to /api/v1/runs and nothing else."""
    code = js_sources()["runs.js"] + js_sources()["runsview.js"]
    assert "send(" not in code and "FormData" not in code
    urls = set(re.findall(r'"(/api/v1/[^"]*)"', code))
    assert urls == {"/api/v1/runs", "/api/v1/runs/", "/api/v1/runs/compare"}
    assert "encodeURIComponent(runId)" in code


def test_a_run_in_the_address_is_checked_as_the_server_checks_it():
    server = re.search(r'RUN_ID = re.compile\(r"([^"]+)"\)',
                       (REPO_ROOT / "toolkit/core/services/ids/comparison.py").read_text(
                           encoding="utf-8")).group(1)
    page = re.search(r"export const RUN_ID = /\^(.+)\$/;", js_sources()["runsview.js"]).group(1)
    assert page == server


def test_a_older_than_b():
    """Δ = B − A is the newer run's number less the older's: a run id starts with its time."""
    code = js_sources()["runs.js"]
    assert "const [a, b] = chosen().sort();" in code


def test_every_reason_has_words():
    code = js_sources()["runsview.js"]
    for reason in REASON_CODES:
        assert re.search(rf"^\s+{reason}: \"", code, re.M), reason


def test_a_row_of_the_list(rendered):
    cells = rendered["row_cells"]
    assert cells[1:] == ["2026-10-30 12:05 UTC20261030T120509Z-0000000a",
                         f"{XSS}12,345 rowstest file: KDDTest+.csv", "Label",
                         "stratified8,641 / 3,704", "★ Random Forestof 2 models", "0.912",
                         "0.944", "Show"]
    box = next(item for item in rendered["row"] if item["tag"] == "INPUT")
    assert box["attributes"]["aria-label"] == "Compare run 20261030T120509Z-0000000a"
    assert next(item for item in rendered["row"] if item["tag"] == "LABEL")["class"] == \
        "choice run-pick", "a 44 px target"
    button = next(item for item in rendered["row"] if item["tag"] == "BUTTON")
    assert button["attributes"]["data-run"] == "20261030T120509Z-0000000a"
    assert button["attributes"]["aria-label"] == "Show run 20261030T120509Z-0000000a"


def test_one_run_shows_its_leaderboard_and_charts(rendered):
    view = rendered["view"]
    assert _texts(view, "H4") == ["Run 20261030T120509Z-0000000a"]
    assert ["Saved", "2026-10-30 12:05 UTC"] == _texts(view, "DT")[:1] + _texts(view, "DD")[:1]
    assert "Leaderboard" in _texts(view, "H3")
    assert [item["class"] for item in view if item["tag"] == "TABLE"][0] == "metrics-table"
    assert any(item["class"] == "run-charts" for item in view)
    assert "★ Random Forest" in _texts(view, "TD")


def test_comparable_runs_have_differences(rendered):
    shown = rendered["comparable"]
    assert _texts(shown, "H4") == ["A: run 20261030T120509Z-0000000a",
                                   "B: run 20261031T090000Z-0000000b"]
    assert "Δ = B − A" in _texts(shown, "P")[0]
    assert _texts(shown, "FIGCAPTION") == ["Random Forest (★ selected in A)"]
    cells = _texts(shown, "TD")
    assert cells[:3] == ["0.900", "0.912", "+0.012"]
    assert ["0.500", "0.512", "−0.004"] == cells[15:18]
    assert ["10 s", "11 s", "+1.2 s"] == cells[18:21]
    assert ["0.800", "0.750", "−0.050"] == cells[21:24], "the recall of a class in both"
    assert "Δ (B − A)" in _texts(shown, "TH")


def test_other_runs_are_side_by_side_with_the_reasons(rendered):
    shown = rendered["apart"]
    assert "Δ (B − A)" not in _texts(shown, "TH")
    assert not any(text.startswith(("+", "−")) for text in _texts(shown, "TD"))
    assert _texts(shown, "LI") == [
        "They were trained on different datasets (the files' SHA-256 differ).",
        "They had different numbers of training or test rows.",
        "NEW_CODE"], "a code this page doesn't know is still shown"


def test_the_models_and_settings_of_one_run(rendered):
    shown = rendered["comparable"]
    paragraphs = _texts(shown, "P")
    assert "Only in A: Decision Tree." in paragraphs, "titled"
    assert "Only in B: xgboost." in paragraphs, "a model without a title keeps its name"
    rows = [item["text"] for item in shown if item["tag"] == "TR"][-4:]
    assert rows == [f"Features3 featuresonly here: {XSS}2 features",
                    "ModelsRandom Forest, Decision TreeRandom Forest, xgboost",
                    "BalancingNone (No Balancing)SMOTE",
                    "Feature selectionNonemutual_info, 20 features"]
    nothing = rendered["no_common"]
    assert "The two runs have no model in common." in _texts(nothing, "P")
    assert any(text.startswith("None: both runs used") for text in _texts(nothing, "P"))


def test_numbers_and_times(rendered):
    assert rendered["deltas"] == ["+0.012", "−0.004", "0.000", "0.000", "0.000", "+1.2 s",
                                  "−4,414 s"], "a difference that rounds to zero has no sign"
    assert rendered["when"] == ["2026-10-30 12:05 UTC", "not a time"]


def test_text_from_a_run_stays_text(rendered):
    text = json.dumps(rendered)
    assert '"tag": "IMG"' not in text
    for key in ("row", "view", "comparable", "apart"):
        for item in rendered[key]:
            assert not any(re.match(r"on|style$", name) for name in item["attributes"]), item


def test_the_template():
    html = RUNS_TEMPLATE.read_text(encoding="utf-8")
    assert 'name="csrf-token"' not in html, "the page only reads: no token to keep"
    for region in ('id="runsStatus" class="message" role="status"',
                   'id="runTitle" tabindex="-1"', 'id="comparisonTitle" tabindex="-1"',
                   'id="runsBox" class="table-container" tabindex="0" role="region"'):
        assert region in html, region
