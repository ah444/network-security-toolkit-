"""The Upload Dataset card (v5.2, the user's request of 2026-10-25): presentation only.

The answer of ``POST /api/v1/dataset`` already has structured fields (``dataset``), so the
page renders only from them and ``message`` is unchanged. The result is a panel of
label/value tiles (``summary.js``, reused by the scan results of step 2.4e); everything from
the uploaded files (file, dataset and column names) is set as text. The rendering is checked
by running the page's own modules in Node (``upload_card.mjs``), skipped where Node is not
installed; the states that need a real page (busy button, error, re-upload, focus) are
checked in the browser for each step.
"""

import io
import json
import re
import shutil
import subprocess

import pytest

from web_js import JS_DIR, REPO_ROOT, TEMPLATE, js_sources

HERE = REPO_ROOT / "tests" / "web"
APP_CSS = JS_DIR.parent / "css" / "app.css"
XSS = "<img src=x onerror=alert(1)>.csv"
NOTE = "Columns that are label-derived, IDs or time are never used as features."
ORDER = ["Dataset", "Label column", "Split", "Official test file", "Files processed",
         "Total rows", "Memory used", "Excluded from features"]


# ---------- the rendering, by the page's own modules ----------

@pytest.fixture(scope="module")
def rendered():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run([node, str(HERE / "upload_card.mjs"), str(JS_DIR)],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _tiles(panel):
    return {tile["label"]: tile for tile in panel["tiles"]}


def test_the_tiles_and_their_order(rendered):
    nsl = rendered["nsl"]
    assert [tile["label"] for tile in nsl["tiles"]] == ORDER
    assert [tile["value"] for tile in nsl["tiles"]] == [
        "nsl_kdd (full)", "attack", "official", "KDDTest+.txt22,544 rows", "2", "125,973",
        "23.1 MB", "level"]
    tiles = _tiles(nsl)
    assert tiles["Dataset"]["classes"] == ["mono", "quiet"]  # the name, then the muted "(full)"
    assert tiles["Official test file"]["classes"] == ["mono", "stat-note"]  # its rows under it
    assert tiles["Excluded from features"]["classes"] == ["mono"]


def test_a_success_panel_is_a_section_named_by_its_heading(rendered):
    nsl = rendered["nsl"]
    assert nsl["tag"] == "SECTION" and nsl["class"] == "summary-panel" and nsl["role"] is None
    assert nsl["labelledby"] == nsl["heading_id"] == "uploadSummaryTitle"
    assert nsl["heading"] == "Uploaded successfully"
    # success is the icon and the words, not only the colour; the icon is hidden from readers
    assert nsl["icon"] == "SVG" and nsl["icon_hidden"] == "true"
    assert nsl["list"] == 1 and "DL" in nsl["tags"] and "DT" in nsl["tags"]
    assert nsl["paragraphs"] == [NOTE]


def test_one_file_without_a_test_file_has_no_test_file_tile(rendered):
    single = rendered["single"]
    assert [tile["label"] for tile in single["tiles"]] == [
        label for label in ORDER if label != "Official test file"]
    assert _tiles(single)["Files processed"]["value"] == "1"


def test_the_split_is_what_the_server_says(rendered):
    assert _tiles(rendered["single"])["Split"]["value"] == "stratified"
    assert _tiles(rendered["many"])["Split"]["value"] == "temporal_per_class"


def test_no_excluded_column_says_none(rendered):
    excluded = _tiles(rendered["single"])["Excluded from features"]
    assert excluded["value"] == "None" and excluded["classes"] == ["quiet"]


def test_several_excluded_columns_are_one_list(rendered):
    excluded = _tiles(rendered["many"])["Excluded from features"]
    assert excluded["value"] == "Flow ID, Source IP, Timestamp" and excluded["classes"] == ["mono"]


def test_a_missing_label_column_says_so(rendered):
    label = _tiles(rendered["unknown"])["Label column"]
    assert label["value"] == "not detected" and label["classes"] == ["quiet"]


def test_names_from_the_files_stay_text(rendered):
    hostile = rendered["hostile"]
    values = [tile["value"] for tile in hostile["tiles"]]
    assert values[:4] == [f"{XSS} ({XSS})", XSS, "official", XSS + "22,544 rows"]
    assert values[-1] == XSS
    assert "IMG" not in hostile["tags"] and "IMG" not in rendered["error"]["tags"]


def test_an_error_is_an_alert_with_the_servers_words(rendered):
    error = rendered["error"]
    assert error["class"] == "summary-panel summary-panel-error" and error["role"] == "alert"
    assert error["heading"] == "Upload failed" and error["icon_hidden"] == "true"
    assert error["paragraphs"] == [f"Could not read {XSS} as a CSV/TXT dataset."]
    assert error["tiles"] == [] and error["list"] == 0


def test_chips_up_to_six_then_five_and_a_count(rendered):
    assert rendered["chips_none"] == []
    assert [chip["text"] for chip in rendered["chips_six"]] == [f"part-{i}.csv" for i in range(1, 7)]
    seven = rendered["chips_seven"]
    assert [chip["text"] for chip in seven] == [f"part-{i}.csv" for i in range(1, 6)] + ["+2 more"]
    assert seven[-1]["class"] == "chip chip-more"
    assert rendered["chips_twenty"][-1]["text"] == "+15 more" and len(rendered["chips_twenty"]) == 6


def test_a_chip_keeps_its_full_name_and_its_direction(rendered):
    for chip in rendered["chips_seven"][:-1] + rendered["chips_hostile"]:
        assert chip["class"] == "chip" and chip["title"] == chip["text"] and chip["dir"] == "auto"
    assert [chip["text"] for chip in rendered["chips_hostile"]] == [XSS, "تدريب.csv"]


def test_counts_have_thousands_separators(rendered):
    assert rendered["numbers"] == ["125,973", "0", "2"]


# ---------- the answer it renders from ----------

def _post(client, files):
    return client.post("/api/v1/dataset", content_type="multipart/form-data",
                       data={"dataset": [(path.open("rb"), name) for path, name in files]})


def test_the_answer_has_the_structured_fields(client, repo_root):
    fixtures = repo_root / "tests" / "fixtures"
    response = _post(client, [(fixtures / "nslkdd_train_sample.txt", "KDDTrain+.txt"),
                              (fixtures / "nslkdd_test_sample_noheader.txt", "KDDTest+.txt")])
    assert response.status_code == 200, response.get_json()
    answer = response.get_json()
    dataset = answer["dataset"]
    assert dataset["files"] == ["KDDTrain+.txt", "KDDTest+.txt"]
    assert isinstance(dataset["rows"], int) and dataset["rows"] > 0
    assert isinstance(dataset["memory_mb"], float)
    assert dataset["test_file"] == "KDDTest+.txt" and isinstance(dataset["test_rows"], int)
    profile = dataset["profile"]
    assert (profile["name"], profile["variant"], profile["label_column"],
            profile["split_strategy"], profile["blocked_columns"]) == (
        "nsl_kdd", "full", "attack", "official", ["level"])
    # the sentence is unchanged (other clients and tests read it)
    assert answer["message"].startswith(f"Successfully processed 2 file(s). Total rows: {dataset['rows']}.")
    assert " KDDTest+.txt is used as the official test file " in answer["message"]


def test_one_file_has_no_test_file(client):
    payload = b"f1,f2,Label\n1,2,A\n3,4,B\n5,6,A\n"
    response = client.post("/api/v1/dataset", content_type="multipart/form-data",
                           data={"dataset": (io.BytesIO(payload), XSS)})
    dataset = response.get_json()["dataset"]
    assert response.status_code == 200
    assert dataset["files"] == [XSS]  # the name as given: the page sets it as text
    assert dataset["test_file"] is None and dataset["test_rows"] is None
    assert dataset["profile"]["blocked_columns"] == []


# ---------- the markup, the code and the styles ----------

@pytest.fixture(scope="module")
def card():
    html = TEMPLATE.read_text(encoding="utf-8")
    return re.search(r'<div class="section" id="uploadDatasetSection">.*?<div id="uploadResult"[^>]*>',
                     html, re.S).group(0)


def test_the_form_keeps_its_ids_names_and_hooks(card):
    assert '<form id="uploadForm" enctype="multipart/form-data">' in card
    assert ('<input type="file" id="dataset" name="dataset" accept=".csv,.txt" multiple required '
            'aria-describedby="datasetChosen">') in card
    assert '<label for="dataset" class="label">Select CSV or TXT File</label>' in card
    assert '<button type="submit" class="btn btn-inline">Upload File</button>' in card
    assert '<div id="uploadResult" class="message" role="status">' in card


def test_the_dropzone_is_the_input_over_an_icon_a_title_and_chips(card):
    zone = re.search(r'<div class="file-input-wrapper dropzone">.*?</div>\s*</div>', card, re.S).group(0)
    assert zone.index('id="dataset"') < zone.index('class="file-input-placeholder"')  # input first
    assert '<span class="dropzone-icon" aria-hidden="true">' in zone
    assert '<span class="dropzone-title" id="datasetChosen">Choose file(s)...</span>' in zone
    assert '<span class="chip-list" id="datasetChips" hidden></span>' in zone


def test_uploading_turns_the_button_off_and_ignores_a_second_submit():
    data = js_sources()["data.js"]
    upload = re.search(r"async function uploadDataset\(event\) \{.*?\n\}", data, re.S).group(0)
    assert upload.index("if (uploading)") < upload.index("uploading = true")
    assert "setUploading(true);" in upload and "setUploading(false);" in upload.split("finally")[1]
    busy = re.search(r"function setUploading\(busy\) \{.*?\n\}", data, re.S).group(0)
    assert "button.disabled = busy;" in busy
    assert 'busy ? "Uploading…" : UPLOAD_LABEL' in busy
    assert 'setAttribute("aria-busy", "true")' in busy and 'removeAttribute("aria-busy")' in busy


def test_the_button_keeps_the_keyboard_focus():
    """Disabling a focused button drops the focus to the page: it is given back after the
    upload, and only if it had it (nothing else's focus is taken)."""
    data = js_sources()["data.js"]
    upload = re.search(r"async function uploadDataset\(event\) \{.*?\n\}", data, re.S).group(0)
    assert upload.index("const focused = document.activeElement === uploadButton();") < upload.index(
        "setUploading(true);")
    assert "if (focused && document.activeElement === document.body) {" in upload.split("finally")[1]
    assert ".focus()" not in data.replace("uploadButton().focus()", "")


def test_a_result_replaces_the_last_one():
    """Results never stack: the area is emptied when an upload starts, and every answer or
    error replaces its whole content."""
    data = js_sources()["data.js"]
    upload = re.search(r"async function uploadDataset\(event\) \{.*?\n\}", data, re.S).group(0)
    assert "setContent(uploadResult);" in upload
    assert "setContent(uploadResult, uploadError(error.message));" in upload
    assert 'setContent(byId("uploadResult"), datasetSummary(result.dataset));' in data
    assert "result.message" not in data.split("function showDataset")[1].split("\n}")[0]


def test_the_dropzone_shows_the_chosen_files():
    data = js_sources()["data.js"]
    assert 'byId("dataset").addEventListener("change", showChosenFiles);' in data
    chosen = re.search(r"function showChosenFiles\(\) \{.*?\n\}", data, re.S).group(0)
    assert 'setContent(byId("datasetChips"), chips(names));' in chosen
    assert 'input.title = names.join("\\n");' in chosen  # the input lies over the chips


@pytest.fixture(scope="module")
def css():
    return re.sub(r"/\*.*?\*/", "", APP_CSS.read_text(encoding="utf-8"), flags=re.S)


def _rule(css, selector, within=None):
    text = css if within is None else re.search(re.escape(within) + r"\s*\{(.*?)\n\}", css, re.S).group(1)
    body = re.search(r"(?:^|\n)\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", text).group(1)
    return dict(re.split(r"\s*:\s*", part.strip(), maxsplit=1) for part in body.split(";") if part.strip())


def test_the_result_is_left_aligned_and_the_button_on_the_left(css):
    assert _rule(css, "#uploadResult")["text-align"] == "start"
    assert _rule(css, ".summary-panel")["text-align"] == "start"
    button = _rule(css, ".btn.btn-inline")
    assert (button["width"], button["margin"], button["min-width"], button["min-height"]) == (
        "auto", "0", "10rem", "46px")


def test_four_tiles_a_row_on_wide_screens_two_elsewhere(css):
    assert _rule(css, ".stat-grid")["grid-template-columns"] == "repeat(2, minmax(0, 1fr))"
    wide = _rule(css, ".stat-grid", "@media (min-width: 1024px)")
    assert wide["grid-template-columns"] == "repeat(4, minmax(0, 1fr))"


def test_on_phones_the_chips_go_under_and_the_button_is_full_width(css):
    phone = css[css.rindex("@media (max-width: 600px)"):]
    zone = _rule(phone, ".dropzone .file-input-placeholder")
    assert zone["grid-template-areas"] == '"icon title" "chips chips"'
    assert _rule(phone, ".btn.btn-inline")["width"] == "100%"
    assert _rule(phone, ".btn.btn-inline")["min-height"] == "48px"
    assert _rule(phone, ".chip")["max-width"] == "100%"


def test_long_names_are_cut_with_an_ellipsis(css):
    chip = _rule(css, ".chip")
    assert (chip["text-overflow"], chip["white-space"], chip["overflow"]) == ("ellipsis", "nowrap", "hidden")


def test_no_movement_when_less_is_asked(css):
    reduced = _rule(css, ".section:hover,\n  .btn:hover", "@media (prefers-reduced-motion: reduce)")
    assert reduced["transform"] == "none"


def test_the_panel_colours_are_the_theme_tokens(css):
    assert _rule(css, ":root")["--muted"] == "var(--placeholder)"
    assert _rule(css, ":root")["--surface-inset"] == "var(--bg)"
    assert _rule(css, ".summary-head")["color"] == "var(--success)"
    assert _rule(css, ".summary-panel-error .summary-head")["color"] == "var(--danger)"
