"""v5.1 step 1.3c: the trainer as steps (Data, Explore, Clean, Features, Train, Use), one
shown at a time, and the page resuming from GET /api/v1/workspace after a reload."""

import re
from html.parser import HTMLParser

import pytest

from web_js import TEMPLATE, js_sources

STEPS = ["data", "explore", "clean", "features", "train", "use"]


class _Page(HTMLParser):
    """Each element's id and classes, with the step section it is in."""

    VOID = {"input", "br", "meta", "link", "img", "hr"}

    def __init__(self):
        super().__init__()
        self.stack, self.elements, self.steps, self.links = [], [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        step = next((s for s in reversed(self.stack) if s), None)
        if tag == "section" and "step" in classes:
            self.steps.append(attrs)
            step = attrs["data-step"]
        if tag == "button" and "step-link" in classes:
            self.links.append(attrs)
        self.elements.append({"tag": tag, "attrs": attrs, "classes": classes, "step": step})
        if tag not in self.VOID:
            self.stack.append(attrs["data-step"] if tag == "section" and "step" in classes else None)

    def handle_endtag(self, tag):
        if tag not in self.VOID:
            self.stack.pop()


@pytest.fixture(scope="module")
def page():
    parser = _Page()
    parser.feed(TEMPLATE.read_text(encoding="utf-8"))
    return parser


def _step_of(page, element_id):
    return next(e["step"] for e in page.elements if e["attrs"].get("id") == element_id)


def test_step_bar_and_sections_follow_the_six_steps(page):
    assert [link["data-step"] for link in page.links] == STEPS
    assert [section["data-step"] for section in page.steps] == STEPS
    assert re.search(r'const STEPS = \["data", "explore", "clean", "features", "train", "use"\]',
                     js_sources()["steps.js"])


def test_only_data_is_open_when_the_page_loads(page):
    assert [s["data-step"] for s in page.steps if "hidden" not in s] == ["data"]
    assert [link["data-step"] for link in page.links if "disabled" not in link] == ["data"]


def test_every_step_has_a_focusable_heading(page):
    for section in page.steps:
        step = section["data-step"]
        title = f"step-{step}-title"
        assert section["id"] == f"step-{step}" and section["aria-labelledby"] == title
        heading = next(e for e in page.elements if e["attrs"].get("id") == title)
        assert heading["tag"] == "h2" and heading["attrs"].get("tabindex") == "-1"
    assert 'byId("step-" + step + "-title").focus()' in js_sources()["steps.js"]


def test_back_and_next_buttons(page):
    def buttons(name):
        return [e["step"] for e in page.elements if name in e["classes"]]
    assert buttons("step-back") == STEPS[1:]
    assert buttons("step-next") == STEPS[:-1]


def test_every_section_is_in_one_step(page):
    sections = [e for e in page.elements if "section" in e["classes"]]
    assert len(sections) == 13 and all(e["step"] in STEPS for e in sections)
    assert not [e for e in page.elements if e["tag"] == "section" and "step" in e["classes"]
                and e["step"] != e["attrs"]["data-step"]]  # steps are not nested


@pytest.mark.parametrize(("element_id", "step"), [
    ("uploadForm", "data"), ("loadModelForm", "data"),
    ("targetForDist", "explore"), ("trainTestSlider", "explore"),
    ("targetValuesToRemove", "clean"), ("balancingMethod", "clean"),
    ("recommendBtn", "features"), ("inputFeatures", "features"),
    ("targetColumn", "train"), ("modelSelection", "train"), ("confusionMatrixSection", "train"),
    ("predictionForm", "use"), ("exportForm", "use"), ("predictLoadedForm", "use"),
])
def test_controls_are_in_their_step(page, element_id, step):
    assert _step_of(page, element_id) == step


def test_unlock_rules():
    steps = js_sources()["steps.js"]
    assert 'const DATASET_STEPS = ["explore", "clean", "features", "train"];' in steps
    assert 'on("dataset:loaded", () => unlock(DATASET_STEPS));' in steps
    assert 'on("run:ready", () => unlock(["use"]));' in steps
    assert 'on("bundle:loaded", () => unlock(["use"]));' in steps
    assert 'return path === "bundle" ? ["data", "use"] : STEPS;' in steps


def test_last_step_is_kept_for_this_tab_only():
    steps = js_sources()["steps.js"]
    assert "sessionStorage.setItem(STORAGE_KEY, step)" in steps
    assert "localStorage" not in steps
    assert steps.count("try {") == 2  # storage can be blocked: the page works without it


def test_target_chosen_in_explore_is_prefilled_later():
    explore, train = js_sources()["explore.js"], js_sources()["train.js"]
    assert 'byId("targetForDist").addEventListener("change", chooseTarget);' in explore
    assert 'byId("targetForRec").value = state.target;' in explore
    assert 'emit("target:chosen", state.target);' in explore
    assert 'on("target:chosen", prefillTarget);' in train


def test_page_resumes_after_setting_up_every_module():
    main = js_sources()["main.js"]
    order = [main.index(call) for call in ("setupData();", "setupExplore();", "setupTrain();",
                                           "setupPredict();", "setupSteps();", "resume();")]
    assert order == sorted(order)
    resume = js_sources()["resume.js"]
    assert 'emit("workspace:resumed", await getJSON("/api/v1/workspace"));' in resume
    assert resume.index('emit("workspace:resumed"') < resume.index('emit("steps:restore")')


@pytest.mark.parametrize(("module", "part"), [
    ("data.js", "answer.dataset"), ("data.js", "answer.bundle"), ("train.js", "answer.run"),
    ("train.js", "answer.job"), ("predict.js", "answer.export"),
])
def test_every_resumed_part_is_shown(module, part):
    code = js_sources()[module]
    assert 'on("workspace:resumed"' in code and part in code
