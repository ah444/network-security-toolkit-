"""5.1 gate (plan §8, accessibility: the whole flow with the keyboard only): keyboard focus
never falls to the page body during a training. Train is disabled while the job runs, and
Cancel is disabled once pressed, so focus moves on before each button loses it."""

import re

from web_js import TEMPLATE, js_sources


def _function(source, name):
    """The body of a top-level ``function name(...) {...}`` (up to the next top-level ``}``)."""
    match = re.search(rf"^(?:async )?function {name}\(.*?^\}}", source, re.S | re.M)
    assert match, name
    return match.group(0)


def test_focus_targets_can_take_focus():
    html = TEMPLATE.read_text(encoding="utf-8")
    for element_id in ("trainingProgress", "configResult"):
        assert re.search(rf'id="{element_id}"[^>]*tabindex="-1"', html), element_id


def test_job_start_moves_focus_to_cancel():
    """Train → the status ("Training models...") while the job is submitted → Cancel."""
    train = _function(js_sources()["train.js"], "train")
    disable = train.index("""querySelector('button[type="submit"]').disabled = true;""")
    assert train.index("configResult.focus();") < disable
    enable = train.index('byId("cancelTrainingBtn").disabled = false;')
    assert train.index('byId("cancelTrainingBtn").focus();') > enable


def test_cancel_moves_focus_to_the_progress_before_it_is_disabled():
    cancel = _function(js_sources()["train.js"], "cancelJob")
    assert (cancel.index('byId("trainingProgress").focus();')
            < cancel.index('byId("cancelTrainingBtn").disabled = true;'))


def test_job_end_moves_lost_focus_to_the_result():
    source = js_sources()["train.js"]
    end = _function(source, "endJob")
    assert (end.index('focusLost(byId("trainingProgress"))')
            < end.index('show(byId("trainingProgress"), false)')), "checked before it is hidden"
    assert 'byId("configResult").focus();' in end
    lost = _function(source, "focusLost")
    assert "document.body" in lost and "region.contains(active)" in lost, \
        "only focus that was lost or in the progress moves; focus the user put elsewhere stays"
