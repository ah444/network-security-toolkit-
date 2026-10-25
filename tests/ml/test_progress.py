"""v5.1 step 1.2d: training reports its progress through the ProgressSink port.

Steps: split + leakage guard + models x (CV folds + final fit); failed and skipped models
count too, so a run always ends at 100 %. An exception raised by ``advance()`` (a
cancelled job) stops training and is never recorded as a model failure.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from toolkit.core.data.settings import PipelineConfig
from toolkit.core.services.ids import registry
from toolkit.core.services.ids.runs import TrainSettings, total_steps, train
from toolkit.core.services.ports import ProgressSink

CLASSES = ("BENIGN", "DoS", "PortScan")


def _frame(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.65, 0.25, 0.10])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 2.0, "PortScan": -2.0}).to_numpy()
    return pd.DataFrame({"a": shift + rng.normal(size=rows), "b": rng.normal(size=rows),
                         "Label": label})


SETTINGS = TrainSettings(label="Label", strategy="stratified", balancer="none", workers=1,
                         models=("decision_tree", "logistic_regression"))


class Counting:
    """A ProgressSink that records what training reported."""

    def __init__(self, stop_after=None):
        self.total, self.done, self.finished, self.stop_after = None, 0, 0, stop_after

    def start(self, total):
        self.total = total

    def advance(self, done=1):
        self.done += done
        if self.stop_after is not None and self.done >= self.stop_after:
            raise Stop("cancelled")

    def finish(self):
        self.finished += 1


class Stop(Exception):
    pass


class Broken(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        raise RuntimeError("boom")


@pytest.fixture
def broken_model(monkeypatch):
    monkeypatch.setitem(registry.MODELS, "broken", registry.ModelSpec(
        "broken", "Broken", "always fails", lambda ctx: Broken(), scale=False,
        encoding="ordinal", class_weight="none"))


def test_counting_sink_is_a_progress_sink():
    assert isinstance(Counting(), ProgressSink)


def test_steps_add_up_to_the_total():
    sink = Counting()
    train(_frame(), settings=SETTINGS, progress=sink)
    assert sink.total == total_steps(SETTINGS) == 2 + 2 * (3 + 1)
    assert sink.done == sink.total and sink.finished == 1


def test_failed_and_skipped_models_still_count(broken_model):
    settings = dataclasses.replace(
        SETTINGS, models=("broken", "svm", "decision_tree"),
        pipeline=PipelineConfig(svm_max_train_rows=10))  # svm skipped: too many rows
    sink = Counting()
    run = train(_frame(), settings=settings, progress=sink)
    assert [f.name for f in run.evaluation.failures] == ["broken"]
    assert set(run.evaluation.training.models) == {"decision_tree"}  # svm skipped in CV
    assert sink.done == sink.total == 2 + 3 * 4


def test_every_model_failing_still_reaches_the_total(broken_model):
    settings = dataclasses.replace(SETTINGS, models=("broken",))
    sink = Counting()
    run = train(_frame(), settings=settings, progress=sink)
    # Found in 1.2d: the final fit got an empty list and trained the *default* models
    # (random forest) that nobody chose. An empty list now trains nothing.
    assert not run.evaluation.training.models
    assert run.evaluation.selected is None
    assert sink.done == sink.total


@pytest.mark.parametrize("stop_after", [1, 3, 8])  # split, inside CV, inside the final fit
def test_exception_from_advance_stops_training(stop_after, caplog):
    sink = Counting(stop_after=stop_after)
    with pytest.raises(Stop):
        train(_frame(), settings=SETTINGS, progress=sink)
    assert sink.finished == 1, "finish() runs after an error too"
    assert "Training decision_tree failed" not in caplog.text
    assert "Training logistic_regression failed" not in caplog.text


def test_without_a_sink_nothing_changes():
    run = train(_frame(), settings=SETTINGS)
    assert set(run.evaluation.training.models) == {"decision_tree", "logistic_regression"}
