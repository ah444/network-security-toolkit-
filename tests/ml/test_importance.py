"""v5.3 F3b: permutation importance on the test split. scikit-learn's own numbers, asked one
feature at a time (so a job can show its progress and stop); a stratified sample of the
test rows above the limit; the same numbers twice with the seed; the warning when one
feature carries most of the effect."""

import numpy as np
import pandas as pd
import pytest
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from toolkit.core.data.settings import ExplainConfig
from toolkit.core.services.ids.evaluation import evaluate
from toolkit.core.services.ids.importance import (
    FeatureImportance,
    concentrated_feature,
    permutation_importances,
    scored_rows,
)
from toolkit.core.services.ids.split import split_dataset

CLASSES = ("BENIGN", "DoS", "PortScan")
FEATURES = ("a", "b", "proto")
MODELS = ("decision_tree", "logistic_regression")


def _frame(rows=900, seed=0):
    """``a`` separates the classes, ``proto`` marks PortScan, ``b`` is noise."""
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.65, 0.25, 0.10])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 2.0, "PortScan": -2.0}).to_numpy()
    return pd.DataFrame({
        "a": shift + rng.normal(size=rows),
        "b": rng.normal(size=rows),
        "proto": np.where(label == "PortScan", "udp", rng.choice(["tcp", "udp"], size=rows)),
        "Label": label,
    })


@pytest.fixture(scope="module")
def split():
    return split_dataset(_frame(), "Label", strategy="stratified")


@pytest.fixture(scope="module")
def models(split):
    result = evaluate(split.train, split.test, "Label", features=FEATURES, models=MODELS, workers=1)
    return result.training.models


class Counting:
    """A ProgressSink that records what was reported, and stops after ``stop_after`` steps."""

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


@pytest.mark.parametrize("name", MODELS)
def test_the_numbers_are_scikit_learns(models, split, name):
    """One call of scikit-learn's function for every feature gives the same numbers."""
    fitted = models[name]

    class Whole:
        def fit(self, *_args):
            return self

        def predict(self, frame):
            return fitted.predict(frame)

    truth = split.test["Label"].astype(str).to_numpy()
    labels = sorted(set(fitted.classes) | set(truth))
    expected = permutation_importance(
        Whole(), split.test[list(FEATURES)], truth, n_repeats=5, random_state=42,
        scoring=lambda e, x, y: f1_score(y, e.predict(x), labels=labels, average="macro",
                                         zero_division=0))
    found = {item.feature: item for item in
             permutation_importances(fitted, split.test, "Label", seed=42).features}
    for index, feature in enumerate(FEATURES):
        assert found[feature].mean == expected.importances_mean[index], feature
        assert found[feature].std == expected.importances_std[index], feature


def test_what_the_model_relies_on(models, split):
    found = permutation_importances(models["logistic_regression"], split.test, "Label", seed=42)
    assert [item.feature for item in found.features] == ["a", "proto", "b"], "ranked by the mean"
    assert found.features[0].mean > 0.3 and abs(found.features[-1].mean) < 0.03, "b is noise"
    assert (found.model, found.rows, found.test_rows, found.repeats, found.seed) == \
        ("logistic_regression", len(split.test), len(split.test), 5, 42)
    truth = split.test["Label"].astype(str)
    predicted = models["logistic_regression"].predict(split.test)
    assert found.baseline_macro_f1 == pytest.approx(f1_score(truth, predicted, average="macro"))


def test_the_same_numbers_twice_with_the_seed(models, split):
    fitted = models["decision_tree"]
    first = permutation_importances(fitted, split.test, "Label", seed=42)
    assert permutation_importances(fitted, split.test, "Label", seed=42) == first
    assert permutation_importances(fitted, split.test, "Label", seed=7) != first


def test_a_sample_of_the_test_rows_above_the_limit(models, split):
    config = ExplainConfig(importance_max_rows=60)
    test = split.test.copy()
    test.loc[test.index[:5], "Label"] = np.nan  # rows without a label are not scored
    rows = scored_rows(test, "Label", 42, config)
    assert len(rows) == 60 and rows["Label"].notna().all()
    assert set(rows["Label"]) == set(CLASSES) and rows.index.isin(test.index).all()
    assert scored_rows(test, "Label", 42, config).index.equals(rows.index), "seeded"
    found = permutation_importances(models["decision_tree"], test, "Label", seed=42, config=config)
    assert (found.rows, found.test_rows) == (60, len(test) - 5)
    assert len(scored_rows(test, "Label", 42)) == len(test) - 5, "5,000 by default"


def test_progress_is_one_step_per_feature(models, split):
    progress = Counting()
    permutation_importances(models["decision_tree"], split.test, "Label", seed=42,
                            progress=progress)
    assert (progress.total, progress.done, progress.finished) == (3, 3, 1)


def test_cancel_stops_between_two_features(models, split, monkeypatch):
    from toolkit.core.services.ids import importance

    asked = []
    real = importance.permutation_importance

    def counted(estimator, *args, **kwargs):
        asked.append(estimator.feature)
        return real(estimator, *args, **kwargs)

    monkeypatch.setattr(importance, "permutation_importance", counted)
    progress = Counting(stop_after=1)
    with pytest.raises(Stop):
        permutation_importances(models["decision_tree"], split.test, "Label", seed=42,
                                progress=progress)
    assert asked == ["a"] and progress.finished == 1


def test_the_model_is_not_trained_again(models, split, monkeypatch):
    fitted = models["decision_tree"]
    monkeypatch.setattr(type(fitted.pipeline), "fit", lambda *a, **k: pytest.fail("fit"))
    permutation_importances(fitted, split.test, "Label", seed=42)


def test_a_copy_of_the_label_carries_most_of_the_effect():
    frame = _frame()
    frame["leak"] = frame["Label"].map({"BENIGN": 0, "DoS": 1, "PortScan": 2}).astype(float)
    split = split_dataset(frame, "Label", strategy="stratified")
    fitted = evaluate(split.train, split.test, "Label", features=("a", "b", "leak"),
                      models=("decision_tree",), workers=1).training.models["decision_tree"]
    found = permutation_importances(fitted, split.test, "Label", seed=42)
    assert found.features[0].feature == "leak" and found.concentrated == "leak"


@pytest.mark.parametrize(("means", "concentrated"), [
    ((0.6, 0.3, 0.1), "x0"),
    ((0.5, 0.5), None),  # half is not most
    ((0.3, 0.3, 0.3), None),
    ((0.2, -0.5, 0.05), "x0"),  # only the positive drops are summed
    ((0.0, -0.1), None),
    ((0.9,), None),  # one feature: nothing to compare it with
])
def test_the_concentrated_feature(means, concentrated):
    features = [FeatureImportance(f"x{i}", mean, 0.0) for i, mean in enumerate(means)]
    assert concentrated_feature(features, 0.5) == concentrated
