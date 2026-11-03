"""v5.3 F3c: why one prediction came out as it did. Each feature's value replaced by its
typical training value (the median of a number, the most frequent category), the change in
the predicted class's probability measured; the typical values written in the record at
training time; a bundle saved before falls back to its imputer's medians (numbers only)."""

import json

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.settings import ExplainConfig
from toolkit.core.services.ids.bundle import unpack
from toolkit.core.services.ids.explanation import (
    NO_PROBABILITIES,
    PIPELINE,
    TRAINING,
    explain,
    typical_values,
)
from toolkit.core.services.ids.pipeline import typical_values as training_typical
from toolkit.core.services.ids.runs import TrainSettings, train

CLASSES = ("BENIGN", "DoS", "PortScan")
KEY = bytes(range(32))  # the key the legacy fixtures were signed with


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
def run():
    return train(_frame(), settings=TrainSettings(
        label="Label", strategy="stratified", workers=1,
        models=("random_forest", "logistic_regression", "svm")))


def _model(run, name):
    return run.evaluation.training.models[name]


def _why(run, name, values, config=None):
    fitted = _model(run, name)
    typical, source = typical_values(fitted, run.to_dict()["training"])
    return explain(fitted, fitted.input_frame(values), typical, source, config)


def test_the_typical_values_are_the_training_splits(run):
    train_rows = run.split.train
    typical = run.to_dict()["training"]["typical_values"]
    assert list(typical) == ["a", "b", "proto"]
    assert typical["a"] == pytest.approx(float(np.median(train_rows["a"])), abs=1e-6)
    assert typical["proto"] == train_rows["proto"].mode()[0]
    assert typical["a"] != pytest.approx(float(np.median(run.split.test["a"])), abs=1e-6)
    json.dumps(typical, allow_nan=False)


def test_typical_values_of_odd_columns():
    frame = pd.DataFrame({"n": [1.0, np.inf, -np.inf, np.nan, 3.0], "empty": [np.nan] * 5,
                          "rate": [0.0, 0.01, 0.02, 0.01, 0.03],
                          "tie": ["b", "a", "b", "a", None], "none": [None] * 5})
    assert dict(training_typical(frame, ("n", "empty", "rate"), ("tie", "none"))) == \
        {"n": 2.0, "empty": None, "rate": 0.01, "tie": "a", "none": None}, \
        "inf is missing, as imputed; the float32 median written as 0.01"


def test_a_value_far_from_the_typical_one_pushes_toward(run):
    why = _why(run, "logistic_regression", {"a": -2.5, "b": 0.0, "proto": "udp"})
    assert why.prediction == "PortScan" and why.source == TRAINING and why.problem is None
    assert why.toward[0].feature == "a" and why.toward[0].value == -2.5
    assert why.toward[0].typical == run.to_dict()["training"]["typical_values"]["a"]
    assert "proto" not in [r.feature for r in why.toward + why.away], "already the typical value"


def test_the_effect_is_the_change_in_probability(run):
    fitted = _model(run, "logistic_regression")
    values = {"a": 2.4, "b": 1.5, "proto": "tcp"}
    why = _why(run, "logistic_regression", values)
    column = fitted.classes.index(why.prediction)
    probability = fitted.predict_proba(fitted.input_frame(values))[0][column]
    assert why.probability == pytest.approx(probability)
    typical = run.to_dict()["training"]["typical_values"]
    for reason in why.toward + why.away:
        replaced = fitted.input_frame({**values, reason.feature: typical[reason.feature]})
        assert reason.effect == pytest.approx(probability - fitted.predict_proba(replaced)[0][column])
    assert [r.effect for r in why.toward] == sorted((r.effect for r in why.toward), reverse=True)
    assert [r.effect for r in why.away] == sorted(r.effect for r in why.away)
    assert all(r.effect > 0 for r in why.toward) and all(r.effect < 0 for r in why.away)


def test_a_category_is_explained(run):
    why = _why(run, "logistic_regression", {"a": 0.4, "b": -0.1, "proto": "tcp"})
    reasons = {r.feature: r for r in why.toward + why.away}
    assert reasons["proto"].value == "tcp" and reasons["proto"].typical == "udp"


def test_at_most_the_top_features(run):
    why = _why(run, "logistic_regression", {"a": 2.4, "b": 1.5, "proto": "tcp"},
               ExplainConfig(local_top=1))
    assert len(why.toward) <= 1 and len(why.away) <= 1 and why.toward


def test_a_model_without_probabilities(run):
    why = _why(run, "svm", {"a": -2.5, "b": 0.0, "proto": "udp"})
    assert why.problem == NO_PROBABILITIES and why.probability is None
    assert why.prediction == str(_model(run, "svm").predict(
        _model(run, "svm").input_frame({"a": -2.5, "b": 0.0, "proto": "udp"}))[0])
    assert why.toward == () and why.away == ()


def test_a_feature_without_a_typical_value(run):
    fitted = _model(run, "logistic_regression")
    recorded = {"typical_values": {"a": 0.3, "b": None, "proto": "udp"}}
    typical, source = typical_values(fitted, recorded)
    assert typical == {"a": 0.3, "proto": "udp"} and source == TRAINING
    why = explain(fitted, fitted.input_frame({"a": -2.5, "b": 3.0, "proto": "tcp"}), typical, source)
    assert why.without_typical == ("b",)


@pytest.mark.parametrize("generation", ["old_names", "v51_names"])
def test_an_older_bundle_explains_its_numbers(fixtures_dir, generation):
    """Saved before F3: no typical values in its record; the medians its imputer learnt."""
    folder = fixtures_dir / "legacy"
    bundle = unpack((folder / f"{generation}.bundle").read_bytes(), key=KEY)
    expected = json.loads((folder / f"{generation}.expected.json").read_text(encoding="utf-8"))
    assert "typical_values" not in bundle.run["training"]
    for name, labels in expected["predictions"].items():
        fitted = bundle.models[name]
        typical, source = typical_values(fitted, bundle.run["training"])
        assert source == PIPELINE and set(typical) == {"a", "b", "c"}, "the numbers only"
        for values, label in zip(expected["rows"], labels, strict=True):
            why = explain(fitted, fitted.input_frame(values), typical, source)
            assert why.prediction == str(label) and why.without_typical == ("proto",)
