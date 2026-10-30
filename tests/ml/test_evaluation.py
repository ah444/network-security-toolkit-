"""Step 0.4d: CV on train, selection before the test set, test scored once, metrics,
train-only balancing recommender, basic leaderboard [L8, L9, F2]."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from toolkit.core.data.settings import EvaluationConfig
from toolkit.core.services.ids import balancing, evaluation, pipeline, registry
from toolkit.core.services.ids.evaluation import evaluate, leaderboard_table
from toolkit.core.services.ids.split import split_dataset

CLASSES = ("BENIGN", "DoS", "PortScan")
FEATURES = ["a", "b", "proto"]
MODELS = ("decision_tree", "logistic_regression", "naive_bayes")


def _frame(rows=600, seed=0):
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
def result(split):
    return evaluate(split.train, split.test, "Label", features=FEATURES, models=MODELS, workers=1)


# --- test-set isolation --------------------------------------------------------------

def test_test_split_never_used_for_selection(split, monkeypatch):
    events = []
    test_rows = set(split.test.index)

    real_train = pipeline.train_models

    def spy_train(frame, *args, **kwargs):
        events.append(("train", set(frame.index)))
        return real_train(frame, *args, **kwargs)

    real_recommend = balancing.recommend_balancing

    def spy_recommend(labels, *args, **kwargs):
        events.append(("recommend", set(labels.index)))
        return real_recommend(labels, *args, **kwargs)

    real_select = evaluation.select_model

    def spy_select(*args, **kwargs):
        events.append(("select", set()))
        return real_select(*args, **kwargs)

    real_predict = pipeline.FittedModel.predict

    def spy_predict(self, frame):
        events.append(("predict", set(frame.index)))
        return real_predict(self, frame)

    monkeypatch.setattr(pipeline, "train_models", spy_train)
    monkeypatch.setattr(balancing, "recommend_balancing", spy_recommend)
    monkeypatch.setattr(evaluation, "select_model", spy_select)
    monkeypatch.setattr(pipeline.FittedModel, "predict", spy_predict)

    evaluate(split.train, split.test, "Label", features=FEATURES, models=MODELS,
             balancer="recommended", workers=1)

    kinds = [kind for kind, _ in events]
    assert {"train", "recommend", "select", "predict"} <= set(kinds)
    for kind, rows in events:
        if kind in ("train", "recommend"):
            assert rows.isdisjoint(test_rows), f"{kind} saw test rows"
    first_test_use = next(i for i, (kind, rows) in enumerate(events)
                          if kind == "predict" and rows & test_rows)
    assert kinds.index("select") < first_test_use, "the model was chosen after looking at test"
    assert all(kind == "predict" for kind in kinds[first_test_use:])


def test_test_scored_exactly_once(split, monkeypatch):
    test_rows = set(split.test.index)
    test_calls = []
    real_predict = pipeline.FittedModel.predict

    def spy_predict(self, frame):
        if set(frame.index) & test_rows:
            assert set(frame.index) == test_rows, "test is scored as a whole"
            test_calls.append(self.name)
        return real_predict(self, frame)

    monkeypatch.setattr(pipeline.FittedModel, "predict", spy_predict)
    evaluate(split.train, split.test, "Label", features=FEATURES, models=MODELS, workers=1)
    assert sorted(test_calls) == sorted(MODELS)


def test_selection_uses_cv_only(result):
    best = min(result.models, key=lambda m: (-m.cv.macro_f1_mean, m.name))  # tie: first name
    assert result.selected == best.name
    assert result.selection_metric == "cv_macro_f1"


# --- result fields -------------------------------------------------------------------

def test_evaluation_result_fields(result, split):
    assert {m.name for m in result.models} == set(MODELS)
    assert result.cv_folds == EvaluationConfig().cv_folds
    assert result.cv_rows == len(split.train) and not result.cv_sampled
    for model in result.models:
        cv = model.cv
        assert len(cv.macro_f1) == result.cv_folds
        assert cv.macro_f1_mean == pytest.approx(np.mean(cv.macro_f1))
        assert cv.macro_f1_std == pytest.approx(np.std(cv.macro_f1))
        assert 0 <= cv.balanced_accuracy_mean <= 1
        scores = model.test
        assert scores.rows == len(split.test)
        assert 0 <= scores.macro_f1 <= 1 and 0 <= scores.balanced_accuracy <= 1
        assert 0 <= scores.accuracy <= 1 and 0 <= scores.weighted_f1 <= 1
        assert scores.labels == CLASSES
        assert [c.label for c in scores.per_class] == list(CLASSES)
        assert sum(c.support for c in scores.per_class) == len(split.test)
        assert np.array(scores.confusion).shape == (3, 3)
        assert np.array(scores.confusion).sum() == len(split.test)
        assert model.fit_seconds > 0 and model.predict_seconds >= 0


def test_per_class_metrics_match_sklearn(result, split):
    from sklearn.metrics import precision_recall_fscore_support

    model = next(m for m in result.models if m.name == "decision_tree")
    fitted = result.training.models["decision_tree"]
    truth = split.test["Label"].astype(str).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, fitted.predict(split.test), labels=list(CLASSES), zero_division=0)
    for i, row in enumerate(model.test.per_class):
        assert (row.precision, row.recall, row.f1, row.support) == pytest.approx(
            (precision[i], recall[i], f1[i], support[i]))


def test_class_only_in_test_reported():
    """NSL-KDD's test file has attack types the training file lacks: recall 0, not a crash."""
    train = _frame(seed=1)
    test = _frame(rows=200, seed=2)
    test.loc[test.index[:10], "Label"] = "apache2"
    out = evaluate(train, test, "Label", features=FEATURES, models=("decision_tree",), workers=1)
    assert out.classes_only_in_test == ("apache2",)
    row = next(c for c in out.models[0].test.per_class if c.label == "apache2")
    assert (row.recall, row.support) == (0.0, 10)


def test_failed_model_left_out(split, monkeypatch):
    class Broken(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            raise RuntimeError("boom")

    monkeypatch.setitem(registry.MODELS, "broken", registry.ModelSpec(
        "broken", "Broken", "always fails", lambda ctx: Broken(), scale=False,
        encoding="ordinal", class_weight="none"))
    out = evaluate(split.train, split.test, "Label", features=FEATURES,
                   models=("broken", "decision_tree"), workers=1)
    assert [m.name for m in out.models] == ["decision_tree"]
    assert [f.name for f in out.failures] == ["broken"]
    assert out.selected == "decision_tree"


def test_cv_runs_on_a_stratified_sample_of_large_training_data():
    train = _frame(rows=3000, seed=3)
    test = _frame(rows=300, seed=4)
    out = evaluate(train, test, "Label", features=FEATURES, models=("decision_tree",),
                   config=EvaluationConfig(cv_max_rows=600), workers=1)
    assert (out.cv_rows, out.cv_sampled) == (600, True)
    assert out.training.train_rows == 3000, "the final model uses all training rows"


def test_deterministic(split):
    first = evaluate(split.train, split.test, "Label", features=FEATURES, models=MODELS, workers=1)
    again = evaluate(split.train, split.test, "Label", features=FEATURES, models=MODELS, workers=1)
    assert [m.cv.macro_f1 for m in first.models] == [m.cv.macro_f1 for m in again.models]
    assert [m.test.macro_f1 for m in first.models] == [m.test.macro_f1 for m in again.models]


# --- leaderboard ---------------------------------------------------------------------

def test_leaderboard_rows(result):
    rows = result.leaderboard
    assert [r.rank for r in rows] == list(range(1, len(MODELS) + 1))
    assert [r.cv_macro_f1_mean for r in rows] == sorted((r.cv_macro_f1_mean for r in rows),
                                                        reverse=True)
    assert rows[0].model == result.selected and rows[0].selected
    assert not any(r.selected for r in rows[1:])
    for row in rows:
        assert [label for label, _ in row.test_recall_per_class] == list(CLASSES)
        assert row.test_min_recall == min(v for _, v in row.test_recall_per_class)


def test_leaderboard_table_text(result):
    text = leaderboard_table(result)
    lines = text.splitlines()
    assert "CV macro-F1" in lines[0] and "Test macro-F1" in lines[0]
    for name in MODELS:
        assert any(name in line for line in lines)
    assert "*" in next(line for line in lines if result.selected in line)


# --- balancing recommender (train data only) ------------------------------------------

def test_recommender_uses_train_only(split, monkeypatch):
    seen = []
    real = balancing.recommend_balancing

    def spy(labels, num_features):
        seen.append((labels.value_counts().to_dict(), num_features))
        return real(labels, num_features)

    monkeypatch.setattr(balancing, "recommend_balancing", spy)
    out = evaluate(split.train, split.test, "Label", features=FEATURES, models=("decision_tree",),
                   balancer="recommended", workers=1)
    assert seen == [(split.train["Label"].value_counts().to_dict(), len(FEATURES))]
    assert out.balancer == out.balancer_recommendation.recommended
    assert out.balancer in registry.BALANCERS


@pytest.mark.parametrize(("counts", "features", "expected", "code"), [
    ({"a": 500, "b": 400}, 10, "none", "MILD_IMBALANCE"),
    ({"a": 5000, "b": 3}, 10, "class_weight", "SMALLEST_CLASS_TOO_SMALL_FOR_SMOTE"),
    ({f"c{i}": 10_000 if i == 0 else 50 for i in range(12)}, 10, "class_weight",
     "MANY_CLASSES_EXTREME_IMBALANCE"),
    ({"a": 150_000, "b": 2_000, "c": 1_000}, 10, "smote_enn", "MULTICLASS_VERY_LARGE_SEVERE"),
    ({"a": 50_000, "b": 2_000, "c": 1_000}, 10, "smote", "MULTICLASS_LARGE_SEVERE"),
    ({"a": 600, "b": 100}, 10, "class_weight", "SMALL_BINARY"),
    ({"a": 5_000, "b": 1_000}, 10, "smote", "MODERATE_MEDIUM"),
    ({"a": 5_000, "b": 100}, 10, "borderline_smote", "SEVERE_LOW_DIMENSIONAL"),
])
def test_recommender_rules_match_v4(counts, features, expected, code):
    labels = pd.Series([name for name, n in counts.items() for _ in range(n)])
    recommendation = balancing.recommend_balancing(labels, features)
    assert (recommendation.recommended, recommendation.reason_code) == (expected, code)
    assert recommendation.recommended in registry.BALANCERS
    assert all(a in registry.BALANCERS for a in recommendation.alternatives)
