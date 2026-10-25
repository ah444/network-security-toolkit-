"""Step 0.4c: train-only pipeline, registries, seeds, failed models, SVM size, workers.

[L2, L3, L7, B1, B5, B20, D8]
"""

import joblib
import numpy as np
import pandas as pd
import pytest
import threadpoolctl
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.validation import check_is_fitted

from toolkit.core.data.settings import PipelineConfig, max_workers
from toolkit.core.services.ids import registry
from toolkit.core.services.ids.pipeline import STEPS, train_models
from toolkit.core.services.ids.split import split_dataset

CLASSES = ("BENIGN", "DoS", "PortScan")


def _frame(rows=300, seed=0):
    """Numeric + categorical features, imbalanced classes, some inf/empty cells."""
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.65, 0.25, 0.10])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 3.0, "PortScan": -3.0}).to_numpy()
    noisy = rng.normal(size=rows)
    noisy[::40] = np.inf
    noisy[5::40] = np.nan
    proto = np.where(label == "PortScan", "udp", rng.choice(["tcp", "udp"], size=rows))
    return pd.DataFrame({
        "m": np.arange(rows, dtype=float),  # marker: tells which rows a step saw
        "a": shift + rng.normal(size=rows),
        "b": noisy,
        "proto": pd.Series(proto, dtype="category"),
        "Label": label,
    })


FEATURES = ["a", "b", "proto"]


@pytest.fixture(scope="module")
def frame():
    return _frame()


# --- registries ----------------------------------------------------------------------

def test_registry_lists_the_v4_models_and_balancers():
    assert set(registry.MODELS) == {
        "random_forest", "logistic_regression", "decision_tree", "knn", "naive_bayes", "svm",
        "gradient_boosting", "adaboost", "extra_trees", "xgboost", "lightgbm"}
    assert set(registry.BALANCERS) == {
        "none", "class_weight", "smote", "smote_enn", "adasyn", "borderline_smote",
        "random_undersample", "tomek", "smote_tomek"}
    assert {"none", "f_classif", "mutual_info"} <= set(registry.SELECTORS)


@pytest.mark.parametrize("classes", [CLASSES, CLASSES[:2]], ids=["multiclass", "binary"])
def test_registry_builds_every_listed_model(frame, classes):
    train = frame[frame["Label"].isin(classes)]
    names = registry.available_models()
    outcome = train_models(train, "Label", features=FEATURES, models=names, workers=1)
    assert outcome.failures == ()
    assert set(outcome.models) == set(names)
    for fitted in outcome.models.values():
        assert set(fitted.predict(train)) <= set(classes)


@pytest.mark.parametrize("balancer", sorted(registry.BALANCERS))
def test_every_balancer_runs(frame, balancer):
    outcome = train_models(frame, "Label", features=FEATURES, models=("random_forest",),
                           balancer=balancer, workers=1)
    assert outcome.failures == (), outcome.failures
    assert outcome.balancer == balancer


def test_unknown_names_rejected(frame):
    with pytest.raises(ValueError, match="nope"):
        train_models(frame, "Label", features=FEATURES, models=("nope",))
    with pytest.raises(ValueError, match="nope"):
        train_models(frame, "Label", features=FEATURES, models=("decision_tree",), balancer="nope")
    with pytest.raises(ValueError, match="Label"):
        train_models(frame, "Label", features=["a", "Label"], models=("decision_tree",))


# --- train only ----------------------------------------------------------------------

def test_pipeline_step_order(frame):
    outcome = train_models(frame, "Label", features=FEATURES, models=("logistic_regression",),
                           balancer="smote", selector="f_classif", k_features=2, workers=1)
    pipeline = outcome.models["logistic_regression"].pipeline
    assert [name for name, _ in pipeline.steps] == list(STEPS)
    assert STEPS == ("prepare", "select", "sample", "model")


def test_sampler_and_selector_see_train_indices_only(monkeypatch):
    data = _frame(rows=400, seed=1)
    split = split_dataset(data, "Label", strategy="stratified")
    seen = {"select": [], "sample": []}

    class SpySelector(SelectKBest):
        def fit(self, X, y=None, **kwargs):
            seen["select"].append(np.asarray(X)[:, 0].copy())
            return super().fit(X, y, **kwargs)

    class SpySampler(RandomOverSampler):
        def fit_resample(self, X, y, **kwargs):
            seen["sample"].append(np.asarray(X)[:, 0].copy())
            return super().fit_resample(X, y, **kwargs)

    monkeypatch.setitem(registry.SELECTORS, "spy", registry.SelectorSpec(
        "spy", "Spy", lambda ctx: SpySelector(f_classif, k="all")))
    monkeypatch.setitem(registry.BALANCERS, "spy", registry.BalancerSpec(
        "spy", "Spy", "records its input", lambda ctx: SpySampler(random_state=ctx.seed)))

    outcome = train_models(split.train, "Label", features=["m", *FEATURES],
                           models=("decision_tree",), balancer="spy", selector="spy", k_features=4,
                           workers=1)
    outcome.models["decision_tree"].predict(split.test)

    train_rows = set(split.train["m"])
    for step in ("select", "sample"):
        assert len(seen[step]) == 1, f"{step} must be fitted once, on train"
        assert set(seen[step][0]) == train_rows, f"{step} saw rows outside train"


def test_unseen_category_no_crash(frame):
    train = frame[frame["proto"] != "udp"].assign(proto=lambda f: f["proto"].astype(str))
    test = frame.head(30).assign(proto=["icmp", np.nan, "udp"] * 10)
    outcome = train_models(train, "Label", features=FEATURES,
                           models=("logistic_regression", "random_forest", "knn"), workers=1)
    for fitted in outcome.models.values():
        assert len(fitted.predict(test)) == 30


SCALED = ("logistic_regression", "knn", "svm", "naive_bayes")


@pytest.mark.parametrize("model", SCALED)
def test_scaled_models_use_signed_log_then_standard_scaling(model):
    """Heavy-tailed flow features (bytes/s up to 1e9) stayed huge after v4's RobustScaler,
    and logistic regression never converged on CIC. Signed log + standard scaling bounds them."""
    rng = np.random.default_rng(7)
    rows = 400
    label = rng.choice(["BENIGN", "DoS"], size=rows)
    rate = np.where(label == "DoS", 1e9, 1e3) * rng.lognormal(0, 2, size=rows)
    data = pd.DataFrame({"rate": rate, "delta": -rate, "proto": "tcp", "Label": label})
    outcome = train_models(data, "Label", features=["rate", "delta", "proto"], models=(model,),
                           workers=1)
    prepare = outcome.models[model].pipeline.named_steps["prepare"]
    numeric = prepare.named_transformers_["numeric"]
    assert [name for name, _ in numeric.steps] == ["finite", "impute", "log", "scale"]
    prepared = prepare.transform(data[["rate", "delta", "proto"]])
    values = prepared[["numeric__rate", "numeric__delta"]].to_numpy()
    assert np.abs(values).max() < 10, "a bounded scale, not millions"
    assert np.allclose(values.mean(axis=0), 0, atol=1e-3)
    assert (np.sign(values[:, 0]) == -np.sign(values[:, 1])).all(), "the sign is kept"


@pytest.mark.parametrize("model", ["random_forest", "xgboost"])
def test_tree_models_are_not_scaled(frame, model):
    outcome = train_models(frame, "Label", features=FEATURES, models=(model,), workers=1)
    numeric = outcome.models[model].pipeline.named_steps["prepare"].named_transformers_["numeric"]
    assert [name for name, _ in numeric.steps] == ["finite", "impute"]


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest"])
def test_prepared_features_are_float32(frame, model):
    """float64 copies of 2 M CIC rows don't fit in an 8 GB laptop's memory."""
    outcome = train_models(frame, "Label", features=FEATURES, models=(model,), workers=1)
    prepared = outcome.models[model].pipeline.named_steps["prepare"].transform(frame[FEATURES])
    assert set(prepared.dtypes) == {np.dtype(np.float32)}


def test_encoding_follows_the_model(frame):
    outcome = train_models(frame, "Label", features=FEATURES,
                           models=("logistic_regression", "random_forest"), workers=1)
    assert outcome.models["logistic_regression"].encoding == "onehot"
    assert outcome.models["random_forest"].encoding == "ordinal"
    assert outcome.numeric_features == ("a", "b")
    assert outcome.categorical_features == ("proto",)


# --- seeds ---------------------------------------------------------------------------

@pytest.mark.parametrize("balancer", ["smote_enn", "smote_tomek"])
def test_identical_runs_identical_results(frame, balancer):
    def run():
        outcome = train_models(frame, "Label", features=FEATURES, models=("random_forest",),
                               balancer=balancer, workers=1)
        return outcome.models["random_forest"]

    first, second = run(), run()
    assert first.pipeline.named_steps["sample"].smote.random_state == PipelineConfig().random_state
    np.testing.assert_array_equal(first.predict(frame), second.predict(frame))
    np.testing.assert_array_equal(first.pipeline.predict_proba(frame[FEATURES]),
                                  second.pipeline.predict_proba(frame[FEATURES]))


def test_every_seeded_estimator_gets_the_seed(frame):
    outcome = train_models(frame, "Label", features=FEATURES, models=registry.available_models(),
                           balancer="smote_tomek", workers=1)
    for name, fitted in outcome.models.items():
        for key, value in fitted.pipeline.get_params().items():
            if key.endswith("random_state"):
                assert value == 42, f"{name}: {key} = {value}"


def test_tiny_classes_not_oversampled():
    data = _frame(rows=200, seed=2)
    data.loc[data.index[0], "Label"] = "Heartbleed"  # one training row: nothing to interpolate
    outcome = train_models(data, "Label", features=FEATURES, models=("decision_tree",),
                           balancer="smote", workers=1)
    assert outcome.failures == ()
    assert "SAMPLER_SKIPPED_TINY_CLASSES" in outcome.notes


# --- class weight --------------------------------------------------------------------

def test_class_weight_applied_where_supported(frame):
    outcome = train_models(frame, "Label", features=FEATURES,
                           models=("random_forest", "knn", "xgboost"), balancer="class_weight",
                           workers=1)
    forest = outcome.models["random_forest"].pipeline.named_steps["model"]
    assert forest.class_weight == "balanced"
    assert "CLASS_WEIGHT_UNSUPPORTED" in outcome.models["knn"].notes
    assert "CLASS_WEIGHT_AS_SAMPLE_WEIGHT" in outcome.models["xgboost"].notes


# --- failed models, SVM size ---------------------------------------------------------

class _Broken(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        raise RuntimeError("boom")


def test_failed_model_not_saved(frame, monkeypatch):
    monkeypatch.setitem(registry.MODELS, "broken", registry.ModelSpec(
        "broken", "Broken", "always fails", lambda ctx: _Broken(), scale=False,
        encoding="ordinal", class_weight="none"))
    outcome = train_models(frame, "Label", features=FEATURES,
                           models=("broken", "decision_tree"), workers=1)
    assert "broken" not in outcome.models
    assert [f.name for f in outcome.failures] == ["broken"]
    assert "boom" in outcome.failures[0].error
    for fitted in outcome.models.values():
        check_is_fitted(fitted.pipeline.named_steps["model"])


def test_all_models_failing_returns_no_models(frame, monkeypatch):
    monkeypatch.setitem(registry.MODELS, "broken", registry.ModelSpec(
        "broken", "Broken", "always fails", lambda ctx: _Broken(), scale=False,
        encoding="ordinal", class_weight="none"))
    outcome = train_models(frame, "Label", features=FEATURES, models=("broken",), workers=1)
    assert outcome.models == {}
    assert len(outcome.failures) == 1


def test_svm_size_check_uses_train_size():
    """v4 compared the whole file with the limit; only the training rows matter."""
    whole = _frame(rows=1000, seed=3)
    split = split_dataset(whole, "Label", strategy="stratified")
    config = PipelineConfig(svm_max_train_rows=900)
    assert len(whole) > config.svm_max_train_rows >= len(split.train)
    trained = train_models(split.train, "Label", features=FEATURES, models=("svm",),
                           config=config, workers=1)
    assert "svm" in trained.models
    skipped = train_models(whole, "Label", features=FEATURES, models=("svm",),
                           config=config, workers=1)
    assert "svm" not in skipped.models
    assert skipped.skipped == (("svm", "SVM_TRAIN_TOO_LARGE"),)


# --- workers -------------------------------------------------------------------------

def test_max_workers_setting():
    cores = __import__("os").cpu_count() or 1
    assert max_workers({}) == max(1, cores - 1)
    assert max_workers({"TOOLKIT_MAX_WORKERS": "1"}) == 1
    assert max_workers({"TOOLKIT_MAX_WORKERS": "100000"}) == cores
    for bad in ("0", "-2", "two", "1.5"):
        with pytest.raises(ValueError, match="TOOLKIT_MAX_WORKERS"):
            max_workers({"TOOLKIT_MAX_WORKERS": bad})


class _ThreadProbe(ClassifierMixin, BaseEstimator):
    """Records the native thread pools' sizes while it is being fitted."""

    sizes: list = []

    def fit(self, X, y):
        _ThreadProbe.sizes = [pool["num_threads"] for pool in threadpoolctl.threadpool_info()]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


def test_effective_workers_capped(frame, monkeypatch):
    monkeypatch.setenv("TOOLKIT_MAX_WORKERS", "2")
    requested = []
    original = joblib.Parallel.__init__

    def spy(self, *args, **kwargs):
        requested.append(kwargs.get("n_jobs", args[0] if args else None))
        original(self, *args, **kwargs)

    monkeypatch.setattr(joblib.Parallel, "__init__", spy)
    monkeypatch.setitem(registry.MODELS, "probe", registry.ModelSpec(
        "probe", "Probe", "thread probe", lambda ctx: _ThreadProbe(), scale=False,
        encoding="ordinal", class_weight="none"))

    outcome = train_models(frame, "Label", features=FEATURES,
                           models=("random_forest", "extra_trees", "knn", "xgboost", "lightgbm",
                                   "probe"))
    assert outcome.workers == 2
    assert requested, "the forest should have used joblib"
    assert max(joblib.effective_n_jobs(n) for n in requested) <= 2
    assert all(size <= 2 for size in _ThreadProbe.sizes), _ThreadProbe.sizes
    for name in ("random_forest", "extra_trees", "knn", "xgboost", "lightgbm"):
        assert outcome.models[name].pipeline.named_steps["model"].get_params()["n_jobs"] == 2


# --- prediction input (0.4g, B2) ----------------------------------------------------------

def test_feature_schema_from_the_fitted_pipeline(frame):
    outcome = train_models(frame, "Label", features=FEATURES,
                           models=("decision_tree", "logistic_regression"), workers=1)
    for fitted in outcome.models.values():
        schema = {f.name: f for f in fitted.feature_schema()}
        assert [f.name for f in fitted.feature_schema()] == FEATURES
        assert (schema["a"].kind, schema["b"].kind) == ("numeric", "numeric")
        assert schema["proto"].kind == "categorical"
        assert schema["proto"].categories == ("tcp", "udp")
        assert schema["a"].categories == ()


def test_input_frame_converts_form_values(frame):
    fitted = train_models(frame, "Label", features=FEATURES, models=("decision_tree",),
                          workers=1).models["decision_tree"]
    row = fitted.input_frame({"a": "1.5", "b": 2, "proto": " udp ", "extra": "ignored"})
    assert list(row.columns) == FEATURES and len(row) == 1
    assert row["a"].iloc[0] == 1.5 and row["b"].iloc[0] == 2.0
    assert row["proto"].iloc[0] == "udp"
    assert fitted.predict(row)[0] in CLASSES
    unseen = fitted.input_frame({"a": "0", "b": "0", "proto": "icmp"})  # unseen category
    assert fitted.predict(unseen)[0] in CLASSES
    with pytest.raises(ValueError, match="'a'"):
        fitted.input_frame({"a": "fast", "b": "0", "proto": "tcp"})
    with pytest.raises(ValueError, match="'b'"):
        fitted.input_frame({"a": "1", "proto": "tcp"})
    with pytest.raises(ValueError, match="'b'"):
        fitted.input_frame({"a": "1", "b": " ", "proto": "tcp"})


def test_predict_proba_in_class_order(frame):
    outcome = train_models(frame, "Label", features=FEATURES, models=("decision_tree", "svm"),
                           workers=1)
    tree = outcome.models["decision_tree"]
    proba = tree.predict_proba(frame.head(5))
    assert proba.shape == (5, len(tree.classes))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(np.asarray(tree.classes)[proba.argmax(axis=1)]) == list(tree.predict(frame.head(5)))
    assert outcome.models["svm"].predict_proba(frame.head(5)) is None  # LinearSVC has none
