"""Step 0.4g: train-only previews for the web trainer (feature scores, balancing) [L2, L3]."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from toolkit.ids_core import registry
from toolkit.ids_core.preview import preview_balancing, recommend_features

CLASSES = ("BENIGN", "DoS", "PortScan")


def _train(rows=300, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=rows, p=[0.7, 0.2, 0.1])
    shift = pd.Series(label).map({"BENIGN": 0.0, "DoS": 4.0, "PortScan": -4.0}).to_numpy()
    return pd.DataFrame({
        "strong": shift + rng.normal(size=rows),
        "weak": rng.normal(size=rows),
        "constant": np.ones(rows),
        "proto": pd.Series(np.where(label == "PortScan", "udp", rng.choice(["tcp", "udp"],
                                                                        size=rows)),
                           dtype="category"),
        "Flow ID": [f"f{i}" for i in range(rows)],  # ID column: blocked by the guard
        "copy": label,  # label copy: blocked by the guard
        "Label": label,
    })


@pytest.fixture(scope="module")
def train():
    return _train()


# --- feature recommendation ----------------------------------------------------------------

def test_recommend_features_on_train_excludes_blocked(train):
    before = train.copy()
    result = recommend_features(train, "Label")
    pd.testing.assert_frame_equal(train, before)  # the data is not changed
    names = [name for name, _ in result.scores]
    assert set(names) == {"strong", "weak", "constant", "proto"}
    assert names[0] == "strong"
    assert dict(result.scores)["constant"] == 0.0
    assert dict(result.blocked) == {"Flow ID": "NAME_PATTERN", "copy": "LABEL_RELABELLING"}
    assert (result.rows, result.train_rows, result.method) == (300, 300, "f_classif")
    assert recommend_features(train, "Label").scores == result.scores  # deterministic


def test_recommend_features_respects_candidates_and_profile(train):
    result = recommend_features(train, "Label", features=["strong", "weak", "Flow ID"],
                                profile_blocked=("weak",))
    assert [name for name, _ in result.scores] == ["strong"]
    assert dict(result.blocked) == {"Flow ID": "NAME_PATTERN", "weak": "PROFILE_BLOCKED"}


def test_one_value_per_class_scores_inf_without_warnings(train):
    """Step 1.3d: the F-test divides by a zero within-class spread; the preview keeps inf
    (shown as ∞), SelectKBest ranks it with the largest float, and nothing warns."""
    data = train.assign(port_flag=(train["Label"] == "PortScan").astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = recommend_features(data, "Label")
        X = data[["strong", "port_flag", "constant"]].to_numpy()
        y = data["Label"].to_numpy()
        scores = registry._Scores("f_classif", 0)
        ranked = scores(X, y)
        support = registry._TopK(scores, k=1).fit(X, y).get_support()
    assert result.scores[0] == ("port_flag", math.inf)
    assert ranked[1] == np.finfo(np.float64).max and np.isnan(ranked[2])
    assert support.tolist() == [False, True, False]


def test_recommend_features_samples_big_training_splits(train):
    result = recommend_features(train, "Label", max_rows=120)
    assert (result.rows, result.train_rows) == (120, 300)
    assert result.scores[0][0] == "strong"


# --- balancing preview ---------------------------------------------------------------------

def _counts(train):
    return dict(train["Label"].value_counts())


def test_preview_smote_grows_minorities_on_train_only(train):
    before_frame = train.copy()
    preview = preview_balancing(train, "Label", "smote", features=["strong", "weak", "proto"])
    pd.testing.assert_frame_equal(train, before_frame)
    assert dict(preview.before) == _counts(train)
    majority = max(_counts(train).values())
    assert dict(preview.after) == {c: majority for c in CLASSES}
    assert preview.changes_rows and preview.balancer == "smote"
    assert (preview.rows, preview.train_rows, preview.sampled) == (300, 300, False)


def test_preview_undersample_and_weights(train):
    smallest = min(_counts(train).values())
    under = preview_balancing(train, "Label", "random_undersample", features=["strong"])
    assert dict(under.after) == {c: smallest for c in CLASSES}
    for name in ("none", "class_weight"):
        same = preview_balancing(train, "Label", name, features=["strong"])
        assert same.after == same.before and not same.changes_rows
    assert "CLASS_WEIGHT_CHANGES_WEIGHTS_NOT_ROWS" in preview_balancing(
        train, "Label", "class_weight", features=["strong"]).notes


def test_preview_recommended_resolves_from_train_labels(train):
    preview = preview_balancing(train, "Label", "recommended", features=["strong", "weak"])
    assert preview.requested == "recommended"
    assert preview.recommendation is not None
    assert preview.balancer == preview.recommendation.recommended
    assert preview.recommendation.train_rows == 300


def test_preview_samples_big_training_splits(train):
    preview = preview_balancing(train, "Label", "smote", features=["strong"], max_rows=150)
    assert (preview.rows, preview.train_rows, preview.sampled) == (150, 300, True)
    assert sum(dict(preview.before).values()) == 150
    assert "SAMPLED" in preview.notes


def test_preview_unknown_balancer_refused(train):
    with pytest.raises(ValueError, match="balancing"):
        preview_balancing(train, "Label", "magic", features=["strong"])
