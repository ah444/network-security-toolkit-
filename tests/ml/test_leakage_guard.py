"""Step 0.4a: leakage guard (single-feature rule, name patterns, ID-like columns) [L6, D9]."""

import numpy as np
import pandas as pd
import pytest

from toolkit.data.settings import LeakageConfig
from toolkit.ids_core.datasets import load_dataset
from toolkit.ids_core.leakage import check_leakage, name_pattern


@pytest.fixture(scope="module")
def leak_frame(request):
    path = request.config.rootpath / "tests" / "fixtures" / "label_leak.csv"
    return load_dataset(path).frame


@pytest.fixture(scope="module")
def leak_report(leak_frame):
    return check_leakage(leak_frame, "Label")


def _findings(report, column):
    return {(f.action, f.reason) for f in report.findings if f.column == column}


def test_renamed_label_copy_blocked(leak_report):
    assert ("block", "LABEL_RELABELLING") in _findings(leak_report, "xyz_col")
    assert "xyz_col" in leak_report.blocked
    assert "xyz_col" not in leak_report.features


def test_encoded_and_prefixed_label_blocked(leak_report):
    for column in ("enc_col", "pre_col"):
        assert ("block", "LABEL_RELABELLING") in _findings(leak_report, column), column
        assert column not in leak_report.features


def test_near_perfect_continuous_feature_warns_only(leak_report):
    finding = next(f for f in leak_report.findings if f.column == "f_strong")
    assert (finding.action, finding.reason) == ("warn", "STRONG_SINGLE_FEATURE")
    assert finding.score >= LeakageConfig().warn_score
    assert finding.values_per_class > LeakageConfig().relabel_max_values_per_class
    assert "f_strong" in leak_report.features
    assert "f_strong" in leak_report.warned


def test_ordinary_features_pass(leak_report):
    assert _findings(leak_report, "f_noise") == set()
    assert {"f_noise", "Idle Mean", "f_strong"} <= set(leak_report.features)
    assert "Label" not in leak_report.features, "the label is never a feature"
    assert all(f.column != "Label" for f in leak_report.findings)


def test_override_recorded(leak_frame):
    report = check_leakage(leak_frame, "Label",
                           allow={"xyz_col": "checked: synthetic copy kept on purpose"})
    assert "xyz_col" in report.features
    assert "xyz_col" not in report.blocked
    override = next(o for o in report.overrides if o.column == "xyz_col")
    assert (override.action, override.reason) == ("block", "LABEL_RELABELLING")
    assert override.score >= 0.99
    assert override.note == "checked: synthetic copy kept on purpose"


def test_override_needs_a_reason(leak_frame):
    with pytest.raises(ValueError, match="reason"):
        check_leakage(leak_frame, "Label", allow={"xyz_col": "  "})


def test_override_of_unknown_column_rejected(leak_frame):
    with pytest.raises(ValueError, match="nope"):
        check_leakage(leak_frame, "Label", allow={"nope": "typo"})


# --- name patterns (whole words only) ------------------------------------------------

@pytest.mark.parametrize("name", [
    "Flow ID", "FlowID", "flow_id", "Src IP", "Destination IP", "srcIP", "Timestamp",
    "Unnamed: 0", "attack_cat", "level", "Attempted Category", "row", "index", "id",
    "Label2", "class",
])
def test_name_pattern_hits(name):
    assert name_pattern(name) is not None, name


@pytest.mark.parametrize("name", [
    "Idle Mean", "Idle Max", "Idle Std", "Bwd IAT Min", "Init_Win_bytes_forward",
    "Destination Port", "Fwd Header Length", "Subflow Bwd Bytes", "num_root", "classification",
    "dst_host_srv_diff_host_rate", "is_host_login", "Packet Length Variance", "rowing",
    "su_attempted",  # NSL-KDD feature, not the CIC "Attempted Category"
])
def test_name_pattern_misses(name):
    assert name_pattern(name) is None, name


def test_idle_mean_not_flagged(leak_report):
    assert _findings(leak_report, "Idle Mean") == set()


def test_unnamed_index_flagged():
    frame = pd.DataFrame({"Unnamed: 0": np.arange(60) * 3 % 7, "x": np.arange(60) % 5,
                          "Label": ["a", "b", "c"] * 20})
    report = check_leakage(frame, "Label")
    assert ("block", "NAME_PATTERN") in _findings(report, "Unnamed: 0")


# --- ID-like columns (integer / text only) -------------------------------------------

def test_integer_row_counter_flagged():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"counter": np.arange(100), "x": rng.normal(size=100),
                          "Label": rng.choice(["a", "b"], size=100)})
    report = check_leakage(frame, "Label")
    assert ("block", "ID_LIKE") in _findings(report, "counter")


def test_unique_text_column_flagged():
    frame = pd.DataFrame({"session": [f"s{i:05d}" for i in range(2000)],
                          "Label": ["a", "b"] * 1000})
    frame = frame.sample(frac=1, random_state=1)  # not in order: uniqueness alone flags it
    assert ("block", "ID_LIKE") in _findings(check_leakage(frame, "Label"), "session")


def test_unique_continuous_feature_not_flagged():
    rng = np.random.default_rng(1)
    durations = np.sort(rng.exponential(1000.0, size=500))  # unique AND increasing
    frame = pd.DataFrame({"Flow Bytes/s": durations, "Label": rng.choice(["a", "b"], 500)})
    assert _findings(check_leakage(frame, "Label"), "Flow Bytes/s") == set()


def test_uniqueness_needs_enough_rows():
    """On a few hundred rows most integer features look unique; only order counts there."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"Flow IAT Max": rng.permutation(300) * 17,
                          "Label": rng.choice(["a", "b"], 300)})
    assert all(f.reason != "ID_LIKE" for f in check_leakage(frame, "Label").findings)
    frame = pd.DataFrame({"Flow IAT Max": rng.permutation(5000) * 17,
                          "Label": rng.choice(["a", "b"], 5000)})
    assert ("block", "ID_LIKE") in _findings(check_leakage(frame, "Label"), "Flow IAT Max")


# --- profile, sampling, determinism --------------------------------------------------

def test_profile_blocked_columns_blocked(leak_frame):
    report = check_leakage(leak_frame, "Label", profile_blocked=("f_noise",))
    assert ("block", "PROFILE_BLOCKED") in _findings(report, "f_noise")
    assert "f_noise" not in report.features


def test_many_class_label_copy_blocked():
    """A depth-3 tree has 8 leaves; a copy of a 20-class label must still be caught."""
    labels = np.array([f"class_{i % 20}" for i in range(2000)])
    codes = np.array([i % 20 for i in range(2000)])
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"code": rng.permutation(20)[codes], "Label": labels})
    assert ("block", "LABEL_RELABELLING") in _findings(check_leakage(frame, "Label"), "code")


def test_runs_on_stratified_sample():
    rng = np.random.default_rng(3)
    labels = rng.choice(["a", "b", "c"], size=20_000, p=[0.7, 0.2, 0.1])
    frame = pd.DataFrame({"x": rng.normal(size=20_000), "Label": labels})
    report = check_leakage(frame, "Label", config=LeakageConfig(sample_rows=1_000))
    assert report.sample_rows == 1_000
    assert report.rows == 20_000


def test_rare_classes_excluded_and_recorded():
    frame = pd.DataFrame({"x": list(range(62)), "Label": ["a"] * 30 + ["b"] * 30 + ["c"] * 2})
    report = check_leakage(frame, "Label")
    assert report.excluded_classes == ("c",)


def test_deterministic(leak_frame):
    assert check_leakage(leak_frame, "Label") == check_leakage(leak_frame, "Label")


def test_missing_and_infinite_values_tolerated():
    rng = np.random.default_rng(4)
    x = rng.normal(size=300)
    x[:10] = np.inf
    x[10:20] = np.nan
    frame = pd.DataFrame({"x": x, "Label": rng.choice(["a", "b"], 300)})
    assert check_leakage(frame, "Label").features == ("x",)


def test_real_cic_ddos_sample_has_no_blocks(fixtures_dir):
    """Easy binary data: strong packet-size features may warn but must not be blocked."""
    loaded = load_dataset(fixtures_dir / "cic_ddos_sample.csv")
    report = check_leakage(loaded.frame, "Label",
                           profile_blocked=loaded.profile.blocked_columns)
    assert report.blocked == ()
    assert len(report.features) == loaded.frame.shape[1] - 1


def test_real_nslkdd_level_blocked(fixtures_dir):
    loaded = load_dataset(fixtures_dir / "nslkdd_train_sample.txt")
    report = check_leakage(loaded.frame, "attack",
                           profile_blocked=loaded.profile.blocked_columns)
    assert "level" in report.blocked
    reasons = _findings(report, "level")
    assert ("block", "PROFILE_BLOCKED") in reasons and ("block", "NAME_PATTERN") in reasons


def test_sample_keeps_every_class_for_cv():
    """A class with 4 of 100,000 rows must still have >= cv_folds rows in the sample."""
    rng = np.random.default_rng(6)
    labels = np.array(["big"] * 99_996 + ["tiny"] * 4)
    frame = pd.DataFrame({"x": rng.normal(size=100_000), "Label": labels})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # sklearn warns when a class has fewer rows than folds
        report = check_leakage(frame, "Label", config=LeakageConfig(sample_rows=1_000))
    assert report.sample_rows == 1_000
    assert report.excluded_classes == ()


def test_real_nslkdd_su_attempted_kept(fixtures_dir):
    loaded = load_dataset(fixtures_dir / "nslkdd_train_sample.txt")
    report = check_leakage(loaded.frame, "attack", profile_blocked=loaded.profile.blocked_columns)
    assert "su_attempted" in report.features
