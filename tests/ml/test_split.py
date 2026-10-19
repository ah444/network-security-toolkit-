"""Step 0.4b: dedupe + profile-defined split with time validation and reason codes [L4, L5, D9]."""

import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from toolkit.config import SplitConfig
from toolkit.ids_core.datasets import load_dataset
from toolkit.ids_core.split import dedupe, split_dataset

TEMPORAL = ("temporal", "temporal_per_class")


def _stamp(moment: datetime) -> str:
    """CIC "TrafficLabelling" style: day first, no zero padding, minute precision."""
    return f"{moment.day}/{moment.month}/{moment.year} {moment.hour}:{moment.minute:02d}"


def _timed_frame(seed=0, attack_days=None, rows_per_class=80):
    """Rows in time order; BENIGN on every day, each attack on one day (as in CIC-IDS2017)."""
    attack_days = attack_days or {"DoS": [5], "PortScan": [7]}
    rng = np.random.default_rng(seed)
    records = []
    for label, days in {"BENIGN": [3, 4, 5, 6, 7], **attack_days}.items():
        for _ in range(rows_per_class):
            day = int(rng.choice(days))
            moment = datetime(2017, 7, day, 8) + timedelta(minutes=int(rng.integers(0, 600)))
            records.append((moment, label))
    records.sort()
    return pd.DataFrame({
        "x": rng.normal(size=len(records)),
        "Timestamp": [_stamp(moment) for moment, _ in records],
        "Label": [label for _, label in records],
    })


def _times(series):
    return pd.to_datetime(series, format="%d/%m/%Y %H:%M")


def _rows(frame, columns):
    return set(map(tuple, frame[columns].astype(str).to_numpy()))


# --- dedupe ------------------------------------------------------------------------

def test_no_duplicate_across_train_test():
    rng = np.random.default_rng(1)
    unique = pd.DataFrame({"a": rng.integers(0, 1000, 500), "b": rng.normal(size=500),
                           "Flow ID": [f"f{i}" for i in range(500)],
                           "Label": rng.choice(["BENIGN", "DoS", "PortScan"], 500)})
    copies = unique.sample(200, replace=True, random_state=2)
    renamed = unique.sample(50, random_state=3).assign(**{"Flow ID": lambda f: f["Flow ID"] + "x"})
    frame = pd.concat([unique, copies, renamed], ignore_index=True)

    result = split_dataset(frame, "Label", strategy="stratified")

    columns = ["a", "b", "Label"]
    assert _rows(result.train, columns).isdisjoint(_rows(result.test, columns))
    assert result.dedupe.rows_before == 750
    assert result.dedupe.rows_after == 500
    assert result.dedupe.removed == 250
    assert sum(n for _, n in result.dedupe.removed_per_class) == 250
    assert "Flow ID" in result.dedupe.ignored_columns, "ID columns don't make a copy unique"
    assert len(result.train) + len(result.test) == 500


def test_dedupe_keeps_first_occurrence_and_reports_conflicts():
    frame = pd.DataFrame({"x": [1, 2, 1, 3, 3], "Label": ["a", "b", "a", "a", "b"]},
                         index=[10, 11, 12, 13, 14])
    kept, stats = dedupe(frame, "Label")
    assert list(kept.index) == [10, 11, 13, 14]
    assert stats.removed == 1
    assert stats.removed_per_class == (("a", 1),)
    assert stats.conflicting_rows == 2, "same features, different labels: counted, not removed"


def test_dedupe_does_not_modify_input():
    frame = pd.DataFrame({"x": [1, 1, 2, 2], "Label": ["a", "a", "b", "b"]})
    before = frame.copy()
    split_dataset(frame, "Label", strategy="stratified")
    pd.testing.assert_frame_equal(frame, before)


# --- official ----------------------------------------------------------------------

def test_nslkdd_uses_official_split(fixtures_dir):
    train = load_dataset(fixtures_dir / "nslkdd_train_sample.txt")
    test = load_dataset(fixtures_dir / "nslkdd_test_sample_noheader.txt")
    assert train.profile.split_strategy == "official"

    result = split_dataset(train.frame, "attack", profile=train.profile, test_frame=test.frame)

    assert (result.strategy, result.requested, result.reason_codes) == ("official", "official", ())
    assert result.official_mode == "as_published"
    assert len(result.train) == len(train.frame)
    assert len(result.test) == len(test.frame)
    assert "CLASS_MISSING_FROM_TRAIN" in result.warnings, "KDDTest+ has attack types KDDTrain+ lacks"
    unseen = {c.label for c in result.coverage if c.train == 0}
    assert {"apache2", "mailbomb"} <= unseen


def _overlapping_official_files():
    train = pd.DataFrame({"x": [1, 2, 3, 4, 4], "Label": ["a", "b", "a", "b", "b"]})
    test = pd.DataFrame({"x": [2, 5, 5, 6, 1], "Label": ["b", "a", "a", "b", "a"]})
    return train, test


def test_official_files_kept_as_published_by_default():
    """Benchmark mode: both files exactly as published; overlaps and copies only reported."""
    train, test = _overlapping_official_files()
    result = split_dataset(train, "Label", strategy="official", test_frame=test)
    assert result.official_mode == "as_published"
    pd.testing.assert_frame_equal(result.train, train)
    pd.testing.assert_frame_equal(result.test, test)
    d = result.dedupe
    assert (d.applied, d.copies_found, d.removed, d.rows_after) == (False, 1, 0, 5)
    assert (d.test_rows_before, d.test_copies, d.test_overlap, d.test_removed) == (5, 1, 2, 0)
    assert {"OFFICIAL_TEST_OVERLAP_KEPT", "OFFICIAL_COPIES_KEPT"} <= set(result.warnings)
    assert "OFFICIAL_TEST_OVERLAP_REMOVED" not in result.warnings


def test_official_strict_mode_removes_copies_and_overlap():
    train, test = _overlapping_official_files()
    result = split_dataset(train, "Label", strategy="official", test_frame=test,
                           config=SplitConfig(official_dedupe=True))
    assert result.official_mode == "deduplicated"
    assert list(result.train["x"]) == [1, 2, 3, 4]
    assert list(result.test["x"]) == [5, 6]
    d = result.dedupe
    assert (d.applied, d.copies_found, d.removed) == (True, 1, 1)
    assert (d.test_copies, d.test_overlap, d.test_removed) == (1, 2, 3)
    assert "OFFICIAL_TEST_OVERLAP_REMOVED" in result.warnings


def test_official_mode_only_for_official_split():
    frame = pd.DataFrame({"x": [1, 1, 2, 3] * 10, "Label": ["a", "a", "b", "b"] * 10})
    result = split_dataset(frame, "Label", strategy="stratified")
    assert result.official_mode is None
    assert result.dedupe.applied and result.dedupe.removed == 37, "other splits always dedupe"


def test_official_without_test_file_falls_back():
    frame = pd.DataFrame({"x": range(40), "Label": ["a", "b"] * 20})
    result = split_dataset(frame, "Label", strategy="official")
    assert result.strategy == "stratified"
    assert result.reason_codes == ("OFFICIAL_TEST_FILE_MISSING",)


def test_official_test_file_needs_same_columns():
    train = pd.DataFrame({"x": [1, 2], "y": [1, 2], "Label": ["a", "b"]})
    test = pd.DataFrame({"x": [3], "Label": ["a"]})
    with pytest.raises(ValueError, match="y"):
        split_dataset(train, "Label", strategy="official", test_frame=test)


# --- temporal ----------------------------------------------------------------------

def test_temporal_per_class_ordering():
    result = split_dataset(_timed_frame(), "Label", strategy="temporal_per_class",
                           time_column="Timestamp")
    assert result.strategy == "temporal_per_class"
    assert result.reason_codes == ()
    for label in ("BENIGN", "DoS", "PortScan"):
        train = _times(result.train.loc[result.train["Label"] == label, "Timestamp"])
        test = _times(result.test.loc[result.test["Label"] == label, "Timestamp"])
        assert len(train) and len(test), label
        assert train.max() < test.min(), f"{label}: a test row is not later than every train row"
    assert all(c.train and c.test for c in result.coverage)
    assert result.time_check.passed


def test_temporal_global_cut_falls_back_when_a_class_has_one_day():
    """DoS only happens on Wednesday, so a global cut late on Friday leaves it out of test."""
    result = split_dataset(_timed_frame(), "Label", strategy="temporal", time_column="Timestamp")
    assert result.reason_codes == ("CLASS_MISSING_FROM_TEST",)
    assert result.strategy == "temporal_per_class"


def test_temporal_global_cut_when_classes_spread():
    every_day = [3, 4, 5, 6, 7]
    frame = _timed_frame(attack_days={"DoS": every_day, "PortScan": every_day})
    result = split_dataset(frame, "Label", strategy="temporal", time_column="Timestamp")
    assert (result.strategy, result.reason_codes) == ("temporal", ())
    assert _times(result.train["Timestamp"]).max() < _times(result.test["Timestamp"]).min()


def _broken(kind):
    frame = _timed_frame()
    sources = None
    if kind == "unparseable":
        frame.loc[frame.index[::20], "Timestamp"] = "not a time"
    elif kind == "missing":
        frame.loc[frame.index[::20], "Timestamp"] = np.nan
    elif kind == "ambiguous":  # 1 Feb vs 2 Jan: both readings equally (dis)ordered
        frame["Timestamp"] = (["1/2/2017 10:00", "2/1/2017 10:00",
                               "2/1/2017 10:00", "1/2/2017 10:00"] * len(frame))[:len(frame)]
    elif kind == "reset":  # two capture files whose clocks overlap
        half = len(frame) // 2
        sources = pd.Series(["monday.csv"] * half + ["tuesday.csv"] * (len(frame) - half),
                            index=frame.index)
        frame["Timestamp"] = list(frame["Timestamp"][:half]) * 2 + list(
            frame["Timestamp"][: len(frame) - 2 * half])
    return frame, sources


@pytest.mark.parametrize(("kind", "code"), [
    ("unparseable", "TIME_PARSE_FAILURE_RATIO_EXCEEDED"),
    ("missing", "TIME_MISSING_RATIO_EXCEEDED"),
    ("ambiguous", "TIME_FORMAT_AMBIGUOUS"),
    ("reset", "TIME_NON_MONOTONIC_ACROSS_FILES"),
])
def test_broken_time_column_falls_back_with_reason_code(kind, code):
    frame, sources = _broken(kind)
    result = split_dataset(frame, "Label", strategy="temporal_per_class",
                           time_column="Timestamp", sources=sources)
    assert result.strategy not in TEMPORAL
    assert result.reason_codes[0] == code
    assert result.time_check.reason == code
    assert not result.time_check.passed


def test_missing_time_column_falls_back_with_reason_code():
    frame = _timed_frame().drop(columns="Timestamp")
    result = split_dataset(frame, "Label", strategy="temporal_per_class", time_column="Timestamp")
    assert result.strategy == "stratified"
    assert result.reason_codes == ("TIME_COLUMN_MISSING", "GROUP_COLUMN_MISSING")
    assert "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC" in result.warnings


def test_a_few_missing_times_are_dropped_and_recorded():
    frame = _timed_frame()
    frame.loc[frame.index[5], "Timestamp"] = np.nan  # 1 of 240 rows < 1 %
    result = split_dataset(frame, "Label", strategy="temporal_per_class", time_column="Timestamp")
    assert result.strategy == "temporal_per_class"
    assert result.time_check.rows_dropped == 1
    assert "TIME_MISSING_ROWS_DROPPED" in result.warnings
    assert len(result.train) + len(result.test) == len(frame) - 1


def test_day_first_resolved_from_values():
    frame = _timed_frame()
    frame.loc[frame.index[-1], "Timestamp"] = "13/7/2017 9:00"  # only valid day-first
    result = split_dataset(frame, "Label", strategy="temporal_per_class", time_column="Timestamp")
    assert result.time_check.format == "day-first"
    assert result.strategy == "temporal_per_class"


def test_month_first_resolved_from_file_order():
    """Both readings parse; only month-first keeps the rows in time order."""
    stamps = [f"{first}/{second}/2017 10:00" for first in range(1, 13) for second in range(1, 6)]
    frame = pd.DataFrame({"x": range(60), "Timestamp": stamps, "Label": ["a", "b"] * 30})
    result = split_dataset(frame, "Label", strategy="temporal", time_column="Timestamp")
    assert result.time_check.format == "month-first"
    assert result.strategy == "temporal"


def test_iso_times_with_mixed_precision_parse():
    frame = _timed_frame()
    frame["Timestamp"] = _times(frame["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    frame.loc[frame.index[::2], "Timestamp"] += ".598966"
    result = split_dataset(frame, "Label", strategy="temporal_per_class", time_column="Timestamp")
    assert (result.strategy, result.time_check.format) == ("temporal_per_class", "ISO8601")


def test_class_with_a_single_timestamp_is_too_coarse():
    frame = _timed_frame()
    frame.loc[frame["Label"] == "DoS", "Timestamp"] = "5/7/2017 10:00"
    result = split_dataset(frame, "Label", strategy="temporal_per_class", time_column="Timestamp")
    assert result.strategy not in TEMPORAL
    assert result.reason_codes[0] == "TIME_TOO_COARSE_FOR_CLASS"


# --- group -------------------------------------------------------------------------

def test_group_never_spans_split():
    rng = np.random.default_rng(4)
    hosts = np.repeat(np.arange(30), 20)
    frame = pd.DataFrame({"x": rng.normal(size=600), "host": hosts,
                          "Label": np.array(["a", "b", "c"])[hosts % 3]})
    result = split_dataset(frame, "Label", strategy="group", group_column="host")
    assert (result.strategy, result.reason_codes) == ("group", ())
    assert set(result.train["host"]).isdisjoint(result.test["host"])
    assert all(c.train and c.test for c in result.coverage)


def test_group_fallback_skipped_when_class_one_sided():
    """Grouping by capture file: each attack lives in one file, so it can't be on both sides."""
    rng = np.random.default_rng(5)
    files = np.repeat([f"day{i}.csv" for i in range(10)], 30)
    labels = np.where(files == "day8.csv", "DoS", np.where(files == "day9.csv", "PortScan", "BENIGN"))
    stamps = [_stamp(datetime(2017, 7, 3, 8) + timedelta(minutes=i)) for i in range(300)]
    frame = pd.DataFrame({"x": rng.normal(size=300), "Timestamp": stamps, "Label": labels})
    frame.loc[frame.index[::25], "Timestamp"] = "garbage"
    result = split_dataset(frame, "Label", strategy="temporal_per_class", time_column="Timestamp",
                           sources=pd.Series(files, index=frame.index))
    assert result.reason_codes == ("TIME_PARSE_FAILURE_RATIO_EXCEEDED", "GROUP_CLASS_ONE_SIDED")
    assert result.strategy == "stratified"


def test_group_needs_enough_groups():
    frame = pd.DataFrame({"x": range(40), "g": [0, 1] * 20, "Label": ["a", "b"] * 20})
    result = split_dataset(frame, "Label", strategy="group", group_column="g")
    assert (result.strategy, result.reason_codes) == ("stratified", ("GROUP_TOO_FEW_GROUPS",))


# --- stratified, coverage, profile, determinism ------------------------------------

def test_cic_without_time_column_uses_stratified_with_warning(fixtures_dir):
    loaded = load_dataset(fixtures_dir / "cic_ddos_sample.csv")
    assert "CIC_NO_TIME_COLUMN" in loaded.profile.warnings
    result = split_dataset(loaded.frame, "Label", profile=loaded.profile)
    assert (result.strategy, result.requested, result.reason_codes) == ("stratified",) * 2 + ((),)
    assert "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC" in result.warnings
    assert result.time_column is None
    assert all(c.train and c.test for c in result.coverage)


def test_stratified_keeps_class_shares():
    frame = pd.DataFrame({"x": range(1000), "Label": ["a"] * 700 + ["b"] * 200 + ["c"] * 100})
    result = split_dataset(frame, "Label", strategy="stratified")
    counts = result.test["Label"].value_counts().to_dict()
    assert counts == {"a": 140, "b": 40, "c": 20}
    assert result.test_size == SplitConfig().test_size


def test_missing_class_in_test_reports_CLASS_MISSING_FROM_TEST():
    frame = pd.DataFrame({"x": range(41), "Label": ["a", "b"] * 20 + ["rare"]})
    result = split_dataset(frame, "Label", strategy="stratified")
    assert "CLASS_MISSING_FROM_TEST" in result.warnings
    rare = next(c for c in result.coverage if c.label == "rare")
    assert (rare.train, rare.test) == (1, 0)


def test_profile_strategy_used_and_override_recorded(fixtures_dir):
    loaded = load_dataset(fixtures_dir / "cic_ddos_sample.csv")
    timed = dataclasses.replace(loaded.profile, split_strategy="temporal_per_class",
                                time_column="Timestamp")
    frame = _timed_frame()
    by_profile = split_dataset(frame, "Label", profile=timed)
    assert (by_profile.strategy, by_profile.overridden) == ("temporal_per_class", False)
    overridden = split_dataset(frame, "Label", profile=timed, strategy="stratified")
    assert (overridden.strategy, overridden.requested) == ("stratified", "stratified")
    assert (overridden.profile_strategy, overridden.overridden) == ("temporal_per_class", True)


def test_deterministic_and_seed_recorded():
    frame = pd.DataFrame({"x": range(300), "Label": ["a", "b", "c"] * 100})
    first = split_dataset(frame, "Label", strategy="stratified")
    again = split_dataset(frame, "Label", strategy="stratified")
    other = split_dataset(frame, "Label", strategy="stratified",
                          config=SplitConfig(random_state=7))
    assert list(first.test.index) == list(again.test.index)
    assert list(first.test.index) != list(other.test.index)
    assert (first.seed, other.seed) == (42, 7)


def test_unknown_strategy_rejected():
    frame = pd.DataFrame({"x": range(10), "Label": ["a", "b"] * 5})
    with pytest.raises(ValueError, match="random"):
        split_dataset(frame, "Label", strategy="random")


def test_unknown_label_rejected():
    frame = pd.DataFrame({"x": range(10), "Label": ["a", "b"] * 5})
    with pytest.raises(ValueError, match="nope"):
        split_dataset(frame, "nope", strategy="stratified")
