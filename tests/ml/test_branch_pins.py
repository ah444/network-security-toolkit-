"""v5.2 step 2.3a: branches of the functions step 2.3 refactors that no other test reached
(measured with ``sys.monitoring`` over the whole suite), pinned on the code before the
refactor: ``split_dataset``, ``check_time_column``, ``load_dataset``, ``check_leakage`` and
``run_json.jsonable`` (``runs._jsonable`` before step 2.3h). They describe today's
behaviour; the refactor must keep it.
"""

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from toolkit.core.data.ids.dataset_files import SAMPLE_BYTES, DatasetError
from toolkit.core.data.settings import SplitConfig
from toolkit.core.services.ids import split as split_module
from toolkit.core.services.ids.datasets import load_dataset
from toolkit.core.services.ids.leakage import check_leakage
from toolkit.core.services.ids.run_json import jsonable as _jsonable
from toolkit.core.services.ids.split import SOURCE_GROUP, TimeCheck, check_time_column, split_dataset


def _frame(rows=300, seed=0, classes=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=rows), "y": rng.integers(0, 50, rows),
                         "Label": rng.choice(classes, rows)})


def _iso_times(rows):
    return pd.date_range("2017-07-03 08:00", periods=rows, freq="min").strftime("%Y-%m-%d %H:%M:%S")


def _profile(**fields):
    """The fields of a ProfileMatch that split_dataset reads."""
    return SimpleNamespace(**{"split_strategy": "stratified", "time_column": None,
                              "group_column": None, "blocked_columns": (), **fields})


# --- split_dataset -----------------------------------------------------------------

def test_an_explicit_time_column_wins_over_the_profiles():
    frame = _frame().assign(When=_iso_times(300))
    result = split_dataset(frame, "Label", time_column="When",
                           profile=_profile(split_strategy="temporal", time_column="Nowhere"))
    assert (result.strategy, result.time_column, result.reason_codes) == ("temporal", "When", ())


def test_an_explicit_group_column_wins_over_the_profiles():
    frame = _frame().assign(Host=[f"h{i % 10}" for i in range(300)])
    result = split_dataset(frame, "Label", group_column="Host",
                           profile=_profile(split_strategy="group", group_column="Nowhere"))
    assert (result.strategy, result.group_column, result.reason_codes) == ("group", "Host", ())


def test_rows_without_a_label_are_dropped_first():
    frame = _frame()
    frame.loc[[3, 10, 50], "Label"] = None
    result = split_dataset(frame, "Label", strategy="stratified")
    assert result.rows_without_label == 3
    assert result.warnings == ("LABEL_MISSING_ROWS_DROPPED", "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC")
    assert not {3, 10, 50} & (set(result.train.index) | set(result.test.index))
    assert len(result.train) + len(result.test) == 297


def test_warnings_keep_the_order_they_were_found_in():
    """Not sorted: the stratified split's warning comes before the coverage ones."""
    frame = _frame()
    frame.loc[0, "Label"] = "Lonely"  # one row: it goes to train
    result = split_dataset(frame, "Label", strategy="stratified")
    assert result.warnings == ("STRATIFIED_SCORES_MAY_BE_OPTIMISTIC", "CLASS_MISSING_FROM_TEST")


def test_the_label_is_never_an_ignored_column():
    result = split_dataset(_frame(), "Label", strategy="stratified", ignore=["Label", "", "y"])
    assert result.dedupe.ignored_columns == ("y",)


def test_the_source_files_are_the_group_when_the_group_column_is_absent():
    sources = [f"day{i % 10}.csv" for i in range(300)]
    result = split_dataset(_frame(), "Label", strategy="group", group_column="Nowhere",
                           sources=sources)
    assert (result.strategy, result.group_column) == ("group", SOURCE_GROUP)


def test_a_reason_code_is_recorded_once():
    result = split_dataset(_frame(), "Label", strategy="temporal")
    assert result.strategy == "stratified"
    assert result.reason_codes == ("TIME_COLUMN_MISSING", "GROUP_COLUMN_MISSING")


def test_the_time_column_is_checked_once(monkeypatch):
    """temporal fails (a class only at the start), temporal_per_class reuses the same check."""
    calls = []
    real = split_module.check_time_column

    def counted(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(split_module, "check_time_column", counted)
    frame = _frame(classes=("A", "B")).assign(When=_iso_times(300))
    frame.loc[:59, "Label"] = "Z"
    result = split_dataset(frame, "Label", strategy="temporal", time_column="When")
    assert (result.strategy, result.reason_codes) == ("temporal_per_class",
                                                      ("CLASS_MISSING_FROM_TEST",))
    assert calls == ["When"]


# --- check_time_column -------------------------------------------------------------

def test_no_time_column():
    frame = _frame()
    assert check_time_column(frame, None, SplitConfig()) == (
        TimeCheck("", False, "TIME_COLUMN_MISSING", None, 1.0, 0.0, 300), None)


def test_a_categorical_time_column_is_read_as_text():
    frame = _frame().assign(When=pd.Categorical(_iso_times(300)))
    check, times = check_time_column(frame, "When", SplitConfig())
    assert (check.passed, check.format) == (True, "ISO8601")
    assert times.iloc[0] == pd.Timestamp("2017-07-03 08:00")


def test_a_datetime_column():
    frame = _frame().assign(When=pd.to_datetime(_iso_times(300)))
    check, times = check_time_column(frame, "When", SplitConfig())
    assert (check.passed, check.format, check.rows_dropped) == (True, "datetime", 0)


def test_a_numeric_time_column_with_an_infinite_value():
    values = np.arange(300, dtype=float)
    values[7] = np.inf
    check, times = check_time_column(_frame().assign(When=values), "When", SplitConfig())
    assert (check.passed, check.format, check.rows_dropped) == (True, "number", 1)
    assert np.isnan(times.iloc[7])


def test_day_and_month_that_cannot_be_told_apart():
    frame = pd.DataFrame({"When": ["01/02/2017 10:00", "02/01/2017 10:00", "01/02/2017 10:00"],
                          "Label": ["A", "B", "A"]})
    check, times = check_time_column(frame, "When", SplitConfig())
    assert (check.passed, check.reason, times) == (False, "TIME_FORMAT_AMBIGUOUS", None)


def test_capture_files_one_after_the_other_pass_and_overlapping_ones_fail():
    frame = _frame().assign(When=_iso_times(300))
    in_order = pd.Series(["mon.csv"] * 150 + ["tue.csv"] * 150, index=frame.index)
    assert check_time_column(frame, "When", SplitConfig(), in_order)[0].passed
    mixed = pd.Series(["mon.csv", "tue.csv"] * 150, index=frame.index)
    check, _ = check_time_column(frame, "When", SplitConfig(), mixed)
    assert check.reason == "TIME_NON_MONOTONIC_ACROSS_FILES"


def test_an_empty_frame_has_no_usable_time():
    frame = _frame().assign(When=_iso_times(300)).iloc[:0]
    check, times = check_time_column(frame, "When", SplitConfig())
    assert (check.reason, check.missing_ratio, times) == ("TIME_MISSING_RATIO_EXCEEDED", 1.0, None)


# --- load_dataset ------------------------------------------------------------------

def _rows_past_the_sample():
    """ASCII rows filling more than the bytes the encoding is guessed from."""
    row = b"1.5,2,BENIGN\n"
    return row * (SAMPLE_BYTES // len(row) + 10)


def test_invalid_utf8_after_the_sample_falls_back_to_cp1252(tmp_path):
    path = tmp_path / "late.csv"
    path.write_bytes(b"a,b,Label\n" + _rows_past_the_sample() + b"2.5,3,caf\xe9\n")
    loaded = load_dataset(path, chunk_rows=1000)
    assert loaded.report.encoding == "cp1252"
    assert loaded.report.warnings == ("ENCODING_FALLBACK_CP1252",)
    assert loaded.frame["Label"].iloc[-1] == "café"


def test_a_byte_cp1252_cannot_read_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a,b,Label\n1,2,caf\xe9\n" + b"3,4,x\x81y\n")
    with pytest.raises(DatasetError, match="Could not read the file as a table"):
        load_dataset(path)


def test_a_row_with_too_many_fields_is_refused(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text("a,b,Label\n" + "1,2,A\n" * 5 + "1,2,3,4,5\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Could not read the file as a table"):
        load_dataset(path)


def test_suffixed_columns_dropped_only_when_they_are_copies(tmp_path):
    path = tmp_path / "suffixed.csv"
    path.write_text("x,x.1,y,y.1,Label\n" + "".join(f"{i},{i},{i},{i + 1},A\n"
                                                     for i in range(20)), encoding="utf-8")
    loaded = load_dataset(path)
    assert loaded.report.dropped_duplicate_columns == ("x.1",)
    assert loaded.report.warnings == ("SUFFIXED_COLUMN_DIFFERS",)
    assert list(loaded.frame.columns) == ["x", "y", "y.1", "Label"]


def test_text_becomes_a_category_up_to_half_distinct_values(tmp_path):
    """CATEGORY_MAX_SHARE = 0.5: 45 % distinct values is a category, 55 % stays text."""
    rows = [f"{i}.5,v{i % 45},w{i % 55},A\n" for i in range(100)]
    path = tmp_path / "shares.csv"
    path.write_text("a,some,most,Label\n" + "".join(rows), encoding="utf-8")
    loaded = load_dataset(path)
    assert str(loaded.frame["some"].dtype) == "category"
    assert loaded.frame["most"].dtype == object
    assert loaded.report.downcast == {"a": "float32", "some": "category", "Label": "category"}


def test_a_numeric_label_is_neither_normalised_nor_a_category(tmp_path):
    path = tmp_path / "numbers.csv"
    path.write_text("a,Label\n" + "".join(f"{i}.5,{i % 2}\n" for i in range(20)), encoding="utf-8")
    loaded = load_dataset(path)
    assert loaded.report.label_values_normalised == 0
    assert str(loaded.frame["Label"].dtype) == "int8"
    assert loaded.report.downcast == {"Label": "int8", "a": "float32"}


# --- check_leakage -----------------------------------------------------------------

def test_one_class_gets_no_score_findings():
    train = _frame(classes=("A",)).assign(copy=lambda f: f["x"])
    report = check_leakage(train, "Label")
    assert report.findings == ()
    assert report.features == ("x", "y", "copy")


# --- run_json.jsonable -------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Pair:
    a: int
    b: float

    @property
    def total(self):
        return self.a + self.b


def test_the_json_form_of_each_type():
    when = datetime(2026, 1, 2, tzinfo=timezone.utc)
    value = {"none": None, "bool": True, "int": 3, "text": "s", "float": 1.5,
             "nan": float("nan"), "inf": float("-inf"), "np_int": np.int64(4),
             "np_float": np.float32(0.5), "array": np.array([1, 2]), "pair": _Pair(1, 2.0),
             "pair_class": _Pair, "tuple": (1, (2,)), 7: "key", "set": {3, 1, 2},
             "frozen": frozenset({"b", "a"}), "when": when, "stamp": pd.Timestamp("2026-01-02"),
             "type": int}
    assert _jsonable(value) == {
        "none": None, "bool": True, "int": 3, "text": "s", "float": 1.5, "nan": None,
        "inf": None, "np_int": 4, "np_float": 0.5, "array": [1, 2],
        "pair": {"a": 1, "b": 2.0, "total": 3.0}, "pair_class": "_Pair", "tuple": [1, [2]],
        "7": "key", "set": [1, 2, 3], "frozen": ["a", "b"], "when": "2026-01-02T00:00:00+00:00",
        "stamp": "2026-01-02T00:00:00", "type": "int"}
    assert _jsonable(object()).startswith("<object object at ")
