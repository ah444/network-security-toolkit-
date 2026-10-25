"""The time column of a temporal split, checked and parsed (plan D9, L5; moved out of
``split.py`` in v5.2 step 2.3f).

Time validation: the empty share and the unparseable share stay under the limits in
``SplitConfig``; ``d/m/Y`` vs ``m/d/Y`` is resolved from the values (one reading fails)
or from the file order (one reading goes backwards less), and is harmless when both
readings order the rows the same way; capture files must not overlap in time.

A failed check names its reason with one of ``split.REASON_CODES``; ``split_dataset`` then
falls back to the next strategy.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from ...data.settings import SplitConfig

_SLASH_DATE = re.compile(
    r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})"
    r"(?:[ T]+(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?\s*([AaPp][Mm])?)?\s*$"
)
_ISO_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class TimeCheck:
    column: str
    passed: bool
    reason: str | None  # one of split.REASON_CODES when not passed
    format: str | None  # ISO8601 | day-first | month-first | same-order | inferred | number | datetime
    missing_ratio: float
    parse_failure_ratio: float
    rows_dropped: int  # rows without a usable time, left out of a temporal split


class _FormatAmbiguous(Exception):
    """Both readings of a d/m/Y column fit the values and the files' order equally."""


def _from_parts(parts: pd.DataFrame, day: int, month: int) -> pd.Series:
    """Datetimes from the groups of ``_SLASH_DATE``; impossible dates become NaT."""
    numbers = parts[[0, 1, 2, 3, 4, 5]].apply(pd.to_numeric)
    hour = numbers[3].fillna(0)
    half = parts[7].str.lower()
    hour = hour.where(~((half == "pm") & (hour < 12)), hour + 12)
    hour = hour.where(~((half == "am") & (hour == 12)), 0)
    times = pd.to_datetime(pd.DataFrame({
        "year": numbers[2], "month": numbers[month], "day": numbers[day],
        "hour": hour, "minute": numbers[4].fillna(0), "second": numbers[5].fillna(0),
    }), errors="coerce")
    nanos = pd.to_numeric(parts[6].fillna("0").str.ljust(9, "0"))
    return times + pd.to_timedelta(nanos, unit="ns")


def _backward_steps(times: pd.Series, sources: pd.Series | None) -> int:
    """How often the time goes back between consecutive rows of the same file."""
    values = pd.Series(times.to_numpy(), index=times.index)
    keys = sources if sources is not None else pd.Series(0, index=times.index)
    steps = 0
    for _, part in values.groupby(keys, sort=False):
        part = part.dropna()
        steps += int((part.diff() < pd.Timedelta(0)).sum())
    return steps


def _same_order(first: pd.Series, second: pd.Series) -> bool:
    if not first.isna().equals(second.isna()):
        return False
    return cast(bool, first.rank(method="dense").equals(second.rank(method="dense")))


def _parse_text_times(text: pd.Series,
                      sources: pd.Series | None) -> tuple[pd.Series, str]:
    """Parse a text column; returns (times, format) or raises _FormatAmbiguous."""
    unique = pd.Series(pd.unique(text.to_numpy()), dtype=object)
    codes = pd.Index(unique).get_indexer(text.to_numpy())

    def per_row(parsed: pd.Series) -> pd.Series:
        return pd.Series(parsed.to_numpy()[codes], index=text.index)

    if unique.str.match(_SLASH_DATE).mean() >= 0.5:
        parts = unique.str.extract(_SLASH_DATE)
        day_first = per_row(_from_parts(parts, day=0, month=1))
        month_first = per_row(_from_parts(parts, day=1, month=0))
        fails = (int(day_first.isna().sum()), int(month_first.isna().sum()))
        if fails[0] != fails[1]:
            return (day_first, "day-first") if fails[0] < fails[1] else (month_first, "month-first")
        if _same_order(day_first, month_first):
            return day_first, "same-order"
        back = (_backward_steps(day_first, sources), _backward_steps(month_first, sources))
        if back[0] == back[1]:
            raise _FormatAmbiguous
        return (day_first, "day-first") if back[0] < back[1] else (month_first, "month-first")
    if unique.str.match(_ISO_DATE).mean() >= 0.5:
        return per_row(pd.to_datetime(unique, format="ISO8601", errors="coerce")), "ISO8601"
    with warnings.catch_warnings():  # "could not infer format": expected for odd formats
        warnings.simplefilter("ignore", UserWarning)
        return per_row(pd.to_datetime(unique, errors="coerce")), "inferred"


def check_time_column(frame: pd.DataFrame, column: str | None, config: SplitConfig,
                      sources: pd.Series | None = None) -> tuple[TimeCheck, pd.Series | None]:
    """Validate a time column; returns the check and the parsed times (NaN/NaT = unusable)."""
    name = column or ""
    if column is None or column not in frame.columns:
        return TimeCheck(name, False, "TIME_COLUMN_MISSING", None, 1.0, 0.0, len(frame)), None
    values = frame[column]
    missing = values.isna()
    if values.dtype == object or isinstance(values.dtype, pd.CategoricalDtype):
        text = values.astype(object).where(~missing).astype(str).str.strip()
        missing = missing | (text == "")
    missing_ratio = float(missing.mean()) if len(values) else 1.0

    def failed(code: str, fmt: str | None = None,
               parse_ratio: float = 0.0) -> tuple[TimeCheck, None]:
        return TimeCheck(column, False, code, fmt, missing_ratio, parse_ratio, len(frame)), None

    if missing_ratio > config.max_time_missing:
        return failed("TIME_MISSING_RATIO_EXCEEDED")
    present = values[~missing]
    if pd.api.types.is_datetime64_any_dtype(values):
        times, fmt = values, "datetime"
    elif values.dtype.kind in "iuf":
        times, fmt = values.astype(np.float64).where(np.isfinite(values.astype(np.float64))), "number"
    else:
        try:
            parsed, fmt = _parse_text_times(present.astype(str).str.strip(),
                                            None if sources is None else sources[~missing])
        except _FormatAmbiguous:
            return failed("TIME_FORMAT_AMBIGUOUS")
        times = parsed.reindex(values.index)
    unusable = times.isna()
    parse_ratio = float(unusable[~missing].mean()) if len(present) else 1.0
    if parse_ratio > config.max_time_parse_failure:
        return failed("TIME_PARSE_FAILURE_RATIO_EXCEEDED", fmt, parse_ratio)
    if sources is not None:
        spans = times[~unusable].groupby(sources[~unusable]).agg(["min", "max"]).sort_values("min")
        if (spans["min"].to_numpy()[1:] < spans["max"].to_numpy()[:-1]).any():
            return failed("TIME_NON_MONOTONIC_ACROSS_FILES", fmt, parse_ratio)
    check = TimeCheck(column, True, None, fmt, missing_ratio, parse_ratio, int(unusable.sum()))
    return check, times
