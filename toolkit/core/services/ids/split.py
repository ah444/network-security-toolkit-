"""Dedupe, then split into train/test the way the dataset profile says (plan D9, L4, L5).

Order: drop rows without a label -> remove exact copies -> split. Copies are found on
every column except the label-free identifiers (profile-blocked columns, name-pattern
hits such as ``Flow ID``/``Timestamp``, the time and group columns): two flows that
differ only in their ID are the same row to a model.

Strategies, each with its prerequisites; when one fails the next is tried and the
failure is recorded as an enumerated reason code (never free text, never silent):

==================  =====================================  ===============================
strategy            prerequisite                           falls back to
==================  =====================================  ===============================
official            a separate test file                   stratified
temporal            valid time column, every class on      temporal_per_class, group,
                    both sides of one global cut           stratified
temporal_per_class  valid time column, each class has      group, stratified
                    more than one distinct time
group               a group column (or source files),      stratified
                    enough groups, no class on one side
stratified          none (rows of a single-row class go    -
                    to train)
==================  =====================================  ===============================

Official split: by default both published files are used exactly as they are (scores
stay comparable with other work); copies and test rows that also occur in train are
counted and reported. ``SplitConfig.official_dedupe`` (strict mode) removes them.

The strategies themselves: ``split_strategies.py``. Time validation: ``split_time.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from ...data.settings import SplitConfig
from . import split_strategies as strategies
from .leakage import name_pattern
from .profiles import ProfileMatch
from .split_strategies import StrategyFailed
from .split_time import TimeCheck, check_time_column

log = logging.getLogger(__name__)

STRATEGIES = ("official", "temporal", "temporal_per_class", "group", "stratified")
FALLBACKS = {
    "official": ("stratified",),
    "temporal": ("temporal_per_class", "group", "stratified"),
    "temporal_per_class": ("group", "stratified"),
    "group": ("stratified",),
    "stratified": (),
}
TEMPORAL = ("temporal", "temporal_per_class")
SOURCE_GROUP = "(source file)"  # group_column value when grouping by the files rows came from

# Why a strategy was not used (SplitResult.reason_codes).
REASON_CODES = (
    "OFFICIAL_TEST_FILE_MISSING",
    "TIME_COLUMN_MISSING",
    "TIME_MISSING_RATIO_EXCEEDED",
    "TIME_PARSE_FAILURE_RATIO_EXCEEDED",
    "TIME_FORMAT_AMBIGUOUS",
    "TIME_NON_MONOTONIC_ACROSS_FILES",
    "TIME_TOO_COARSE_FOR_CLASS",
    "CLASS_MISSING_FROM_TEST",
    "CLASS_MISSING_FROM_TRAIN",
    "GROUP_COLUMN_MISSING",
    "GROUP_TOO_FEW_GROUPS",
    "GROUP_CLASS_ONE_SIDED",
)
# Facts about the split that was made (SplitResult.warnings).
WARNING_CODES = (
    "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC",
    "CLASS_MISSING_FROM_TEST",
    "CLASS_MISSING_FROM_TRAIN",
    "OFFICIAL_TEST_OVERLAP_KEPT",
    "OFFICIAL_COPIES_KEPT",
    "OFFICIAL_TEST_OVERLAP_REMOVED",
    "TIME_MISSING_ROWS_DROPPED",
    "LABEL_MISSING_ROWS_DROPPED",
)


@dataclass(frozen=True)
class DedupeStats:
    rows_before: int
    rows_after: int
    removed: int
    removed_per_class: tuple[tuple[str, int], ...]
    conflicting_rows: int  # kept rows whose features equal another row's, label differs
    compared_columns: int
    ignored_columns: tuple[str, ...]
    applied: bool = True  # False: official files kept as published, copies only counted
    copies_found: int = 0  # copies in the (training) frame, removed or not
    # Official split only: the separate test file.
    test_rows_before: int = 0
    test_copies: int = 0  # copies inside the test file
    test_overlap: int = 0  # test rows that also occur in train
    test_removed: int = 0  # strict mode: copies + overlap; otherwise 0


@dataclass(frozen=True)
class ClassCoverage:
    label: str
    train: int
    test: int


@dataclass(frozen=True, eq=False)
class SplitResult:
    train: pd.DataFrame  # original index kept, so rows can be traced back
    test: pd.DataFrame
    strategy: str  # the one used
    requested: str  # by the user, else the profile, else "stratified"
    profile_strategy: str | None
    overridden: bool  # the user chose something other than the profile's strategy
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    time_column: str | None  # used by the split (temporal strategies only)
    group_column: str | None  # used by the split (group only)
    time_check: TimeCheck | None
    official_mode: str | None  # "as_published" | "deduplicated" (official split only)
    dedupe: DedupeStats
    coverage: tuple[ClassCoverage, ...]
    rows_without_label: int
    seed: int
    test_size: float


# --- Dedupe ------------------------------------------------------------------------

def dedupe(frame: pd.DataFrame, label: str,
           ignore: Sequence[str] = ()) -> tuple[pd.DataFrame, DedupeStats]:
    """Remove exact copies (first occurrence kept); identifier columns are not compared."""
    ignored = tuple(c for c in frame.columns
                    if c != label and (c in set(ignore) or name_pattern(str(c))))
    columns = [c for c in frame.columns if c not in ignored]
    copies = frame.duplicated(subset=columns, keep="first")
    kept = frame[~copies]
    removed = frame.loc[copies, label].astype(str).value_counts()
    features = [c for c in columns if c != label]
    conflicting = int(kept.duplicated(subset=features, keep=False).sum()) if features else 0
    stats = DedupeStats(
        rows_before=len(frame),
        rows_after=len(kept),
        removed=int(copies.sum()),
        copies_found=int(copies.sum()),
        removed_per_class=tuple(sorted((str(k), int(v)) for k, v in removed.items() if v)),
        conflicting_rows=conflicting,
        compared_columns=len(columns),
        ignored_columns=ignored,
    )
    return kept, stats


# --- Entry point -------------------------------------------------------------------

def _coverage(train: pd.Series, test: pd.Series) -> tuple[ClassCoverage, ...]:
    train_counts = train.astype(str).value_counts()
    test_counts = test.astype(str).value_counts()
    names = sorted(set(train_counts.index[train_counts > 0]) | set(test_counts.index[test_counts > 0]))
    return tuple(ClassCoverage(n, int(train_counts.get(n, 0)), int(test_counts.get(n, 0)))
                 for n in names)


def split_dataset(
    frame: pd.DataFrame,
    label: str,
    *,
    profile: ProfileMatch | None = None,
    strategy: str | None = None,
    time_column: str | None = None,
    group_column: str | None = None,
    test_frame: pd.DataFrame | None = None,
    sources: Sequence[str] | pd.Series | None = None,
    ignore: Sequence[str] = (),
    config: SplitConfig | None = None,
) -> SplitResult:
    """Dedupe ``frame`` and split it into train and test.

    ``strategy`` overrides the profile's (recorded as ``overridden``). ``sources`` names
    the file each row came from (one entry per row, in row order): it is used to check
    that capture files don't overlap in time and as the group of last resort.
    ``test_frame`` is the separate test file of an ``official`` split.
    """
    config = config or SplitConfig()
    if label not in frame.columns:
        raise ValueError(f"Label column {label!r} not found")
    profile_strategy = profile.split_strategy if profile else None
    requested = strategy or profile_strategy or "stratified"
    if requested not in STRATEGIES:
        raise ValueError(f"Unknown split strategy {requested!r}; use one of {STRATEGIES}")
    if profile is not None:
        time_column = time_column or profile.time_column
        group_column = group_column or profile.group_column
    if sources is not None:
        sources = pd.Series(np.asarray(sources), index=frame.index)
    ignored = [c for c in (*ignore, *(profile.blocked_columns if profile else ()),
                           time_column, group_column) if c and c != label]

    result_warnings: list[str] = []
    has_label = frame[label].notna()
    without_label = int((~has_label).sum())
    if without_label:
        frame = frame[has_label]
        result_warnings.append("LABEL_MISSING_ROWS_DROPPED")
    kept, stats = dedupe(frame, label, ignored)
    if sources is not None:
        sources = cast(pd.Series, sources).loc[kept.index]
    labels = kept[label].astype(str).to_numpy()

    reasons: list[str] = []
    time_check: TimeCheck | None = None
    times: pd.Series | None = None
    used_group: str | None = None
    for candidate in (requested, *FALLBACKS[requested]):
        try:
            if candidate == "official":
                if test_frame is None:
                    raise StrategyFailed("OFFICIAL_TEST_FILE_MISSING")
                train_part, test_part, stats, found = strategies.official(
                    frame, kept, stats, test_frame, config)
                result_warnings.extend(found)
            elif candidate in TEMPORAL:
                if time_check is None:
                    time_check, times = check_time_column(kept, time_column, config, sources)
                if not time_check.passed:
                    assert time_check.reason is not None  # a failed check always has one
                    raise StrategyFailed(time_check.reason)
                split = (strategies.temporal if candidate == "temporal"
                         else strategies.temporal_per_class)
                train_pos, test_pos = split(labels, times, config)
                if time_check.rows_dropped:
                    result_warnings.append("TIME_MISSING_ROWS_DROPPED")
            elif candidate == "group":
                if group_column and group_column in kept.columns:
                    groups, used_group = kept[group_column], group_column
                else:
                    groups, used_group = sources, SOURCE_GROUP
                train_pos, test_pos = strategies.group(labels, groups, config)
            else:
                train_pos, test_pos = strategies.stratified(labels, config)
                result_warnings.append("STRATIFIED_SCORES_MAY_BE_OPTIMISTIC")
        except StrategyFailed as failure:
            if failure.code not in reasons:
                reasons.append(failure.code)
            continue
        if candidate != "official":
            train_part = kept.iloc[np.sort(train_pos)]
            test_part = kept.iloc[np.sort(test_pos)]
        break

    coverage = _coverage(train_part[label], test_part[label])
    if any(c.test == 0 for c in coverage):
        result_warnings.append("CLASS_MISSING_FROM_TEST")
    if any(c.train == 0 for c in coverage):
        result_warnings.append("CLASS_MISSING_FROM_TRAIN")
    log.info("Split %s (requested %s): %d train / %d test rows, reasons %s, warnings %s",
             candidate, requested, len(train_part), len(test_part), reasons, result_warnings)
    return SplitResult(
        train=train_part,
        test=test_part,
        strategy=candidate,
        requested=requested,
        profile_strategy=profile_strategy,
        overridden=strategy is not None and profile_strategy not in (None, strategy),
        reason_codes=tuple(reasons),
        warnings=tuple(dict.fromkeys(result_warnings)),
        time_column=time_column if candidate in TEMPORAL else None,
        group_column=used_group if candidate == "group" else None,
        time_check=time_check,
        official_mode=(("deduplicated" if config.official_dedupe else "as_published")
                       if candidate == "official" else None),
        dedupe=stats,
        coverage=coverage,
        rows_without_label=without_label,
        seed=config.random_state,
        test_size=config.test_size,
    )
