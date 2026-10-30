"""The split strategies (plan D9, L5; moved out of ``split.py`` in v5.2 step 2.3g).

Each one takes the deduplicated rows and returns the train and test positions (the
official split: the two frames), or raises ``StrategyFailed`` with the reason code when
its prerequisite is not met. ``split.py`` picks the strategy, tries the fallbacks and
builds the result.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ...data.settings import SplitConfig

if TYPE_CHECKING:  # split.py imports this module; the type is only named here
    from .split import DedupeStats


class StrategyFailed(Exception):
    """A strategy's prerequisite is not met; ``code`` is one of ``split.REASON_CODES``."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _test_count(rows: int, test_size: float) -> int:
    """Test rows for a class: rounded share, at least 1 and at most rows - 1 (0 if rows < 2)."""
    if rows < 2:
        return 0
    return min(rows - 1, max(1, int(np.floor(rows * test_size + 0.5))))


def _class_positions(labels: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """Each class (sorted) and the positions of its rows."""
    for name in sorted(np.unique(labels)):
        yield name, np.flatnonzero(labels == name)


def stratified(labels: np.ndarray, config: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Random positions per class: each class's share goes to the test (seeded)."""
    rng = np.random.default_rng(config.random_state)
    train, test = [], []
    for _, positions in _class_positions(labels):
        shuffled = rng.permutation(positions)
        count = _test_count(len(positions), config.test_size)
        test.append(shuffled[:count])
        train.append(shuffled[count:])
    return np.concatenate(train), np.concatenate(test)


def _cut(times: np.ndarray, test_rows: int) -> int | None:
    """Position splitting sorted ``times`` so equal times stay on one side; None if impossible."""
    rows = len(times)
    point = rows - test_rows
    if times[point - 1] < times[point]:
        return point
    low = int(np.searchsorted(times, times[point], "left"))
    high = int(np.searchsorted(times, times[point], "right"))
    options = [p for p in (low, high) if 0 < p < rows]
    if not options:
        return None
    return min(options, key=lambda p: (abs(rows - p - test_rows), p))


def _time_order(times: pd.Series, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``positions`` sorted by their time (stable), with the sorted times as numbers."""
    keys = times.to_numpy()[positions]
    keys = keys.astype("datetime64[ns]").astype(np.int64) if keys.dtype.kind == "M" else keys
    order = np.argsort(keys, kind="stable")
    return positions[order], keys[order]


def temporal(labels: np.ndarray, times: pd.Series,
             config: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """The latest rows as the test, at a time cut that keeps equal times on one side; fails if no
    such cut exists or a class with two rows or more ends up on one side."""
    usable = np.flatnonzero(times.notna().to_numpy())
    ordered, keys = _time_order(times, usable)
    point = _cut(keys, _test_count(len(ordered), config.test_size))
    if point is None:
        raise StrategyFailed("TIME_TOO_COARSE_FOR_CLASS")
    train, test = ordered[:point], ordered[point:]
    counts = pd.Series(labels[usable]).value_counts()
    splittable = set(counts[counts >= 2].index)
    if splittable - set(labels[test]):
        raise StrategyFailed("CLASS_MISSING_FROM_TEST")
    if splittable - set(labels[train]):
        raise StrategyFailed("CLASS_MISSING_FROM_TRAIN")
    return train, test


def temporal_per_class(labels: np.ndarray, times: pd.Series,
                       config: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Each class's latest rows as its test, cut where equal times stay on one side; fails if a
    class can't be cut so."""
    usable = times.notna().to_numpy()
    train, test = [], []
    for _, positions in _class_positions(labels):
        positions = positions[usable[positions]]
        count = _test_count(len(positions), config.test_size)
        ordered, keys = _time_order(times, positions)
        if count == 0:
            train.append(ordered)
            continue
        point = _cut(keys, count)
        if point is None:
            raise StrategyFailed("TIME_TOO_COARSE_FOR_CLASS")
        train.append(ordered[:point])
        test.append(ordered[point:])
    return np.concatenate(train), np.concatenate(test) if test else np.array([], dtype=int)


def group(labels: np.ndarray, groups: pd.Series | None,
          config: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Train and test without a group on both sides (one fold of a stratified group K-fold); fails
    without groups, with too few, or when a class is left on one side."""
    if groups is None:
        raise StrategyFailed("GROUP_COLUMN_MISSING")
    folds = max(2, round(1 / config.test_size))
    keys = groups.astype(str).to_numpy()
    if len(np.unique(keys)) < folds:
        raise StrategyFailed("GROUP_TOO_FEW_GROUPS")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    with warnings.catch_warnings():  # a class in few groups: reported below as a reason code
        warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
        train, test = next(splitter.split(np.zeros(len(labels)), labels, keys))
    counts = pd.Series(labels).value_counts()
    splittable = set(counts[counts >= 2].index)
    if splittable - set(labels[test]) or splittable - set(labels[train]):
        raise StrategyFailed("GROUP_CLASS_ONE_SIDED")
    return train, test


def official(frame: pd.DataFrame, kept: pd.DataFrame, stats: DedupeStats,
             test_frame: pd.DataFrame, config: SplitConfig,
             ) -> tuple[pd.DataFrame, pd.DataFrame, DedupeStats, list[str]]:
    """Train and test of an official split, the updated stats and the warnings."""
    lacking = [c for c in frame.columns if c not in test_frame.columns]
    if lacking:
        raise ValueError(f"The official test file lacks columns: {lacking}")
    test = test_frame[list(frame.columns)]
    compared = [c for c in frame.columns if c not in stats.ignored_columns]
    copies = test.duplicated(subset=compared, keep="first").to_numpy()
    in_train = (test[compared].astype(object)
                .merge(kept[compared].astype(object).assign(_in_train=True),
                       on=compared, how="left")["_in_train"].notna().to_numpy())
    stats = replace(stats, test_rows_before=len(test), test_copies=int(copies.sum()),
                    test_overlap=int(in_train.sum()))
    found: list[str] = []
    if config.official_dedupe:
        drop = copies | in_train
        stats = replace(stats, test_removed=int(drop.sum()))
        if in_train.any():
            found.append("OFFICIAL_TEST_OVERLAP_REMOVED")
        return kept, test[~drop], stats, found
    stats = replace(stats, applied=False, rows_after=len(frame), removed=0, removed_per_class=())
    if in_train.any():
        found.append("OFFICIAL_TEST_OVERLAP_KEPT")
    if stats.copies_found or copies.any():
        found.append("OFFICIAL_COPIES_KEPT")
    return frame, test, stats, found
