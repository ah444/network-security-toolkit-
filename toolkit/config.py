"""Tunable settings in one place (plan D17). Tests rely on these defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeakageConfig:
    """Thresholds of the leakage guard (plan D9, step 0.4a)."""

    # Single-feature rule: balanced accuracy of a one-feature tree (stratified CV).
    block_score: float = 0.99  # ... and relabelling-like -> block
    warn_score: float = 0.95  # otherwise -> warn
    # "Relabelling": on average at most this many distinct values per class.
    relabel_max_values_per_class: float = 2.0
    # Tree size as leaves: 8 = the capacity of the plan's depth-3 tree. With many classes
    # it grows to relabel_max_values_per_class leaves per class, so a copy of a 20-class
    # label is still caught (a greedy tree needs about one split per class, not log2).
    tree_leaves: int = 8
    cv_folds: int = 3
    sample_rows: int = 50_000  # stratified sample of the training split for the CV check
    # ID-like columns (integer or text only).
    id_unique_share: float = 0.99
    id_unique_min_rows: int = 1_000  # below this, most integer features look unique
    id_increasing_min_rows: int = 20
    random_state: int = 0


@dataclass(frozen=True)
class SplitConfig:
    """Train/test split settings (plan D9, step 0.4b)."""

    test_size: float = 0.2
    random_state: int = 42  # the seed the v4 app used for its split
    # Time-column checks before a temporal split; above these the split falls back.
    max_time_parse_failure: float = 0.01  # share of non-empty values that don't parse
    max_time_missing: float = 0.01  # share of empty values
