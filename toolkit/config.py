"""Tunable settings in one place (plan D17). Tests rely on these defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
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
    # Official split (published train + test files): False keeps both files as published,
    # so scores stay comparable; True ("strict") removes copies and test rows also in train.
    official_dedupe: bool = False


def max_workers(environ: Mapping[str, str] | None = None) -> int:
    """CPU workers for training and CV: ``TOOLKIT_MAX_WORKERS``, else all cores but one (D8).

    Never more than the machine has; a value that isn't a positive whole number is an error,
    so a typo can't silently use every core.
    """
    environ = os.environ if environ is None else environ
    cores = os.cpu_count() or 1
    raw = environ.get("TOOLKIT_MAX_WORKERS", "").strip()
    if not raw:
        return max(1, cores - 1)
    try:
        workers = int(raw)
    except ValueError:
        workers = 0
    if workers < 1:
        raise ValueError(f"TOOLKIT_MAX_WORKERS must be a whole number >= 1, not {raw!r}")
    return min(workers, cores)


@dataclass(frozen=True)
class PipelineConfig:
    """Training pipeline settings (plan D9, D8, step 0.4c)."""

    random_state: int = 42  # every estimator and sampler, including nested ones (B20)
    smote_max_neighbors: int = 5  # lowered for small classes (smallest train class - 1)
    # Linear SVM is skipped above this many **training** rows (B5; v4 used the whole file).
    svm_max_train_rows: int = 100_000
