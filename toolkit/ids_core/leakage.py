"""Leakage guard: find columns that must not be used as features (plan D9, L6).

Run it on the **training split only** (never look at the test set to choose
features). Four independent checks, each giving reason codes:

- ``PROFILE_BLOCKED``: the dataset profile says so (e.g. NSL-KDD ``level``).
- ``NAME_PATTERN``: the name contains a label/ID/time word as a whole word
  (``id`` matches ``Flow ID`` but not ``Idle Mean``).
- ``ID_LIKE``: an integer or text column that is (almost) all unique, or strictly
  increasing in row order. Floats are never flagged this way: real flow features
  such as ``Flow Bytes/s`` are almost all unique too.
- Single-feature rule: a small decision tree (8 leaves, or 2 per class when there
  are more classes) on that one feature predicts the label. ``LABEL_RELABELLING``
  (block) when the score is near perfect **and** each class maps to about one
  value (a renamed, encoded or prefixed copy of the label);
  ``STRONG_SINGLE_FEATURE`` (warn only) when the score is high but the feature
  varies within classes, as genuinely strong features do on easy data.

Blocked columns are removed from ``features`` unless the caller allows them with a
written reason; every such override is recorded in the report (for the TrainingRun).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from ..config import LeakageConfig

# Whole-word patterns (token sequences), checked after splitting the name into words.
NAME_PATTERNS = (
    "label", "attack", "attack cat", "level", "attempted category", "class", "ip", "src ip", "dst ip",
    "timestamp", "flow id", "flowid", "id", "row", "index", "unnamed",
)
_PATTERN_TOKENS = tuple((p, tuple(p.split())) for p in NAME_PATTERNS)


@dataclass(frozen=True)
class ColumnFinding:
    column: str
    action: str  # "block" | "warn"
    reason: str  # PROFILE_BLOCKED | NAME_PATTERN | ID_LIKE | LABEL_RELABELLING | STRONG_SINGLE_FEATURE
    detail: str = ""
    score: float | None = None  # balanced accuracy of the one-feature tree
    values_per_class: float | None = None


@dataclass(frozen=True)
class Override:
    column: str
    action: str
    reason: str
    score: float | None
    note: str  # why the user kept it


@dataclass(frozen=True)
class LeakageReport:
    features: tuple[str, ...]  # columns that may be used as features
    blocked: tuple[str, ...]
    warned: tuple[str, ...]
    findings: tuple[ColumnFinding, ...]
    overrides: tuple[Override, ...]
    rows: int
    sample_rows: int  # rows used by the single-feature check
    excluded_classes: tuple[str, ...]  # too rare for the cross-validation
    config: LeakageConfig


def _words(name: str) -> tuple[str, ...]:
    """'srcIP' -> ('src', 'ip'); 'Flow_ID' -> ('flow', 'id'); 'Label2' -> ('label', '2')."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)  # camelCase
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", spaced)
    return tuple(re.findall(r"[a-z0-9]+", spaced.lower()))


def name_pattern(name: str) -> str | None:
    """The first leakage pattern found in the column name as whole words, else None."""
    words = _words(name)
    for pattern, tokens in _PATTERN_TOKENS:
        size = len(tokens)
        if any(words[i:i + size] == tokens for i in range(len(words) - size + 1)):
            return pattern
    return None


def _id_like(column: pd.Series, config: LeakageConfig) -> str | None:
    kind = column.dtype.kind
    is_text = kind == "O" or isinstance(column.dtype, (pd.CategoricalDtype, pd.StringDtype))
    if not (kind in "iu" or is_text):
        return None
    values = column.dropna()
    if kind in "iu" and len(values) >= config.id_increasing_min_rows:
        if values.sort_index().is_monotonic_increasing and values.is_unique:
            return "strictly increasing in row order"
    if len(values) >= config.id_unique_min_rows:
        share = values.nunique() / len(values)
        if share >= config.id_unique_share:
            return f"{share:.1%} unique values"
    return None


def _as_number(column: pd.Series) -> np.ndarray:
    """One numeric column for the tree: text -> category codes, inf -> missing."""
    if column.dtype.kind in "iufb":
        values = column.to_numpy(dtype=np.float64)
    else:
        codes = column.astype("category").cat.codes.to_numpy(dtype=np.float64)
        values = np.where(codes < 0, np.nan, codes)
    values[~np.isfinite(values)] = np.nan
    return values.reshape(-1, 1)


def _class_quotas(counts: pd.Series, total: int, minimum: int) -> dict[str, int]:
    """Rows per class for a sample of ``total`` rows: proportional (largest remainder),
    but never fewer than ``minimum`` (or the whole class if it is smaller)."""
    floor = {c: min(n, minimum) for c, n in counts.items()}
    spare = total - sum(floor.values())
    if spare <= 0:  # so many classes that the minimums alone fill the sample
        return floor
    extra = {c: (n - floor[c]) * spare / (counts.sum() - sum(floor.values()))
             for c, n in counts.items()}
    quotas = {c: floor[c] + int(extra[c]) for c in counts.index}
    left = total - sum(quotas.values())
    for c in sorted(counts.index, key=lambda c: (-(extra[c] - int(extra[c])), str(c)))[:left]:
        quotas[c] += 1
    return quotas


def _sample(frame: pd.DataFrame, label: str, config: LeakageConfig):
    """Drop rows without a label and classes too rare for CV; stratified sample if large.

    Every remaining class keeps at least ``cv_folds`` rows in the sample, so the
    stratified cross-validation never sees a class with fewer members than folds.
    """
    frame = frame[frame[label].notna()]
    labels = frame[label].astype(str)
    counts = labels.value_counts()
    rare = tuple(sorted(counts[counts < config.cv_folds].index))
    if rare:
        keep = ~labels.isin(rare)
        frame, labels, counts = frame[keep], labels[keep], counts.drop(list(rare))
    if len(frame) > config.sample_rows:
        quotas = _class_quotas(counts, config.sample_rows, config.cv_folds)
        rng = np.random.default_rng(config.random_state)
        picked = [rng.choice(np.flatnonzero(labels.to_numpy() == c), size=n, replace=False)
                  for c, n in sorted(quotas.items())]
        frame = frame.iloc[np.sort(np.concatenate(picked))]
    return frame, rare


def _single_feature(x: np.ndarray, y: np.ndarray, leaves: int, config: LeakageConfig) -> float:
    tree = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=config.random_state)
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True,
                            random_state=config.random_state)
    return float(cross_val_score(tree, x, y, cv=folds, scoring="balanced_accuracy").mean())


def check_leakage(
    train: pd.DataFrame,
    label: str,
    *,
    features: Sequence[str] | None = None,
    profile_blocked: Sequence[str] = (),
    allow: Mapping[str, str] | None = None,
    config: LeakageConfig | None = None,
) -> LeakageReport:
    """Check candidate features of the **training split** for label leakage and IDs.

    ``allow`` maps a column to the user's reason for keeping it despite a finding.
    """
    config = config or LeakageConfig()
    candidates = [c for c in (features if features is not None else train.columns) if c != label]
    allow = dict(allow or {})
    for column, note in allow.items():
        if column not in candidates:
            raise ValueError(f"Override for unknown column: {column!r}")
        if not str(note).strip():
            raise ValueError(f"Override for {column!r} needs a reason")

    findings: list[ColumnFinding] = []
    for column in candidates:
        if column in profile_blocked:
            findings.append(ColumnFinding(column, "block", "PROFILE_BLOCKED",
                                          "the dataset profile lists it"))
        pattern = name_pattern(column)
        if pattern:
            findings.append(ColumnFinding(column, "block", "NAME_PATTERN", f"word '{pattern}'"))
        id_detail = _id_like(train[column], config)
        if id_detail:
            findings.append(ColumnFinding(column, "block", "ID_LIKE", id_detail))

    sample, rare = _sample(train[[*candidates, label]], label, config)
    y = sample[label].astype(str).to_numpy()
    classes = np.unique(y)
    already = {f.column for f in findings}
    if len(classes) >= 2:
        leaves = max(config.tree_leaves,
                     math.ceil(config.relabel_max_values_per_class * len(classes)))
        for column in candidates:
            if column in already:
                continue
            score = _single_feature(_as_number(sample[column]), y, leaves, config)
            if score < config.warn_score:
                continue
            per_class = float(sample.groupby(y, observed=True)[column]
                              .nunique(dropna=False).mean())
            relabel = per_class <= config.relabel_max_values_per_class
            if score >= config.block_score and relabel:
                findings.append(ColumnFinding(column, "block", "LABEL_RELABELLING",
                                              "about one value per class", score, per_class))
            else:
                findings.append(ColumnFinding(column, "warn", "STRONG_SINGLE_FEATURE",
                                              "predicts the label alone; may be legitimate",
                                              score, per_class))

    overrides = tuple(Override(f.column, f.action, f.reason, f.score, allow[f.column].strip())
                      for f in findings if f.column in allow)
    blocked = tuple(dict.fromkeys(f.column for f in findings
                                  if f.action == "block" and f.column not in allow))
    warned = tuple(dict.fromkeys(f.column for f in findings if f.action == "warn"))
    return LeakageReport(
        features=tuple(c for c in candidates if c not in blocked),
        blocked=blocked,
        warned=warned,
        findings=tuple(findings),
        overrides=overrides,
        rows=len(train),
        sample_rows=len(sample),
        excluded_classes=rare,
        config=config,
    )
