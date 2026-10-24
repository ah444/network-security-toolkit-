"""Train-only previews for the web trainer (step 0.4g; L2, L3).

v4 scored features and ran SMOTE on the whole dataset, so the test rows shaped what the
user then trained on. Both previews here take the **training split** only (the caller
splits first, as training will) and never change the data:

- ``recommend_features``: leakage guard, then one score per allowed feature;
- ``preview_balancing``: class counts before and after a balancing method.

Big training splits are previewed on a seeded stratified sample (``max_rows``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from ..config import LeakageConfig, PipelineConfig, max_workers
from . import registry
from .balancing import BalancingRecommendation, recommend_balancing
from .leakage import check_leakage
from .pipeline import _prepare
from .sampling import stratified_positions

DEFAULT_MAX_ROWS = 200_000
_MIN_PER_CLASS = 50  # rows kept per class when sampling (or the whole class)


@dataclass(frozen=True)
class FeatureScores:
    scores: tuple[tuple[str, float], ...]  # allowed features, best first (may be inf)
    blocked: tuple[tuple[str, str], ...]  # (column, reason code) from the leakage guard
    method: str
    rows: int  # training rows scored
    train_rows: int


@dataclass(frozen=True)
class BalancingPreview:
    requested: str
    balancer: str  # as resolved ("recommended" -> the recommender's choice)
    before: tuple[tuple[str, int], ...]  # class counts of the rows used
    after: tuple[tuple[str, int], ...]
    changes_rows: bool
    rows: int
    train_rows: int
    sampled: bool
    recommendation: BalancingRecommendation | None
    notes: tuple[str, ...]


def _sample(train: pd.DataFrame, label: str, max_rows: int, seed: int) -> pd.DataFrame:
    data = train[train[label].notna()]
    if len(data) <= max_rows:
        return data
    labels = data[label].astype(str).to_numpy()
    classes = len(np.unique(labels))
    minimum = max(1, min(_MIN_PER_CLASS, max_rows // (2 * classes)))  # never above max_rows
    return data.iloc[stratified_positions(labels, max_rows, minimum, seed)]


def _prepared(data: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Features as tree models see them (inf -> missing, imputed, ordinal categories)."""
    numeric = [c for c in features if data[c].dtype.kind in "iufb"]
    categorical = [c for c in features if c not in numeric]
    prepare = _prepare(numeric, categorical, registry.MODELS["decision_tree"])
    return np.asarray(prepare.fit_transform(data[list(features)]), dtype=np.float64), \
        numeric + categorical


def recommend_features(
    train: pd.DataFrame,
    label: str,
    *,
    features: Sequence[str] | None = None,
    profile_blocked: Sequence[str] = (),
    method: str = "f_classif",
    max_rows: int = DEFAULT_MAX_ROWS,
    seed: int = 42,
    leakage_config: LeakageConfig | None = None,
) -> FeatureScores:
    """Score the features the leakage guard allows, on the training split only."""
    if method not in ("f_classif", "mutual_info"):
        raise ValueError(f"Unknown scoring method: {method!r}")
    guard = check_leakage(train, label, features=features, profile_blocked=profile_blocked,
                          config=leakage_config)
    reasons: dict[str, str] = {}
    for finding in guard.findings:
        if finding.action == "block" and finding.column in guard.blocked:
            reasons.setdefault(finding.column, finding.reason)
    blocked = tuple((c, reasons.get(c, "BLOCKED")) for c in guard.blocked)
    data = _sample(train, label, max_rows, seed)
    scores: dict[str, float] = {}
    if guard.features:
        X, order = _prepared(data, guard.features)
        y = LabelEncoder().fit_transform(data[label].astype(str))
        found = registry._Scores(method, seed).raw(X, y)  # inf kept: one value per class
        scores = {name: float(np.nan_to_num(score, posinf=np.inf))
                  for name, score in zip(order, found, strict=True)}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], guard.features.index(item[0])))
    return FeatureScores(tuple(ranked), blocked, method, len(data), int(train[label].notna().sum()))


def _counts(labels: pd.Series) -> tuple[tuple[str, int], ...]:
    counts = labels.astype(str).value_counts()
    return tuple((str(c), int(n)) for c, n in counts.items())


def preview_balancing(
    train: pd.DataFrame,
    label: str,
    balancer: str,
    *,
    features: Sequence[str],
    max_rows: int = DEFAULT_MAX_ROWS,
    config: PipelineConfig | None = None,
    workers: int | None = None,
) -> BalancingPreview:
    """Class counts before/after ``balancer`` on the training split (nothing is kept)."""
    config = config or PipelineConfig()
    requested = balancer
    recommendation = None
    if balancer == "recommended":
        recommendation = recommend_balancing(train[label], len(features))
        balancer = recommendation.recommended
    if balancer not in registry.BALANCERS:
        raise ValueError(f"Unknown balancing method: {requested!r}")
    data = _sample(train, label, max_rows, config.random_state)
    notes = ["SAMPLED"] if len(data) < int(train[label].notna().sum()) else []
    before = _counts(data[label])
    spec = registry.BALANCERS[balancer]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data[label].astype(str))
    context = registry.BuildContext(
        seed=config.random_state, workers=workers or max_workers(),
        class_counts={i: int(n) for i, n in enumerate(np.bincount(y))},
        class_weight=spec.class_weight, smote_max_neighbors=config.smote_max_neighbors)
    sampler = spec.build(context)
    if spec.class_weight:
        notes.append("CLASS_WEIGHT_CHANGES_WEIGHTS_NOT_ROWS")
    if sampler is None:
        after = before
        if spec.oversamples:
            notes.append("NOTHING_TO_OVERSAMPLE")
    else:
        X, _ = _prepared(data, features)
        with threadpool_limits(limits=context.workers):
            _, resampled = sampler.fit_resample(X, y)
        after = _counts(pd.Series(encoder.inverse_transform(resampled)))
    return BalancingPreview(
        requested=requested, balancer=balancer, before=before, after=after,
        changes_rows=after != before, rows=len(data),
        train_rows=int(train[label].notna().sum()), sampled="SAMPLED" in notes,
        recommendation=recommendation, notes=tuple(notes))
