"""What a model relies on (v5.3 F3): permutation importance on the test split.

The values of one feature are shuffled and the drop in macro-F1 is measured; a feature the
model doesn't use costs nothing. It is scikit-learn's ``permutation_importance``, asked one
feature at a time so a job can show its progress and stop between two features; the numbers
are the same as one call for every feature (the same seed for each column). The columns
shuffled are the ones the user chose, before one-hot encoding: one number per feature. The
models' own ``feature_importances_`` and ``coef_`` are not used: not every model has them,
and a tree's favour features with many values.

The test split is reused here, after training and evaluation: the result is a diagnosis,
never a way to choose features, models or hyperparameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from ...data.settings import ExplainConfig
from ..ports import NullProgress, ProgressSink
from .pipeline import FittedModel
from .sampling import stratified_positions


@dataclass(frozen=True)
class FeatureImportance:
    """The drop in macro-F1 when one feature's values are shuffled, over the repeats."""
    feature: str
    mean: float
    std: float


@dataclass(frozen=True)
class Importance:
    """The permutation importance of one model: the features ranked by their mean drop, how
    many test rows were scored (a sample above ``importance_max_rows``), the repeats and the
    seed, the macro-F1 before any shuffle, and the feature that carries most of the effect,
    if one does (check that it is not a copy of the label)."""
    model: str
    rows: int
    test_rows: int
    repeats: int
    seed: int
    baseline_macro_f1: float
    features: tuple[FeatureImportance, ...]
    dominant: str | None


class _OneFeature:
    """``fitted`` seen through one feature: scikit-learn shuffles the only column it is given,
    and the other features keep their values."""

    def __init__(self, fitted: FittedModel, rows: pd.DataFrame, feature: str) -> None:
        self.fitted, self.rows, self.feature = fitted, rows, feature

    def fit(self, *_args: Any) -> _OneFeature:
        """Already fitted: scikit-learn only checks that the method is there."""
        return self

    def predict(self, column: pd.DataFrame) -> np.ndarray:
        """The predictions with this feature's values from ``column``."""
        frame = self.rows.copy()
        frame[self.feature] = column[self.feature].to_numpy()
        return self.fitted.predict(frame)


def scored_rows(test: pd.DataFrame, label: str, seed: int,
                config: ExplainConfig | None = None) -> pd.DataFrame:
    """The labelled test rows, or a stratified sample of them above ``importance_max_rows``."""
    config = config or ExplainConfig()
    test = test[test[label].notna()]
    if len(test) <= config.importance_max_rows:
        return test
    positions = stratified_positions(test[label].astype(str).to_numpy(), config.importance_max_rows,
                                     config.importance_min_per_class, seed)
    return test.iloc[positions]


def dominant_feature(features: Sequence[FeatureImportance], share: float) -> str | None:
    """The feature whose mean drop is more than ``share`` of the summed positive drops (of two
    features or more), or None."""
    total = sum(item.mean for item in features if item.mean > 0)
    if len(features) < 2 or total <= 0:
        return None
    top = max(features, key=lambda item: item.mean)
    return top.feature if top.mean > share * total else None


def permutation_importances(fitted: FittedModel, test: pd.DataFrame, label: str, *, seed: int,
                            config: ExplainConfig | None = None,
                            progress: ProgressSink | None = None) -> Importance:
    """The permutation importance of ``fitted`` on ``test`` (see the module docstring).

    ``progress`` gets ``start(len(features))``, one ``advance()`` per feature, then
    ``finish()``, also after an error; an exception raised by ``advance()`` (a cancelled
    job) stops it, and nothing is returned."""
    config = config or ExplainConfig()
    progress = progress or NullProgress()
    rows = scored_rows(test, label, seed, config)
    truth = rows[label].astype(str).to_numpy()
    labels = sorted(set(fitted.classes) | set(truth))

    def macro_f1(estimator: _OneFeature, column: pd.DataFrame, expected: np.ndarray) -> float:
        return float(f1_score(expected, estimator.predict(column), labels=labels,
                              average="macro", zero_division=0))

    found = []
    progress.start(len(fitted.features))
    try:
        for feature in fitted.features:
            result = permutation_importance(_OneFeature(fitted, rows, feature), rows[[feature]],
                                            truth, scoring=macro_f1,
                                            n_repeats=config.importance_repeats, random_state=seed)
            found.append(FeatureImportance(feature, float(result.importances_mean[0]),
                                           float(result.importances_std[0])))
            progress.advance()
    finally:
        progress.finish()
    ranked = tuple(sorted(found, key=lambda item: (-item.mean, item.feature)))
    baseline = float(f1_score(truth, fitted.predict(rows), labels=labels, average="macro",
                              zero_division=0))
    return Importance(model=fitted.name, rows=len(rows), test_rows=int(test[label].notna().sum()),
                      repeats=config.importance_repeats, seed=seed, baseline_macro_f1=baseline,
                      features=ranked, dominant=dominant_feature(ranked, config.dominant_share))
