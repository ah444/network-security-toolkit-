"""Why one prediction came out as it did (v5.3 F3): an approximation, not SHAP.

Each feature's value is replaced by the feature's typical value in the training split (the
median of a number, the most frequent category) and the probability of the predicted class
is measured again: a feature whose real value raised that probability pushes toward the
class, one that lowered it pushes away. One prediction of a small table (a row per changed
feature), so it is immediate. A model without probabilities gets no explanation.

Its limits, to show beside it: it is approximate, not causal; correlated features share and
hide each other's effect; one feature set to its typical value alone can make a row that
never occurs; the probabilities are not calibrated.

The typical values are in the run's record since v5.3 F3; a bundle saved before has only
the medians its pipeline learnt (the numeric imputer), so its categories go unexplained.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...data.settings import ExplainConfig
from .pipeline import FittedModel

log = logging.getLogger(__name__)

NO_PROBABILITIES = "NO_PROBABILITIES"
TRAINING = "training"  # the typical values recorded from the training split
PIPELINE = "pipeline"  # an older bundle: the medians its imputer learnt, numbers only
NOISE = 1e-9  # a change in probability this small is rounding, not a reason


@dataclass(frozen=True)
class Reason:
    """One feature of the row: its value, its typical value, and how much its real value
    raised (positive) or lowered (negative) the predicted class's probability."""
    feature: str
    value: Any
    typical: Any
    effect: float


@dataclass(frozen=True)
class Explanation:
    """The prediction, its probability, the features that push most toward it and most away
    from it, the features with no typical value, where the typical values came from, and why
    there is no explanation (``NO_PROBABILITIES``), if so."""
    prediction: str
    probability: float | None
    toward: tuple[Reason, ...]
    away: tuple[Reason, ...]
    without_typical: tuple[str, ...]
    source: str
    problem: str | None = None


def typical_values(fitted: FittedModel,
                   training: Mapping[str, Any] | None) -> tuple[dict[str, Any], str]:
    """The typical value of each of ``fitted``'s features and where they came from: the
    record's ``training`` part (``TRAINING``), else the medians of the pipeline's imputer
    (``PIPELINE``: a bundle saved before v5.3 F3; numbers only)."""
    if training is not None and "typical_values" in training:
        recorded = training["typical_values"]
        return {f: recorded[f] for f in fitted.features if recorded.get(f) is not None}, TRAINING
    medians: dict[str, Any] = {}
    for part, transformer, columns in fitted.pipeline.named_steps["prepare"].transformers_:
        if part == "numeric":
            statistics = transformer.named_steps["impute"].statistics_
            medians.update((column, float(value)) for column, value in
                           zip(columns, statistics, strict=True) if np.isfinite(value))
    return medians, PIPELINE


def _plain(value: Any) -> Any:
    """A NumPy value as a Python one."""
    return value.item() if isinstance(value, np.generic) else value


def _same(value: Any, typical: Any) -> bool:
    """Whether the row already holds the typical value (a number, or a category as text)."""
    if isinstance(typical, (int, float)) and not isinstance(typical, bool):
        try:
            return float(value) == float(typical)
        except (TypeError, ValueError):
            return False
    return str(value) == str(typical)


def explain(fitted: FittedModel, row: pd.DataFrame, typical: Mapping[str, Any], source: str,
            config: ExplainConfig | None = None) -> Explanation:
    """Why ``fitted`` predicts what it does for the one-row ``row`` (see the module docstring),
    with the ``typical`` values from ``typical_values``."""
    config = config or ExplainConfig()
    prediction = str(fitted.predict(row)[0])
    without = tuple(f for f in fitted.features if f not in typical)
    proba = fitted.predict_proba(row)
    if proba is None:
        return Explanation(prediction, None, (), (), without, source, NO_PROBABILITIES)
    column = fitted.classes.index(prediction)
    values = {f: _plain(row[f].iloc[0]) for f in fitted.features}
    changed = [f for f in fitted.features if f in typical and not _same(values[f], typical[f])]
    reasons = []
    if changed:
        table = pd.concat([row[list(fitted.features)]] * len(changed), ignore_index=True)
        for index, feature in enumerate(changed):
            table.loc[index, feature] = typical[feature]
        replaced = fitted.predict_proba(table)
        assert replaced is not None  # the same model gave the row's probabilities
        effects = proba[0][column] - replaced[:, column]
        reasons = [Reason(f, values[f], typical[f], float(effect))
                   for f, effect in zip(changed, effects, strict=True) if abs(effect) > NOISE]
    toward = sorted((r for r in reasons if r.effect > 0), key=lambda r: (-r.effect, r.feature))
    away = sorted((r for r in reasons if r.effect < 0), key=lambda r: (r.effect, r.feature))
    return Explanation(prediction, float(proba[0][column]), tuple(toward[:config.local_top]),
                       tuple(away[:config.local_top]), without, source)


def why(fitted: FittedModel, row: pd.DataFrame,
        training: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``explain`` with the typical values of a record's ``training`` part, as the answer's
    ``why`` (the web's, F3d, and the CLI's, F3f). None when it fails (logged): the prediction
    stands without it."""
    try:
        typical, source = typical_values(fitted, training)
        return asdict(explain(fitted, row, typical, source))
    except Exception:
        log.exception("Explanation error")
        return None
