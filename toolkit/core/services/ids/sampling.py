"""Stratified row samples that keep every class (used by the leakage guard and the CV)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def class_quotas(counts: pd.Series, total: int, minimum: int) -> dict[str, int]:
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


def stratified_positions(labels: np.ndarray, total: int, minimum: int, seed: int) -> np.ndarray:
    """Sorted positions of a seeded stratified sample of ``total`` rows (see class_quotas)."""
    counts = pd.Series(labels).value_counts()
    quotas = class_quotas(counts, total, minimum)
    rng = np.random.default_rng(seed)
    picked = [rng.choice(np.flatnonzero(labels == c), size=n, replace=False)
              for c, n in sorted(quotas.items())]
    return np.sort(np.concatenate(picked))
