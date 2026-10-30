"""Balancing recommender: the v4 rule table, fed with **training labels only** (plan D9).

v4 (``recommend_balancing_method`` in the web app) used class counts of the whole
dataset, so the test rows influenced the choice. The rules are kept as they were (same
thresholds, same answers); each answer now has a reason code for the TrainingRun.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BalancingRecommendation:
    """The recommended balancing method with its reason and alternatives, and the training-split
    figures the rule table used."""
    recommended: str  # a key of registry.BALANCERS
    reason_code: str
    reason: str
    alternatives: tuple[str, ...]
    imbalance_ratio: float  # smallest / largest class
    severity: str  # mild | moderate | severe | extreme
    size: str  # small | medium | large | very_large
    dimensionality: str  # low | medium | high
    train_rows: int
    classes: int
    majority_count: int
    minority_count: int
    features: int


def _severity(ratio: float) -> str:
    """The imbalance band of the smallest / largest class ratio, v4's thresholds."""
    if ratio >= 0.5:
        return "mild"
    if ratio >= 0.1:
        return "moderate"
    if ratio >= 0.01:
        return "severe"
    return "extreme"


def _size(rows: int) -> str:
    """The training rows' size band, v4's thresholds."""
    if rows < 1_000:
        return "small"
    if rows < 10_000:
        return "medium"
    if rows < 100_000:
        return "large"
    return "very_large"


# C901 kept: one ordered decision table (the first case that holds wins); split, the order
# of the cases would be spread over functions
def _rule(severity: str, size: str, dimensionality: str, classes: int, rows: int,  # noqa: C901
          ratio: float, minority: int, features: int) -> tuple[str, str, str, tuple[str, ...]]:
    """(recommended, reason code, reason, alternatives), in v4's order of cases."""
    severe = severity in ("severe", "extreme")
    multiclass = classes > 2
    if minority < 6 and severe:
        return ("class_weight", "SMALLEST_CLASS_TOO_SMALL_FOR_SMOTE",
                f"The smallest class has only {minority} rows, too few for SMOTE (needs 6 or more). "
                "Class weights are the safest option; consider removing very small classes.",
                ("random_undersample",))
    if severity == "mild":
        return ("none", "MILD_IMBALANCE",
                f"The classes are fairly balanced (ratio {ratio:.2%}); balancing may not be needed.",
                ("class_weight",))
    if classes > 10 and severity == "extreme":
        return ("class_weight", "MANY_CLASSES_EXTREME_IMBALANCE",
                f"{classes} classes with extreme imbalance: class weights are safest, SMOTE may "
                "fail on very small classes.", ("smote_enn", "random_undersample"))
    if multiclass and size == "very_large" and severe:
        return ("smote_enn", "MULTICLASS_VERY_LARGE_SEVERE",
                f"Large multi-class training data ({classes} classes, {rows:,} rows) with {severity} "
                "imbalance: SMOTE-ENN oversamples and removes noisy synthetic rows.",
                ("class_weight", "smote_tomek", "adasyn"))
    if multiclass and size == "large" and severe:
        return ("smote", "MULTICLASS_LARGE_SEVERE",
                f"Multi-class training data ({classes} classes) with {severity} imbalance.",
                ("class_weight", "smote_enn", "adasyn"))
    if multiclass:
        return ("class_weight", "MULTICLASS",
                f"Multi-class ({classes} classes) with {severity} imbalance: class weights are fast "
                "and add no synthetic rows.",
                ("smote", "adasyn") if size in ("large", "very_large") else ("smote",))
    if size == "small":
        return ("class_weight", "SMALL_BINARY",
                f"Small training data ({rows} rows): class weights avoid overfitting to synthetic rows.",
                ("smote",))
    if size == "very_large":
        if severe:
            return ("smote_enn", "VERY_LARGE_BINARY_SEVERE",
                    f"Large training data with {severity} imbalance ({ratio:.2%}).",
                    ("class_weight", "random_undersample", "smote_tomek"))
        return ("class_weight", "VERY_LARGE_BINARY",
                "Large training data with moderate imbalance: class weights are fast and effective.",
                ("smote", "random_undersample"))
    if dimensionality == "high":
        return ("class_weight", "HIGH_DIMENSIONAL",
                f"{features} features: synthetic rows may be unrealistic in high dimensions.",
                ("random_undersample",))
    if severity == "moderate":
        if size == "medium":
            return ("smote", "MODERATE_MEDIUM",
                    f"Moderate imbalance ({ratio:.2%}) with medium-sized training data.",
                    ("class_weight", "adasyn"))
        return ("class_weight", "MODERATE_LARGE",
                "Moderate imbalance with large training data: class weights are often enough.",
                ("smote", "smote_enn"))
    if severity == "severe":
        if dimensionality == "low":
            return ("borderline_smote", "SEVERE_LOW_DIMENSIONAL",
                    f"Severe imbalance ({ratio:.2%}) with few features: Borderline-SMOTE focuses on "
                    "the class border.", ("smote_enn", "adasyn", "class_weight"))
        return ("smote_enn", "SEVERE",
                f"Severe imbalance ({ratio:.2%}): SMOTE-ENN oversamples and cleans.",
                ("class_weight", "borderline_smote"))
    if severity == "extreme":
        return ("smote_enn", "EXTREME",
                f"Extreme imbalance (ratio {ratio:.4%}); also consider collecting more minority rows.",
                ("smote_tomek", "class_weight", "adasyn"))
    return ("class_weight", "DEFAULT", "Class weights are a safe, fast choice.", ("smote",))


def recommend_balancing(train_labels: pd.Series, num_features: int) -> BalancingRecommendation:
    """Recommend a balancing method from the **training split's** labels."""
    counts = train_labels.dropna().astype(str).value_counts()
    if counts.empty:
        raise ValueError("No labelled training rows")
    majority, minority = int(counts.iloc[0]), int(counts.iloc[-1])
    ratio = minority / majority
    rows = int(counts.sum())
    severity, size = _severity(ratio), _size(rows)
    dimensionality = "low" if num_features < 20 else "medium" if num_features < 100 else "high"
    recommended, code, reason, alternatives = _rule(
        severity, size, dimensionality, len(counts), rows, ratio, minority, num_features)
    return BalancingRecommendation(
        recommended=recommended, reason_code=code, reason=reason, alternatives=alternatives,
        imbalance_ratio=round(ratio, 4), severity=severity, size=size,
        dimensionality=dimensionality, train_rows=rows, classes=len(counts),
        majority_count=majority, minority_count=minority, features=num_features,
    )
