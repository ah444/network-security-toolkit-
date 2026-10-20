"""Models, balancers and feature selectors as registries (plan §6.1, replaces v4's if/elif chains).

Each entry says how to build a fresh estimator from a ``BuildContext`` (seed, workers,
training class counts), so the pipeline never needs to know a model by name. Every
random estimator gets the seed, nested samplers included (B20), and every parallel one
gets ``workers`` instead of ``n_jobs=-1`` (D8). Add a model by adding an entry.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class BuildContext:
    seed: int
    workers: int
    class_counts: Mapping[int, int]  # training rows per encoded class
    class_weight: bool = False  # the "class_weight" balancer was chosen
    k_features: int | None = None
    smote_max_neighbors: int = 5

    @property
    def oversample_targets(self) -> dict[int, int]:
        """Classes to grow to the majority's size; single-row classes can't be interpolated."""
        biggest = max(self.class_counts.values())
        return {c: biggest for c, n in self.class_counts.items() if 2 <= n < biggest}

    @property
    def smote_neighbors(self) -> int:
        targets = self.oversample_targets
        smallest = min((self.class_counts[c] for c in targets), default=2)
        return max(1, min(self.smote_max_neighbors, smallest - 1))

    @property
    def balanced(self) -> str | None:
        return "balanced" if self.class_weight else None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    description: str
    build: Callable[[BuildContext], Any]
    scale: bool  # numeric features: signed log + standard scaling (linear / distance models)
    encoding: str  # categorical features: "onehot" (linear / distance) | "ordinal" (trees)
    class_weight: str  # how "class_weight" is honoured: "param" | "sample_weight" | "none"
    default: bool = False
    requires: str | None = None  # optional package
    max_train_rows: str | None = None  # name of the PipelineConfig limit, if any


@dataclass(frozen=True)
class BalancerSpec:
    name: str
    title: str
    description: str
    build: Callable[[BuildContext], Any | None]  # an imblearn sampler, or None
    oversamples: bool = False
    class_weight: bool = False


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    title: str
    build: Callable[[BuildContext], Any | None]


def _xgboost(ctx: BuildContext):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=100, random_state=ctx.seed, n_jobs=ctx.workers, verbosity=0)


def _lightgbm(ctx: BuildContext):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=100, random_state=ctx.seed, n_jobs=ctx.workers, verbose=-1,
                          class_weight=ctx.balanced)


MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in (
    ModelSpec("random_forest", "Random Forest", "Ensemble of decision trees, good for most cases",
              lambda ctx: RandomForestClassifier(n_estimators=100, random_state=ctx.seed,
                                                 n_jobs=ctx.workers, class_weight=ctx.balanced),
              scale=False, encoding="ordinal", class_weight="param", default=True),
    ModelSpec("logistic_regression", "Logistic Regression", "Fast and interpretable linear model",
              lambda ctx: LogisticRegression(max_iter=1000, random_state=ctx.seed,
                                             class_weight=ctx.balanced),
              scale=True, encoding="onehot", class_weight="param", default=True),
    ModelSpec("decision_tree", "Decision Tree", "Simple and interpretable tree-based model",
              lambda ctx: DecisionTreeClassifier(random_state=ctx.seed, class_weight=ctx.balanced),
              scale=False, encoding="ordinal", class_weight="param"),
    ModelSpec("knn", "K-Nearest Neighbors", "Instance-based learning, good for smaller datasets",
              lambda ctx: KNeighborsClassifier(n_neighbors=5, n_jobs=ctx.workers),
              scale=True, encoding="onehot", class_weight="none"),
    ModelSpec("naive_bayes", "Naive Bayes", "Fast probabilistic classifier",
              lambda ctx: GaussianNB(), scale=True, encoding="onehot", class_weight="sample_weight"),
    ModelSpec("svm", "Support Vector Machine (linear)",
              "Effective for high-dimensional data (slower on large datasets)",
              lambda ctx: LinearSVC(random_state=ctx.seed, max_iter=1000, class_weight=ctx.balanced),
              scale=True, encoding="onehot", class_weight="param", max_train_rows="svm_max_train_rows"),
    ModelSpec("gradient_boosting", "Gradient Boosting", "Powerful boosting algorithm, slower training",
              lambda ctx: GradientBoostingClassifier(n_estimators=100, random_state=ctx.seed),
              scale=False, encoding="ordinal", class_weight="sample_weight"),
    ModelSpec("adaboost", "AdaBoost", "Adaptive boosting, focuses on hard examples",
              lambda ctx: AdaBoostClassifier(n_estimators=100, random_state=ctx.seed),
              scale=False, encoding="ordinal", class_weight="sample_weight"),
    ModelSpec("extra_trees", "Extra Trees", "Similar to Random Forest with more randomness",
              lambda ctx: ExtraTreesClassifier(n_estimators=100, random_state=ctx.seed,
                                               n_jobs=ctx.workers, class_weight=ctx.balanced),
              scale=False, encoding="ordinal", class_weight="param"),
    ModelSpec("xgboost", "XGBoost", "Optimized gradient boosting, very powerful", _xgboost,
              scale=False, encoding="ordinal", class_weight="sample_weight", requires="xgboost"),
    ModelSpec("lightgbm", "LightGBM", "Fast gradient boosting by Microsoft", _lightgbm,
              scale=False, encoding="ordinal", class_weight="param", requires="lightgbm"),
)}


def _smote(ctx: BuildContext) -> SMOTE:
    return SMOTE(random_state=ctx.seed, k_neighbors=ctx.smote_neighbors,
                 sampling_strategy=ctx.oversample_targets)


def _oversampler(build: Callable[[BuildContext], Any]) -> Callable[[BuildContext], Any | None]:
    """Nothing to grow (all classes equal, or only single-row minorities): no sampler."""
    return lambda ctx: build(ctx) if ctx.oversample_targets else None


BALANCERS: dict[str, BalancerSpec] = {spec.name: spec for spec in (
    BalancerSpec("none", "None", "Train on the data as it is", lambda ctx: None),
    BalancerSpec("class_weight", "Class weights", "Weight rare classes up instead of adding rows",
                 lambda ctx: None, class_weight=True),
    BalancerSpec("smote", "SMOTE", "Synthetic rows for minority classes", _oversampler(_smote),
                 oversamples=True),
    BalancerSpec("smote_enn", "SMOTE + ENN", "SMOTE, then remove noisy rows",
                 _oversampler(lambda ctx: SMOTEENN(random_state=ctx.seed, smote=_smote(ctx),
                                                   n_jobs=ctx.workers)), oversamples=True),
    BalancerSpec("adasyn", "ADASYN", "Synthetic rows where the classes are hardest to separate",
                 _oversampler(lambda ctx: ADASYN(random_state=ctx.seed,
                                                 n_neighbors=ctx.smote_neighbors,
                                                 sampling_strategy=ctx.oversample_targets)),
                 oversamples=True),
    BalancerSpec("borderline_smote", "Borderline SMOTE", "SMOTE near the class borders",
                 _oversampler(lambda ctx: BorderlineSMOTE(random_state=ctx.seed,
                                                          k_neighbors=ctx.smote_neighbors,
                                                          sampling_strategy=ctx.oversample_targets)),
                 oversamples=True),
    BalancerSpec("random_undersample", "Random undersampling", "Drop rows of the bigger classes",
                 lambda ctx: RandomUnderSampler(random_state=ctx.seed)),
    BalancerSpec("tomek", "Tomek links", "Remove borderline pairs of different classes",
                 lambda ctx: TomekLinks(n_jobs=ctx.workers)),
    BalancerSpec("smote_tomek", "SMOTE + Tomek", "SMOTE, then remove Tomek links",
                 _oversampler(lambda ctx: SMOTETomek(random_state=ctx.seed, smote=_smote(ctx),
                                                     tomek=TomekLinks(n_jobs=ctx.workers),
                                                     n_jobs=ctx.workers)), oversamples=True),
)}


class _Scores:
    """Feature scores for SelectKBest; constant features score lowest without warnings."""

    def __init__(self, method: str, seed: int):
        self.method = method
        self.seed = seed

    def __call__(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        varies = np.ptp(X, axis=0) > 0
        scores = np.full(X.shape[1], np.nan)
        if varies.any():
            if self.method == "f_classif":
                found = f_classif(X[:, varies], y)[0]
            else:
                found = mutual_info_classif(X[:, varies], y, random_state=self.seed)
            scores[varies] = np.nan_to_num(found, nan=0.0, posinf=np.finfo(np.float64).max)
        return scores


class _TopK(SelectKBest):
    """SelectKBest that quietly keeps every feature when k is larger than their number
    (one-hot encoding decides the number only at fit time)."""

    def _check_params(self, X, y):
        pass


def _top_k(method: str):
    return lambda ctx: _TopK(_Scores(method, ctx.seed), k=ctx.k_features)


SELECTORS: dict[str, SelectorSpec] = {spec.name: spec for spec in (
    SelectorSpec("none", "All chosen features", lambda ctx: None),
    SelectorSpec("f_classif", "ANOVA F-test (linear)", _top_k("f_classif")),
    SelectorSpec("mutual_info", "Mutual information (non-linear)", _top_k("mutual_info")),
)}


def is_available(spec: ModelSpec) -> bool:
    return spec.requires is None or importlib.util.find_spec(spec.requires) is not None


def available_models() -> tuple[str, ...]:
    return tuple(name for name, spec in MODELS.items() if is_available(spec))


def default_models() -> tuple[str, ...]:
    return tuple(name for name, spec in MODELS.items() if spec.default and is_available(spec))
