"""Evaluation: CV on train, model chosen before the test set is seen, test scored once
(plan D9, D10; L8, L9, F2 basic).

Order, so the test split can't influence any choice:

1. balancing recommendation (if asked for) from the training labels;
2. stratified K-fold CV on the training split (a stratified sample of it when large);
3. final fit of every model on the whole training split;
4. selection: best mean CV macro-F1 (ties: first name);
5. each model scores the test split exactly once.

Metrics are the ones that stay honest on imbalanced IDS data: macro-F1 and balanced
accuracy lead; accuracy and weighted F1 are shown for comparison; per-class precision,
recall, F1 and the confusion matrix show which attacks are missed.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

from ..data.settings import EvaluationConfig, PipelineConfig, max_workers
from ..ports import NullProgress, ProgressSink
from . import balancing, pipeline, registry
from .sampling import stratified_positions

SELECTION_METRIC = "cv_macro_f1"


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # rows of this class in the scored data


@dataclass(frozen=True)
class Scores:
    rows: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    labels: tuple[str, ...]  # order of per_class and of the confusion matrix
    per_class: tuple[ClassMetrics, ...]
    confusion: tuple[tuple[int, ...], ...]  # rows: true label, columns: predicted


@dataclass(frozen=True)
class CVSummary:
    folds: int
    macro_f1: tuple[float, ...]  # one value per fold
    balanced_accuracy: tuple[float, ...]
    accuracy: tuple[float, ...]
    recall_per_class: tuple[tuple[str, float], ...]  # mean over folds

    @property
    def macro_f1_mean(self) -> float:
        return float(np.mean(self.macro_f1))

    @property
    def macro_f1_std(self) -> float:
        return float(np.std(self.macro_f1))

    @property
    def balanced_accuracy_mean(self) -> float:
        return float(np.mean(self.balanced_accuracy))

    @property
    def balanced_accuracy_std(self) -> float:
        return float(np.std(self.balanced_accuracy))

    @property
    def accuracy_mean(self) -> float:
        return float(np.mean(self.accuracy))


@dataclass(frozen=True)
class ModelEvaluation:
    name: str
    cv: CVSummary
    test: Scores
    fit_seconds: float  # final fit on the whole training split
    predict_seconds: float  # the whole test split


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    model: str
    title: str
    selected: bool
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    cv_balanced_accuracy_mean: float
    cv_balanced_accuracy_std: float
    test_macro_f1: float
    test_balanced_accuracy: float
    test_accuracy: float
    test_min_recall: float  # weakest class in test
    test_recall_per_class: tuple[tuple[str, float], ...]  # classes present in test
    fit_seconds: float
    predict_seconds: float


@dataclass(frozen=True, eq=False)
class EvaluationResult:
    models: tuple[ModelEvaluation, ...]  # leaderboard order
    selected: str | None
    selection_metric: str
    leaderboard: tuple[LeaderboardRow, ...]
    cv_folds: int
    cv_rows: int
    cv_sampled: bool
    balancer: str
    balancer_recommendation: balancing.BalancingRecommendation | None
    selector: str
    k_features: int | None
    failures: tuple[pipeline.ModelFailure, ...]
    skipped: tuple[tuple[str, str], ...]
    classes_train: tuple[str, ...]
    classes_only_in_test: tuple[str, ...]
    training: pipeline.TrainingOutcome  # the fitted models (whole training split)
    notes: tuple[str, ...]
    seed: int
    workers: int


def _quiet(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """sklearn warns when a fold/test lacks a class the model predicts; that's expected."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return func(*args, **kwargs)


def score(truth: np.ndarray, predicted: np.ndarray, labels: Sequence[str]) -> Scores:
    """All metrics for one prediction; ``labels`` fixes the per-class order."""
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0)
    return Scores(
        rows=len(truth),
        accuracy=float(accuracy_score(truth, predicted)),
        balanced_accuracy=float(_quiet(balanced_accuracy_score, truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        labels=tuple(labels),
        per_class=tuple(ClassMetrics(label, float(p), float(r), float(f), int(s))
                        for label, p, r, f, s in zip(labels, precision, recall, f1, support,
                                                     strict=True)),
        confusion=tuple(tuple(int(v) for v in row)
                        for row in confusion_matrix(truth, predicted, labels=labels)),
    )


def _summary(fold_scores: Sequence[Scores]) -> CVSummary:
    labels = sorted({c.label for s in fold_scores for c in s.per_class})
    recall = {label: [c.recall for s in fold_scores for c in s.per_class
                      if c.label == label and c.support] for label in labels}
    return CVSummary(
        folds=len(fold_scores),
        macro_f1=tuple(s.macro_f1 for s in fold_scores),
        balanced_accuracy=tuple(s.balanced_accuracy for s in fold_scores),
        accuracy=tuple(s.accuracy for s in fold_scores),
        recall_per_class=tuple((label, float(np.mean(v))) for label, v in recall.items() if v),
    )


def select_model(summaries: Mapping[str, CVSummary]) -> tuple[str, ...]:
    """Models ranked by mean CV macro-F1 (ties: first name). Sees CV results only."""
    return tuple(sorted(summaries, key=lambda name: (-summaries[name].macro_f1_mean, name)))


class _TrainOptions(TypedDict):
    """The ``pipeline.train_models`` options shared by the CV folds and the final fit."""

    balancer: str
    selector: str
    k_features: int | None
    config: PipelineConfig
    workers: int


def _cross_validate(frame: pd.DataFrame, label: str, features: Sequence[str],
                    names: Sequence[str], train_options: _TrainOptions,
                    config: EvaluationConfig, notes: list[str], progress: ProgressSink,
                    ) -> tuple[dict[str, CVSummary], dict[str, pipeline.ModelFailure]]:
    labels = frame[label].astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    if (counts < config.cv_folds).any():
        notes.append("CV_CLASS_SMALLER_THAN_FOLDS")
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True,
                            random_state=config.random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        splits = list(folds.split(np.zeros(len(labels)), labels))
    fold_scores: dict[str, list[Scores]] = {name: [] for name in names}
    failed: dict[str, pipeline.ModelFailure] = {}
    for fold, (fit_rows, check_rows) in enumerate(splits):
        remaining = [n for n in names if n not in failed]
        if not remaining:  # every model failed: count the folds left as done
            progress.advance(len(names) * (len(splits) - fold))
            break
        if len(remaining) < len(names):
            progress.advance(len(names) - len(remaining))  # models that failed earlier
        outcome = pipeline.train_models(frame.iloc[fit_rows], label, features=features,
                                        models=remaining, progress=progress, **train_options)
        failed.update({f.name: f for f in outcome.failures})
        held_out = frame.iloc[check_rows]
        truth = labels[check_rows]
        fold_labels = sorted(set(outcome.classes) | set(truth))
        for name, fitted in outcome.models.items():
            fold_scores[name].append(score(truth, fitted.predict(held_out), fold_labels))
    summaries = {name: _summary(s) for name, s in fold_scores.items()
                 if name not in failed and len(s) == len(splits)}
    return summaries, failed


def _row(rank: int, evaluation: ModelEvaluation, selected: str | None) -> LeaderboardRow:
    present = [c for c in evaluation.test.per_class if c.support]
    recall = tuple((c.label, c.recall) for c in present)
    return LeaderboardRow(
        rank=rank, model=evaluation.name, title=registry.MODELS[evaluation.name].title,
        selected=evaluation.name == selected,
        cv_macro_f1_mean=evaluation.cv.macro_f1_mean, cv_macro_f1_std=evaluation.cv.macro_f1_std,
        cv_balanced_accuracy_mean=evaluation.cv.balanced_accuracy_mean,
        cv_balanced_accuracy_std=evaluation.cv.balanced_accuracy_std,
        test_macro_f1=evaluation.test.macro_f1,
        test_balanced_accuracy=evaluation.test.balanced_accuracy,
        test_accuracy=evaluation.test.accuracy,
        test_min_recall=min((v for _, v in recall), default=0.0),
        test_recall_per_class=recall,
        fit_seconds=evaluation.fit_seconds, predict_seconds=evaluation.predict_seconds,
    )


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
    *,
    features: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    balancer: str = "none",
    selector: str = "none",
    k_features: int | None = None,
    config: EvaluationConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    workers: int | None = None,
    progress: ProgressSink | None = None,
) -> EvaluationResult:
    """Cross-validate, fit, select and test; see the module docstring.

    ``balancer="recommended"`` asks ``balancing.recommend_balancing`` (training labels only).
    ``progress`` gets one step per model per CV fold and per model for the final fit
    (``models x (folds + 1)``), also for models that failed or were skipped.
    """
    config = config or EvaluationConfig()
    progress = progress or NullProgress()
    pipeline_config = pipeline_config or PipelineConfig()
    workers = workers or max_workers()
    features = tuple(features if features is not None else (c for c in train.columns if c != label))
    names = tuple(models) if models else registry.default_models()
    train = train[train[label].notna()]
    test = test[test[label].notna()]
    notes: list[str] = []

    recommendation = None
    if balancer == "recommended":
        recommendation = balancing.recommend_balancing(train[label], len(features))
        balancer = recommendation.recommended
    options: _TrainOptions = {"balancer": balancer, "selector": selector,
                              "k_features": k_features, "config": pipeline_config,
                              "workers": workers}

    cv_frame = train
    if len(train) > config.cv_max_rows:
        positions = stratified_positions(train[label].astype(str).to_numpy(), config.cv_max_rows,
                                         config.cv_folds, config.random_state)
        cv_frame = train.iloc[positions]
    summaries, cv_failed = _cross_validate(cv_frame, label, features, names, options, config, notes,
                                           progress)

    final_names = [n for n in names if n in summaries]
    if len(final_names) < len(names):
        progress.advance(len(names) - len(final_names))  # no final fit for these
    final = pipeline.train_models(train, label, features=features, models=final_names,
                                  progress=progress, **options)
    failures = {**cv_failed, **{f.name: f for f in final.failures}}
    ranked = [n for n in select_model(summaries) if n in final.models]
    selected = ranked[0] if ranked else None

    truth = test[label].astype(str).to_numpy()
    test_labels = sorted(set(final.classes) | set(truth))
    evaluations = []
    for name in ranked:
        start = time.perf_counter()
        predicted = final.models[name].predict(test)
        seconds = time.perf_counter() - start
        evaluations.append(ModelEvaluation(name, summaries[name], score(truth, predicted, test_labels),
                                           final.models[name].fit_seconds, seconds))

    return EvaluationResult(
        models=tuple(evaluations),
        selected=selected,
        selection_metric=SELECTION_METRIC,
        leaderboard=tuple(_row(i, e, selected) for i, e in enumerate(evaluations, start=1)),
        cv_folds=config.cv_folds,
        cv_rows=len(cv_frame),
        cv_sampled=len(cv_frame) < len(train),
        balancer=balancer,
        balancer_recommendation=recommendation,
        selector=selector,
        k_features=k_features,
        failures=tuple(failures[n] for n in names if n in failures),
        skipped=final.skipped,
        classes_train=final.classes,
        classes_only_in_test=tuple(sorted(set(truth) - set(final.classes))),
        training=final,
        notes=tuple(dict.fromkeys([*notes, *final.notes])),
        seed=config.random_state,
        workers=workers,
    )


def leaderboard_table(result: EvaluationResult) -> str:
    """Plain-text leaderboard (F2 basic); ``*`` marks the selected model."""
    header = ("Rank  Model                 CV macro-F1      CV balanced acc  Test macro-F1  "
              "Test balanced acc  Test min recall  Fit s    Predict s")
    lines = [header, "-" * len(header)]
    for r in result.leaderboard:
        mark = "*" if r.selected else " "
        lines.append(
            f"{r.rank:>3}{mark}  {r.model:<20}  {r.cv_macro_f1_mean:.3f} ± {r.cv_macro_f1_std:.3f}"
            f"    {r.cv_balanced_accuracy_mean:.3f} ± {r.cv_balanced_accuracy_std:.3f}    "
            f"{r.test_macro_f1:>13.3f}  {r.test_balanced_accuracy:>17.3f}  "
            f"{r.test_min_recall:>15.3f}  {r.fit_seconds:>7.1f}  {r.predict_seconds:>9.2f}")
    lines.append(f"Selected by {result.selection_metric} (CV on {result.cv_rows:,} training rows, "
                 f"{result.cv_folds} folds); the test split was scored once per model.")
    return "\n".join(lines)
