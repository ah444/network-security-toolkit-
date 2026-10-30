"""Training pipeline fitted on the training split only (plan D9; L2, L3, L7, B1, B5, B20, D8).

One ``imblearn`` Pipeline per model, so every learned step sees training rows only and
is saved together with the model:

    prepare (inf -> missing, impute, [signed log, standardise] numeric; encode categorical)
      -> select (optional) -> sample (optional, skipped at prediction) -> model

Linear / distance models get ``sign(x) * log1p(|x|)`` then standard scaling: flow features
are heavy-tailed (bytes/s up to 1e9), and after v4's RobustScaler logistic regression
never converged on CIC (balanced accuracy 0.144 vs 0.691, see the 0.4c checklist).
Encoding follows the model: one-hot for linear / distance models (no false ordering),
ordinal for trees; unseen categories at prediction time never crash. Models are
trained one after another, each with ``workers`` threads; native thread pools are
capped too. A model that fails is reported and left out, never returned unfitted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from ...data.settings import PipelineConfig, max_workers
from ..ports import NullProgress, ProgressSink
from . import registry

log = logging.getLogger(__name__)

STEPS = ("prepare", "select", "sample", "model")


@dataclass(frozen=True, eq=False)
class FittedModel:
    """One trained pipeline (prepare, select, sample, model) with what is needed to use it: its
    input ``features`` in order, the label text of its class numbers, how categories were encoded,
    whether numbers were scaled, and the fit time. Pickled in bundles, so its module and name must
    not move without an alias."""
    name: str
    pipeline: Pipeline
    features: tuple[str, ...]
    classes: tuple[str, ...]  # label text for the model's class numbers 0..n-1
    encoding: str
    scaled: bool
    fit_seconds: float
    notes: tuple[str, ...] = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """The predicted label of each row of ``frame`` (only ``features`` are used)."""
        codes = np.asarray(self.pipeline.predict(frame[list(self.features)]), dtype=int)
        return cast(np.ndarray, np.asarray(self.classes, dtype=object)[codes])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Class probabilities, columns in ``classes`` order; None if the model has none."""
        if not hasattr(self.pipeline, "predict_proba"):
            return None
        return np.asarray(self.pipeline.predict_proba(frame[list(self.features)]))

    def feature_schema(self) -> tuple[FeatureInfo, ...]:
        """Each input feature as the fitted pipeline sees it (for prediction forms)."""
        kinds: dict[str, FeatureInfo] = {}
        for part, transformer, columns in self.pipeline.named_steps["prepare"].transformers_:
            if part == "numeric":
                kinds.update((c, FeatureInfo(c, "numeric")) for c in columns)
            elif part == "categorical":
                for column, values in zip(columns, transformer.categories_, strict=True):
                    known = tuple(str(v) for v in values if not pd.isna(v))
                    kinds[column] = FeatureInfo(column, "categorical", known)
        return tuple(kinds[name] for name in self.features)

    def input_frame(self, values: Mapping[str, Any]) -> pd.DataFrame:
        """One row from form values (text or numbers); raises ValueError naming the field.
        Unknown categories are allowed: the encoder handles them (L7)."""
        row: dict[str, list[Any]] = {}
        for info in self.feature_schema():
            value = values.get(info.name)
            text = "" if value is None else str(value).strip()
            if not text:
                raise ValueError(f"Enter a value for {info.name!r}")
            if info.kind == "numeric":
                try:
                    row[info.name] = [float(text)]
                except ValueError:
                    raise ValueError(f"{info.name!r} must be a number") from None
            else:
                row[info.name] = [text]
        return pd.DataFrame(row, columns=list(self.features))


@dataclass(frozen=True)
class FeatureInfo:
    """An input feature as a prediction form shows it: its kind and, if any, the categories."""
    name: str
    kind: str  # "numeric" | "categorical"
    categories: tuple[str, ...] = ()  # categorical: the values seen in training


@dataclass(frozen=True)
class ModelFailure:
    """A model that failed to train, and the error's message."""
    name: str
    error: str


@dataclass(frozen=True, eq=False)
class TrainingOutcome:
    """What ``train_models`` produced: the models that trained, the ones that failed or were skipped
    (with the reason code), the classes and columns, the training rows per class, the balancing and
    selection used, the seed, the workers and notes."""
    models: dict[str, FittedModel]  # only models that trained successfully
    failures: tuple[ModelFailure, ...]
    skipped: tuple[tuple[str, str], ...]  # (model, reason code)
    classes: tuple[str, ...]
    features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    class_counts: tuple[tuple[str, int], ...]  # training rows per class
    balancer: str
    selector: str
    k_features: int | None
    seed: int
    workers: int
    train_rows: int
    notes: tuple[str, ...]


def _finite(values: Any) -> np.ndarray:
    """inf -> missing, so the imputer handles both (module level: bundles pickle it).

    float32 halves the memory of every later step (2 M CIC rows: float64 needed more than
    an 8 GB laptop has). Tree models (sklearn, XGBoost, LightGBM) use float32 internally
    anyway; linear / distance models lose only digits beyond the 7th.
    """
    array = np.array(values, dtype=np.float32)
    array[~np.isfinite(array)] = np.nan
    return array


def _signed_log(values: Any) -> np.ndarray:
    """sign(x) * log1p(|x|): keeps the sign and order, squeezes heavy tails."""
    array = np.asarray(values)
    return cast(np.ndarray, np.sign(array) * np.log1p(np.abs(array)))


def _prepare(numeric: Sequence[str], categorical: Sequence[str],
             spec: registry.ModelSpec) -> ColumnTransformer:
    """The pipeline's first step for ``spec``: numbers made finite (inf → missing), the median
    imputed and, for linear and distance models, a signed log and scaling; categories one-hot
    (unknown ones ignored) or ordinal (unknown -1, missing -2)."""
    parts = []
    if numeric:
        steps = [("finite", FunctionTransformer(_finite, feature_names_out="one-to-one")),
                 ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if spec.scale:
            steps += [("log", FunctionTransformer(_signed_log, feature_names_out="one-to-one")),
                      ("scale", StandardScaler())]
        parts.append(("numeric", SkPipeline(steps), list(numeric)))
    if categorical:
        if spec.encoding == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                    dtype=np.float32)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                     encoded_missing_value=-2, dtype=np.float32)
        parts.append(("categorical", encoder, list(categorical)))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)


def _check_names(train: pd.DataFrame, label: str, features: Sequence[str], models: Sequence[str],
                 balancer: str, selector: str, k_features: int | None) -> None:
    """ValueError with the message for the first problem of the request: the label among the
    features, a missing column, no feature, an unknown model, balancer or selector, or selection
    without ``k_features``."""
    if label in features:
        raise ValueError(f"The label column {label!r} can't be a feature")
    missing = [c for c in (*features, label) if c not in train.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    if not features:
        raise ValueError("No features chosen")
    unknown = [m for m in models if m not in registry.MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")
    if balancer not in registry.BALANCERS:
        raise ValueError(f"Unknown balancing method: {balancer!r}")
    if selector not in registry.SELECTORS:
        raise ValueError(f"Unknown feature selector: {selector!r}")
    if selector != "none" and not k_features:
        raise ValueError("Feature selection needs k_features >= 1")


def train_models(
    train: pd.DataFrame,
    label: str,
    *,
    features: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    balancer: str = "none",
    selector: str = "none",
    k_features: int | None = None,
    config: PipelineConfig | None = None,
    workers: int | None = None,
    progress: ProgressSink | None = None,
) -> TrainingOutcome:
    """Fit one pipeline per model on the **training split**; see the module docstring.

    ``progress.advance()`` is called once per model (fitted, failed or skipped), outside
    the per-model error handling: an exception it raises (a cancelled job) stops the
    training and is never recorded as a model failure."""
    config = config or PipelineConfig()
    progress = progress or NullProgress()
    workers = workers or max_workers()
    features = tuple(features if features is not None else (c for c in train.columns if c != label))
    # None means the default models; an empty list means none (every model failed in CV).
    names = tuple(models) if models is not None else registry.default_models()
    _check_names(train, label, features, names, balancer, selector, k_features)

    data = train[train[label].notna()]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[label].astype(str))
    classes = tuple(str(c) for c in label_encoder.classes_)
    if len(classes) < 2:
        raise ValueError("Training needs at least two classes")
    X = data[list(features)]
    numeric = tuple(c for c in features if X[c].dtype.kind in "iufb")
    categorical = tuple(c for c in features if c not in numeric)
    counts = np.bincount(y)

    balancer_spec = registry.BALANCERS[balancer]
    selector_spec = registry.SELECTORS[selector]
    context = registry.BuildContext(
        seed=config.random_state, workers=workers,
        class_counts={i: int(n) for i, n in enumerate(counts)},
        class_weight=balancer_spec.class_weight, k_features=k_features,
        smote_max_neighbors=config.smote_max_neighbors,
    )
    notes: list[str] = []
    if balancer_spec.oversamples and (counts < 2).any():
        notes.append("SAMPLER_SKIPPED_TINY_CLASSES")

    fitted: dict[str, FittedModel] = {}
    failures: list[ModelFailure] = []
    skipped: list[tuple[str, str]] = []
    for name in names:
        spec = registry.MODELS[name]
        if not registry.is_available(spec):
            skipped.append((name, "MODEL_NOT_INSTALLED"))
            progress.advance()
            continue
        limit = getattr(config, spec.max_train_rows) if spec.max_train_rows else None
        if limit is not None and len(data) > limit:
            skipped.append((name, f"{name.upper()}_TRAIN_TOO_LARGE"))
            progress.advance()
            continue
        model_notes: list[str] = []
        fit_params = {}
        if context.class_weight and spec.class_weight == "sample_weight":
            fit_params["model__sample_weight"] = compute_sample_weight("balanced", y)
            model_notes.append("CLASS_WEIGHT_AS_SAMPLE_WEIGHT")
        elif context.class_weight and spec.class_weight == "none":
            model_notes.append("CLASS_WEIGHT_UNSUPPORTED")
        try:
            pipeline = Pipeline([
                ("prepare", _prepare(numeric, categorical, spec)),
                ("select", selector_spec.build(context) or "passthrough"),
                ("sample", balancer_spec.build(context) or "passthrough"),
                ("model", spec.build(context)),
            ]).set_output(transform="pandas")  # named columns reach the model (and explanations)
            start = time.perf_counter()
            with threadpool_limits(limits=workers):
                pipeline.fit(X, y, **fit_params)
            seconds = time.perf_counter() - start
        except Exception as error:  # one model failing must not stop the others (B1)
            log.exception("Training %s failed", name)
            failures.append(ModelFailure(name, f"{type(error).__name__}: {error}"))
            progress.advance()
            continue
        log.info("Trained %s in %.1f s", name, seconds)
        fitted[name] = FittedModel(name, pipeline, features, classes, spec.encoding, spec.scale,
                                   seconds, tuple(model_notes))
        progress.advance()

    return TrainingOutcome(
        models=fitted,
        failures=tuple(failures),
        skipped=tuple(skipped),
        classes=classes,
        features=features,
        numeric_features=numeric,
        categorical_features=categorical,
        class_counts=tuple(zip(classes, (int(n) for n in counts), strict=True)),
        balancer=balancer,
        selector=selector,
        k_features=k_features,
        seed=config.random_state,
        workers=workers,
        train_rows=len(data),
        notes=tuple(notes),
    )
