"""What the data functions step 2.3 refactors produce for a set of files, as plain JSON data
(v5.2 step 2.3a): the loader's report, column types and a hash of the frame; the split
(strategy, reason codes, warnings, the rows on each side, coverage, dedupe and time check);
the leakage report and the balancing recommendation on the training split; and
``balancing._rule`` for every combination of its inputs.

``tests/test_goldens.py`` compares today's results with ``tests/fixtures/golden/outputs.json``,
written once by ``tests/fixtures/golden/make_goldens.py`` from the code before the refactor.
The conversion to JSON is written here, not taken from the code under test.
"""

import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:  # the v5.2 layout (step 2.3b and later)
    from toolkit.core.services.ids import balancing
    from toolkit.core.services.ids.datasets import load_dataset
    from toolkit.core.services.ids.leakage import check_leakage
    from toolkit.core.services.ids.split import split_dataset
except ModuleNotFoundError:  # the v5.1 layout
    from toolkit.services.ids import balancing
    from toolkit.services.ids.datasets import load_dataset
    from toolkit.services.ids.leakage import check_leakage
    from toolkit.services.ids.split import split_dataset

FIXTURES = ("cic_ddos_sample.csv", "cic2017_original_sample.csv", "nslkdd_train_sample.txt",
            "nslkdd_test_sample_noheader.txt", "label_leak.csv")
OFFICIAL = ("nslkdd_train_sample.txt", "nslkdd_test_sample_noheader.txt")


def plain(value):
    """JSON data: dataclasses as dicts, numpy scalars as numbers, tuples as lists; floats
    that are not finite as strings (JSON has no NaN)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def frame_hash(frame: pd.DataFrame) -> str:
    """Values, index and column names (types are recorded separately)."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update("\0".join(map(str, frame.columns)).encode("utf-8"))
    return digest.hexdigest()


def _rows(frame: pd.DataFrame) -> dict:
    index = np.asarray(frame.index, dtype=np.int64)
    return {"rows": len(index), "index_sha256": hashlib.sha256(index.tobytes()).hexdigest()}


def _split(result) -> dict:
    data = {name: plain(getattr(result, name)) for name in (
        "strategy", "requested", "profile_strategy", "overridden", "reason_codes", "warnings",
        "time_column", "group_column", "time_check", "official_mode", "dedupe", "coverage",
        "rows_without_label", "seed", "test_size")}
    return {**data, "train": _rows(result.train), "test": _rows(result.test)}


def _after_split(result, loaded) -> dict:
    label = loaded.profile.label_column
    features = [c for c in result.train.columns if c != label]
    leakage = check_leakage(result.train, label, profile_blocked=loaded.profile.blocked_columns)
    recommendation = balancing.recommend_balancing(result.train[label], len(features))
    return {"leakage": plain(dataclasses.replace(leakage, config=None)),
            "balancing": plain(recommendation)}


def file_outputs(path: Path) -> dict:
    loaded = load_dataset(path)
    label = loaded.profile.label_column
    out = {"profile": loaded.profile.profile.name, "variant": loaded.profile.variant,
           "report": plain(dataclasses.replace(loaded.report, sha256="")),
           "dtypes": {str(c): str(t) for c, t in loaded.frame.dtypes.items()},
           "frame_sha256": frame_hash(loaded.frame)}
    default = split_dataset(loaded.frame, label, profile=loaded.profile)
    out["split_default"] = _split(default)
    out["split_stratified"] = _split(split_dataset(loaded.frame, label, profile=loaded.profile,
                                                   strategy="stratified"))
    out.update(_after_split(default, loaded))
    return out


def official_outputs(train_path: Path, test_path: Path) -> dict:
    train, test = load_dataset(train_path), load_dataset(test_path)
    label = train.profile.label_column
    return _split(split_dataset(train.frame, label, profile=train.profile, test_frame=test.frame))


def rule_table() -> list:
    """``balancing._rule`` for every combination of its inputs (one row each)."""
    ratios = {"mild": 0.6, "moderate": 0.2, "severe": 0.05, "extreme": 0.005}
    rows = {"small": 500, "medium": 5_000, "large": 50_000, "very_large": 500_000}
    features = {"low": 10, "medium": 50, "high": 200}
    table = []
    for severity, ratio in ratios.items():
        for size, count in rows.items():
            for dimensionality, width in features.items():
                for classes in (2, 3, 11):
                    for minority in (5, 50):
                        answer = balancing._rule(severity, size, dimensionality, classes, count,
                                                 ratio, minority, width)
                        table.append([severity, size, dimensionality, classes, minority,
                                      plain(answer)])
    return table


def outputs(fixtures_dir: Path, names=FIXTURES, official=OFFICIAL) -> dict:
    data = {"files": {name: file_outputs(fixtures_dir / name) for name in names}}
    if official:
        data["official"] = official_outputs(fixtures_dir / official[0], fixtures_dir / official[1])
    data["rule_table"] = rule_table()
    return data
