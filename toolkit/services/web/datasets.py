"""The web trainer's dataset work: uploads, combining files, the class filter, and the
training split the analyses look at (the training split only, D9)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from ...data.settings import SplitConfig
from ...data.web import uploads
from ..ids.datasets import LoadedDataset, load_dataset
from ..ids.runs import DatasetRef
from ..ids.split import SplitResult, split_dataset
from .errors import NO_DATASET, UNKNOWN_COLUMN, ServiceError
from .workspace import UploadedData, Workspace, dataset_bytes

log = logging.getLogger(__name__)


def read_upload(stream: IO[bytes], path: Path) -> LoadedDataset:
    """An uploaded file, on disk at ``path`` only while the loader reads it (S11)."""
    try:
        uploads.save(stream, path)
        return load_dataset(path)
    finally:
        uploads.remove(path)


def label_values(frame: pd.DataFrame, target: str) -> pd.Series:
    return frame[target].dropna().astype(str)


def combine(loaded: list[tuple[str, LoadedDataset]]) -> UploadedData:
    """One UploadedData from the loaded files: an official train/test pair (NSL-KDD with a
    file named '*test*'), or several files with the same columns stacked together."""
    names = [name for name, _ in loaded]
    first = loaded[0][1]
    for name, dataset in loaded[1:]:
        if dataset.profile.profile.name != first.profile.profile.name:
            raise ValueError(f"{name} looks like a different dataset than {names[0]}.")
        if list(dataset.frame.columns) != list(first.frame.columns):
            raise ValueError(f"Column mismatch in {name}. All files must have identical columns.")
    test_files = [i for i, name in enumerate(names) if "test" in name.lower()]
    if len(loaded) == 2 and first.profile.split_strategy == "official" and len(test_files) == 1:
        test_name, test = loaded[test_files[0]]
        train_name, train = loaded[1 - test_files[0]]
        return UploadedData(frame=train.frame, profile=train.profile, files=names,
                            source=DatasetRef.from_loaded(train_name, train),
                            test_frame=test.frame,
                            test_source=DatasetRef.from_loaded(test_name, test))
    if len(loaded) == 1:
        return UploadedData(frame=first.frame, profile=first.profile, files=names,
                            source=DatasetRef.from_loaded(names[0], first))
    frame = pd.concat([dataset.frame for _, dataset in loaded], ignore_index=True)
    combined = hashlib.sha256("\n".join(d.report.sha256 for _, d in loaded).encode()).hexdigest()
    sources = pd.Series(np.repeat(names, [len(d.frame) for _, d in loaded]), index=frame.index)
    return UploadedData(frame=frame, profile=first.profile, files=names, sources=sources,
                        source=DatasetRef(" + ".join(names), combined, len(frame),
                                          len(frame.columns)))


def require_data(ws: Workspace) -> UploadedData:
    if ws.data is None:
        raise ServiceError("No dataset uploaded.", NO_DATASET)
    return ws.data


def require_target(data: UploadedData, target: str) -> str:
    if not target or target not in data.frame.columns:
        raise ServiceError("Target column not specified or not found.", UNKNOWN_COLUMN)
    return target


def training_split(ws: Workspace, target: str, test_size: float) -> SplitResult:
    """The split training will use (same profile, strategy, seed), cached for the previews."""
    data = require_data(ws)
    key = (id(data.frame), target, test_size)
    if ws.split is None or ws.split[0] != key:
        ws.split = None  # free the old split first (big datasets)
        split = split_dataset(data.frame, target, profile=data.profile,
                              test_frame=data.test_frame, sources=data.sources,
                              config=SplitConfig(test_size=test_size))
        ws.split = (key, split)
    return ws.split[1]


def candidate_features(data: UploadedData, target: str,
                        chosen: list[str] | None = None) -> list[str]:
    blocked = set(data.profile.blocked_columns)
    if chosen:
        if any(c not in data.frame.columns for c in chosen):
            raise ServiceError("Unknown input feature.", UNKNOWN_COLUMN)
        return [c for c in chosen if c != target]
    return [c for c in data.frame.columns if c != target and c not in blocked]


def remove_classes(ws: Workspace, data: UploadedData, target: str,
                   to_remove: list[str]) -> tuple[int, list[str]]:
    """Remove the rows of the ``to_remove`` classes (from the test file too); the rows
    removed and the classes left."""
    if not to_remove:
        raise ServiceError("No values selected for removal.")
    current = set(label_values(data.frame, target))
    unknown = [v for v in to_remove if v not in current]
    if unknown:
        raise ServiceError(f"Values not found in target: {', '.join(unknown)}")
    remaining = sorted(current - set(to_remove))
    if not remaining:
        raise ServiceError("Cannot remove all target values. At least 1 class must "
                           "remain.")

    before = len(data.frame)
    keep = ~data.frame[target].astype(str).isin(to_remove)
    data.frame = data.frame[keep].reset_index(drop=True)
    if data.sources is not None:
        data.sources = data.sources[keep].reset_index(drop=True)
    if data.test_frame is not None and target in data.test_frame.columns:
        test_keep = ~data.test_frame[target].astype(str).isin(to_remove)
        data.test_frame = data.test_frame[test_keep].reset_index(drop=True)
    ws.split = None
    ws.data_bytes = dataset_bytes(data)  # smaller now: no room to reserve
    removed = before - len(data.frame)
    log.info("Removed %d target values (%d rows)", len(to_remove), removed)
    return removed, remaining
