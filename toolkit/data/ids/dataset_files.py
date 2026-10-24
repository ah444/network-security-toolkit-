"""Dataset files: their first bytes, their SHA-256, their rows as table chunks.

What the bytes mean (encoding, separator, profile, column types) is decided by
``toolkit.services.ids.datasets``; this module only reads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import pandas as pd

SAMPLE_BYTES = 64 * 1024  # what the encoding, separator and profile are found from


class DatasetError(ValueError):
    """The file can't be read as a dataset (message is safe to show to a user)."""


def read_sample(path: str | Path) -> tuple[bytes, bool]:
    """The first ``SAMPLE_BYTES`` of the file, and whether that is all of it."""
    with Path(path).open("rb") as handle:
        sample = handle.read(SAMPLE_BYTES)
        complete = not handle.read(1)
    return sample, complete


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chunks(path: str | Path, *, separator: str, names: Sequence[str], skip_header: bool,
                encoding: str, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """The rows, ``chunk_rows`` at a time, every value as pandas' C engine reads it."""
    return cast(Iterator[pd.DataFrame], pd.read_csv(
        path,
        sep=separator,
        header=None,
        names=list(names),
        skiprows=1 if skip_header else 0,
        encoding=encoding,
        engine="c",
        chunksize=chunk_rows,
        skipinitialspace=True,
        skip_blank_lines=True,
    ))
