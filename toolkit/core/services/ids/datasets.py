"""Dataset loader: CSV/TXT files -> a clean, typed DataFrame plus a report of what was done.

Steps (plan D16, B17, R7):
1. Read only the start of the file to find the encoding (BOM, UTF-8, else cp1252) and
   the separator (the one that splits the sample rows into the same number of fields).
2. Recognise the dataset from that sample (``profiles.detect_profile``); this also
   decides whether the first line is a header and gives names to headerless files.
3. Read the file with pandas' C engine in chunks; narrow each chunk's dtypes so the
   peak memory stays near the final size.
4. Strip header spaces, drop ``X.1`` columns that duplicate ``X``, normalise label
   values the profile asks for, and record everything in a ``LoadReport``.

Values are not "cleaned" here: ``inf`` and empty cells are counted, not changed.
What to do with them is a pipeline decision (step 0.4c).
"""

from __future__ import annotations

import codecs
import csv
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.ids.dataset_files import DatasetError as DatasetError  # the callers' error
from ...data.ids.dataset_files import file_sha256, read_chunks, read_sample
from .profiles import ProfileMatch, detect_profile

log = logging.getLogger(__name__)

SAMPLE_ROWS = 200
SEPARATORS = (",", ";", "\t", "|")
DEFAULT_CHUNK_ROWS = 200_000
# Text columns with at most this share of distinct values become 'category'.
CATEGORY_MAX_SHARE = 0.5
_SUFFIXED = re.compile(r"^(?P<base>.+)\.(?P<n>\d+)$")


@dataclass(frozen=True)
class DatasetInfo:
    """Format and profile of a file, found from its first ``SAMPLE_BYTES`` only."""

    encoding: str
    separator: str
    raw_header: tuple[str, ...] | None  # the header line as written, if there is one
    profile: ProfileMatch


@dataclass(frozen=True)
class LoadReport:
    sha256: str
    encoding: str
    separator: str
    has_header: bool
    rows: int
    columns: int
    renamed_columns: dict[str, str] = field(default_factory=dict)  # as written -> stripped
    dropped_duplicate_columns: tuple[str, ...] = ()
    downcast: dict[str, str] = field(default_factory=dict)  # column -> narrower dtype
    missing_values: int = 0
    infinite_values: int = 0
    label_values_normalised: int = 0  # rows whose label text was fixed
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    profile: ProfileMatch
    report: LoadReport


# --- Sniffing ----------------------------------------------------------------------

def _detect_encoding(sample: bytes) -> str:
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    try:
        # A multi-byte character may be cut at the end of the sample.
        sample.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.start < len(sample) - 3:
            return "cp1252"
    return "utf-8"


def _sample_text(sample: bytes, encoding: str, complete: bool) -> str:
    text = sample.decode(encoding, errors="replace")
    if not complete and "\n" in text:
        text = text[: text.rindex("\n")]  # drop the last, possibly cut, line
    return text


def _sniff_separator(lines: list[str]) -> str:
    """The separator that splits the sample lines most consistently into > 1 fields.

    Ranked by: every line has the same field count, then how many lines share the
    most common count, then that count. Ties keep the order of ``SEPARATORS``.
    """
    def score(separator: str) -> tuple[bool, int, int]:
        counts = [len(row) for row in csv.reader(lines, delimiter=separator)]
        common, freq = Counter(counts).most_common(1)[0]
        if common < 2:
            return (False, 0, 0)
        return (freq == len(counts), freq, common)

    return max(SEPARATORS, key=score)


def inspect_dataset(path: str | Path) -> DatasetInfo:
    """Find encoding, separator and profile from the start of the file."""
    sample, complete = read_sample(path)
    if not sample.strip():
        raise DatasetError("The file is empty.")
    encoding = _detect_encoding(sample)
    text = _sample_text(sample, encoding, complete)
    lines = [line for line in text.splitlines() if line.strip()][:SAMPLE_ROWS]
    if not lines:
        raise DatasetError("The file is empty.")
    separator = _sniff_separator(lines)
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=separator))
    match = detect_profile(rows)
    raw_header = tuple(rows[0]) if match.has_header else None
    return DatasetInfo(encoding, separator, raw_header, match)


# --- Loading -----------------------------------------------------------------------

def _narrow(frame: pd.DataFrame) -> pd.DataFrame:
    """Smallest exact numeric dtypes: ints shrink, floats become float32 only if lossless."""
    for name in frame.columns:
        column = frame[name]
        if column.dtype.kind in "iu":
            frame[name] = pd.to_numeric(column, downcast="integer")
        elif column.dtype == np.float64:
            narrow = column.to_numpy(dtype=np.float32)
            if np.array_equal(narrow.astype(np.float64), column.to_numpy(), equal_nan=True):
                frame[name] = narrow
    return frame


def _unify_text(column: pd.Series) -> pd.Series:
    """A column read as numbers in one chunk and text in another: make it one type."""
    values = column.dropna()
    if values.map(lambda v: isinstance(v, str)).all():
        return column
    numbers = pd.to_numeric(column, errors="coerce")
    if numbers.notna().sum() == len(values):
        return numbers
    return column.map(lambda v: v if pd.isna(v) or isinstance(v, str) else str(v))


def _read_frame(path: Path, info: DatasetInfo, encoding: str, chunk_rows: int) -> pd.DataFrame:
    chunks = read_chunks(path, separator=info.separator, names=info.profile.column_names,
                         skip_header=info.profile.has_header, encoding=encoding,
                         chunk_rows=chunk_rows)
    parts = [_narrow(chunk) for chunk in chunks]
    if not parts or not any(len(part) for part in parts):
        raise DatasetError("The file has a header but no data rows.")
    return pd.concat(parts, ignore_index=True)


def load_dataset(path: str | Path, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> LoadedDataset:
    """Read a CSV/TXT dataset; see the module docstring for the steps."""
    path = Path(path)
    info = inspect_dataset(path)
    match = info.profile
    warnings: list[str] = []

    encoding = info.encoding
    try:
        try:
            frame = _read_frame(path, info, encoding, chunk_rows)
        except UnicodeDecodeError:
            if encoding != "utf-8":
                raise
            encoding = "cp1252"  # invalid UTF-8 after the sample
            warnings.append("ENCODING_FALLBACK_CP1252")
            frame = _read_frame(path, info, encoding, chunk_rows)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        if isinstance(error, DatasetError):
            raise
        raise DatasetError(f"Could not read the file as a table: {error}") from None

    for name in frame.columns:
        if frame[name].dtype == object:
            frame[name] = _unify_text(frame[name])
    frame = _narrow(frame)  # again: chunks may have narrowed differently

    dropped: list[str] = []
    for name in list(frame.columns):
        found = _SUFFIXED.match(name)
        if found and found["base"] in frame.columns:
            if frame[name].equals(frame[found["base"]]):
                dropped.append(name)
            else:
                warnings.append("SUFFIXED_COLUMN_DIFFERS")
    if dropped:
        log.info("Dropped duplicate columns: %s", dropped)
        frame = frame.drop(columns=dropped)

    normalised = 0
    label = match.label_column
    if label in frame.columns and match.label_replacements and frame[label].dtype == object:
        before = frame[label]
        after = before
        for old, new in match.label_replacements:
            after = after.str.replace(old, new, regex=False)
        normalised = int(((before != after) & before.notna()).sum())
        frame[label] = after

    downcast: dict[str, str] = {}
    for name in frame.columns:
        column = frame[name]
        if column.dtype == object:
            distinct = column.nunique(dropna=True)
            if name == label or distinct <= max(1, CATEGORY_MAX_SHARE * len(column)):
                frame[name] = column.astype("category")
                downcast[name] = "category"
        elif column.dtype.kind in "iuf" and column.dtype.itemsize < 8:
            downcast[name] = str(column.dtype)

    infinite = sum(int(np.isinf(frame[name].to_numpy()).sum())
                   for name in frame.columns if frame[name].dtype.kind == "f")
    renamed = {}
    if info.raw_header is not None:  # names come from the header, so lengths match
        renamed = {raw: clean for raw, clean in zip(info.raw_header, match.column_names,
                                                    strict=True) if raw != clean}
    report = LoadReport(
        sha256=file_sha256(path),
        encoding=encoding,
        separator=info.separator,
        has_header=match.has_header,
        rows=len(frame),
        columns=frame.shape[1],
        renamed_columns=renamed,
        dropped_duplicate_columns=tuple(dropped),
        downcast=downcast,
        missing_values=int(frame.isna().sum().sum()),
        infinite_values=infinite,
        label_values_normalised=normalised,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    return LoadedDataset(frame, match, report)
