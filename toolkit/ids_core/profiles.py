"""Dataset profiles: recognise a dataset from its content, never from its file name.

A profile is chosen from the header names (case, spaces and underscores ignored)
and, for files without a header, from value patterns in the sample rows. The
same content therefore gets the same profile after renaming the file, shuffling
rows or changing line endings / encoding (plan D9, F4, L10).

Each match also says which column is the label, which columns must never be
used as features (label-derived, IDs, time), and which split strategy fits
(used from step 0.4b). Warnings are reason codes, not free text.

Column names below are copied from real files only (the local CIC-IDS2017 and
NSL-KDD copies, and the v4 model's ``config.json`` for the "improved" CIC
version). Schemas that aren't available locally (UNSW-NB15, the full improved
CIC header) are not guessed; see ``docs/datasets.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Rows = Sequence[Sequence[str]]


@dataclass(frozen=True)
class DatasetProfile:
    """A known dataset family plus the function that recognises it."""

    name: str
    title: str
    detect: Callable[[Rows], ProfileMatch | None]


@dataclass(frozen=True)
class ProfileMatch:
    """What was recognised in one file, and how to treat its columns."""

    profile: DatasetProfile
    variant: str
    has_header: bool
    column_names: tuple[str, ...]  # stripped, unique, in file order
    label_column: str | None
    blocked_columns: tuple[str, ...] = ()  # never features: label-derived, IDs, time
    split_strategy: str = "stratified"  # official | temporal_per_class | stratified
    time_column: str | None = None
    label_replacements: tuple[tuple[str, str], ...] = ()  # applied to label values
    warnings: tuple[str, ...] = ()


def normalise(name: str) -> str:
    """Comparison key for a column name: lower case, '_' and runs of spaces -> ' '."""
    return re.sub(r"[\s_]+", " ", name.strip().lower()).strip()


def clean_column_names(raw: Sequence[str]) -> tuple[str, ...]:
    """Strip surrounding spaces; make names unique with '.1', '.2' like pandas does."""
    names: list[str] = []
    seen: set[str] = set()
    for original in raw:
        name = original.strip() or "unnamed"
        candidate, counter = name, 1
        while candidate in seen:
            candidate = f"{name}.{counter}"
            counter += 1
        seen.add(candidate)
        names.append(candidate)
    return tuple(names)


def is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _find(names: Sequence[str], *wanted: str) -> str | None:
    """The first name (in the order of ``wanted``) whose normalised form matches."""
    by_key = {normalise(n): n for n in reversed(names)}
    for key in wanted:
        if key in by_key:
            return by_key[key]
    return None


# --- CIC-IDS2017 -------------------------------------------------------------------

# Header of the original "MachineLearningCSV" files (local CIC-2017.csv), stripped.
CIC_ORIGINAL_COLUMNS = (
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label",
)

# Names known from the "improved" re-labelled version: the v4 model's config.json
# (trained on it). Only a partial view of that header.
CIC_IMPROVED_KNOWN_COLUMNS = (
    "Attempted Category", "Bwd Segment Size Avg", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Bwd Packet Length Max", "Subflow Bwd Bytes", "Packet Length Std",
    "Packet Length Mean",
)

_CIC_FEATURE_KEYS = frozenset(
    normalise(n) for n in CIC_ORIGINAL_COLUMNS + CIC_IMPROVED_KNOWN_COLUMNS
) - {"label"}
_CIC_ORIGINAL_FEATURE_KEYS = frozenset(normalise(n) for n in CIC_ORIGINAL_COLUMNS) - {"label"}
# Identifier columns found in CIC variants with flow metadata: never features.
_CIC_ID_KEYS = ("flow id", "source ip", "src ip", "destination ip", "dst ip")
_CIC_MIN_KNOWN, _CIC_MIN_SHARE = 5, 0.8


def _detect_cic(rows: Rows) -> ProfileMatch | None:
    names = clean_column_names(rows[0])
    label = _find(names, "label")
    if label is None:
        return None
    keys = {normalise(n) for n in names if n != label}
    known = keys & _CIC_FEATURE_KEYS
    if len(known) < _CIC_MIN_KNOWN or len(known) / len(keys) < _CIC_MIN_SHARE:
        return None

    warnings: list[str] = []
    attempted = _find(names, "attempted category")
    if attempted is not None:
        variant = "improved"
    elif _CIC_ORIGINAL_FEATURE_KEYS <= keys:
        variant = "original"
    else:
        variant = "original_subset"
        warnings.append("COLUMN_SUBSET")

    time_column = _find(names, "timestamp")
    blocked = [attempted, time_column] + [_find(names, key) for key in _CIC_ID_KEYS]
    if time_column is None:
        warnings.append("CIC_NO_TIME_COLUMN")
    return ProfileMatch(
        profile=CIC_IDS2017,
        variant=variant,
        has_header=True,
        column_names=names,
        label_column=label,
        blocked_columns=tuple(n for n in blocked if n is not None),
        split_strategy="temporal_per_class" if time_column else "stratified",
        time_column=time_column,
        # Web-attack labels: U+FFFD (broken en dash in the UTF-8 copy) or a real en dash.
        label_replacements=(("�", "-"), ("–", "-")),
        warnings=tuple(warnings),
    )


# --- NSL-KDD -----------------------------------------------------------------------

# Header of the local KDDTrain+.txt copy, stripped (41 features + attack + level).
NSL_KDD_COLUMNS = (
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack",
    "level",
)
_NSL_CORE_KEYS = frozenset(
    normalise(n) for n in ("duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes")
)
_NSL_ALL_KEYS = frozenset(normalise(n) for n in NSL_KDD_COLUMNS)
# Values seen in the local KDDTrain+.txt (columns 2 and 4).
_NSL_PROTOCOLS = frozenset({"tcp", "udp", "icmp"})
_NSL_FLAGS = frozenset({"OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"})
_NSL_MIN_ROW_SHARE = 0.9


def _nsl_match(names: tuple[str, ...], has_header: bool, variant: str,
               warnings: list[str]) -> ProfileMatch:
    label = _find(names, "attack", "class", "label")
    if label is None:
        warnings.append("LABEL_COLUMN_UNKNOWN")
    level = _find(names, "level")
    return ProfileMatch(
        profile=NSL_KDD,
        variant=variant,
        has_header=has_header,
        column_names=names,
        label_column=label,
        blocked_columns=(level,) if level else (),
        split_strategy="official",  # the published KDDTrain+ / KDDTest+ files
        warnings=tuple(warnings),
    )


def _looks_like_nsl_row(row: Sequence[str]) -> bool:
    return (len(row) in (len(NSL_KDD_COLUMNS), len(NSL_KDD_COLUMNS) - 1)
            and is_number(row[0])
            and row[1].strip() in _NSL_PROTOCOLS
            and row[3].strip() in _NSL_FLAGS)


def _detect_nsl_kdd(rows: Rows) -> ProfileMatch | None:
    names = clean_column_names(rows[0])
    keys = {normalise(n) for n in names}
    if _NSL_CORE_KEYS <= keys:
        variant = "full" if _NSL_ALL_KEYS <= keys else "column_subset"
        return _nsl_match(names, True, variant, [])

    # No header (the official KDDTest+.txt): recognise the rows themselves.
    if sum(map(_looks_like_nsl_row, rows)) >= _NSL_MIN_ROW_SHARE * len(rows):
        width = len(rows[0])
        variant = "full" if width == len(NSL_KDD_COLUMNS) else "no_level_column"
        return _nsl_match(NSL_KDD_COLUMNS[:width], False, variant, ["HEADERLESS_NAMES_APPLIED"])
    return None


# --- Generic (any other CSV) -------------------------------------------------------

_LABEL_KEYS = ("label", "class", "attack", "attack cat", "target")


def _first_row_is_header(rows: Rows) -> bool:
    """A header has fewer numeric fields than a typical data row (or the file is all text)."""
    def numeric(row: Sequence[str]) -> int:
        return sum(1 for value in row if value.strip() and is_number(value))

    rest = sorted(numeric(row) for row in rows[1:])
    if not rest:
        return numeric(rows[0]) < len(rows[0])
    typical = rest[len(rest) // 2]
    return numeric(rows[0]) < typical or typical == 0


def _detect_generic(rows: Rows) -> ProfileMatch:
    warnings = ["UNKNOWN_SCHEMA"]
    has_header = _first_row_is_header(rows)
    if has_header:
        names = clean_column_names(rows[0])
    else:
        names = tuple(f"col_{i}" for i in range(1, len(rows[0]) + 1))
        warnings.append("NO_HEADER")
    label = _find(names, *_LABEL_KEYS)
    warnings.append("LABEL_COLUMN_BY_NAME" if label else "LABEL_COLUMN_UNKNOWN")
    return ProfileMatch(
        profile=GENERIC,
        variant="generic",
        has_header=has_header,
        column_names=names,
        label_column=label,
        warnings=tuple(warnings),
    )


# --- Registry ----------------------------------------------------------------------

CIC_IDS2017 = DatasetProfile("cic_ids2017", "CIC-IDS2017 (CICFlowMeter flows)", _detect_cic)
NSL_KDD = DatasetProfile("nsl_kdd", "NSL-KDD", _detect_nsl_kdd)
GENERIC = DatasetProfile("generic", "Unknown CSV (generic)", _detect_generic)

# Checked in order; the generic profile always matches, so it stays last.
PROFILES: tuple[DatasetProfile, ...] = (CIC_IDS2017, NSL_KDD, GENERIC)


def detect_profile(rows: Rows) -> ProfileMatch:
    """Recognise a dataset from its first parsed rows (the possible header included)."""
    rows = [row for row in rows if any(value.strip() for value in row)]
    if not rows:
        raise ValueError("no rows to inspect")
    for profile in PROFILES:
        match = profile.detect(rows)
        if match is not None:
            return match
    raise AssertionError("the generic profile always matches")  # pragma: no cover
