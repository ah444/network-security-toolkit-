"""Tunable settings in one place (plan D17). Tests rely on these defaults."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeakageConfig:
    """Thresholds of the leakage guard (plan D9, step 0.4a)."""

    # Single-feature rule: balanced accuracy of a one-feature tree (stratified CV).
    block_score: float = 0.99  # ... and relabelling-like -> block
    warn_score: float = 0.95  # otherwise -> warn
    # "Relabelling": on average at most this many distinct values per class.
    relabel_max_values_per_class: float = 2.0
    # Tree size as leaves: 8 = the capacity of the plan's depth-3 tree. With many classes
    # it grows to relabel_max_values_per_class leaves per class, so a copy of a 20-class
    # label is still caught (a greedy tree needs about one split per class, not log2).
    tree_leaves: int = 8
    cv_folds: int = 3
    sample_rows: int = 50_000  # stratified sample of the training split for the CV check
    # ID-like columns (integer or text only).
    id_unique_share: float = 0.99
    id_unique_min_rows: int = 1_000  # below this, most integer features look unique
    id_increasing_min_rows: int = 20
    random_state: int = 0


@dataclass(frozen=True)
class SplitConfig:
    """Train/test split settings (plan D9, step 0.4b)."""

    test_size: float = 0.2
    random_state: int = 42  # the seed the v4 app used for its split
    # Time-column checks before a temporal split; above these the split falls back.
    max_time_parse_failure: float = 0.01  # share of non-empty values that don't parse
    max_time_missing: float = 0.01  # share of empty values
    # Official split (published train + test files): False keeps both files as published,
    # so scores stay comparable; True ("strict") removes copies and test rows also in train.
    official_dedupe: bool = False


REPO_ROOT = Path(__file__).resolve().parent.parent


def instance_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Local, gitignored data of this install (runs, signing key): ``TOOLKIT_INSTANCE_DIR``,
    else ``instance/`` in the project folder."""
    environ = os.environ if environ is None else environ
    value = environ.get("TOOLKIT_INSTANCE_DIR", "").strip()
    return Path(value) if value else REPO_ROOT / "instance"


ENV_FILE = REPO_ROOT / ".env"
PLACEHOLDER = re.compile(r"^your_.*_here$", re.IGNORECASE)  # left from an example file


def read_env_file(path: str | Path) -> dict[str, str]:
    """``NAME=value`` lines of a ``.env``-style file; {} when it doesn't exist.

    Blank lines, ``#`` comments, lines without ``=``, empty values and placeholders
    (``your_..._here``) are skipped; ``export`` and surrounding quotes are removed.
    """
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    except (OSError, UnicodeDecodeError) as error:
        log.warning("Can't read %s: %s", path, error)
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if value and not PLACEHOLDER.match(value):
            values[name] = value
    return values


def env_value(name: str, environ: Mapping[str, str] | None = None,
              env_file: str | Path | None = None) -> str:
    """A secret or user setting: the environment variable, else the line in ``.env``
    (project folder), else ''. Placeholders count as unset. Only the API keys and
    ``OPENROUTER_MODEL`` are read this way; ``TOOLKIT_*`` settings come from the
    environment only."""
    environ = os.environ if environ is None else environ
    value = environ.get(name, "").strip()
    if value and not PLACEHOLDER.match(value):
        return value
    return read_env_file(ENV_FILE if env_file is None else env_file).get(name, "")


def max_workers(environ: Mapping[str, str] | None = None) -> int:
    """CPU workers for training and CV: ``TOOLKIT_MAX_WORKERS``, else all cores but one (D8).

    Never more than the machine has; a value that isn't a positive whole number is an error,
    so a typo can't silently use every core.
    """
    environ = os.environ if environ is None else environ
    cores = os.cpu_count() or 1
    raw = environ.get("TOOLKIT_MAX_WORKERS", "").strip()
    if not raw:
        return max(1, cores - 1)
    try:
        workers = int(raw)
    except ValueError:
        workers = 0
    if workers < 1:
        raise ValueError(f"TOOLKIT_MAX_WORKERS must be a whole number >= 1, not {raw!r}")
    return min(workers, cores)


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def log_level(environ: Mapping[str, str] | None = None) -> int:
    """Level of the log files: ``TOOLKIT_LOG_LEVEL`` (DEBUG, INFO, WARNING, ERROR), else INFO.

    Any other value is an error, so a typo can't silently turn logging off.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get("TOOLKIT_LOG_LEVEL", "").strip()
    if not raw:
        return logging.INFO
    if raw.upper() not in LOG_LEVELS:
        raise ValueError(f"TOOLKIT_LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, not {raw!r}")
    return logging.getLevelNamesMapping()[raw.upper()]


@dataclass(frozen=True)
class PipelineConfig:
    """Training pipeline settings (plan D9, D8, step 0.4c)."""

    random_state: int = 42  # every estimator and sampler, including nested ones (B20)
    smote_max_neighbors: int = 5  # lowered for small classes (smallest train class - 1)
    # Linear SVM is skipped above this many **training** rows (B5; v4 used the whole file).
    svm_max_train_rows: int = 100_000


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation and model selection (plan D9, D10, step 0.4d)."""

    cv_folds: int = 3
    # CV runs on a stratified sample of the training split above this size (recorded);
    # 3 folds of RF on 2 M CIC rows would take about half an hour on a laptop.
    cv_max_rows: int = 200_000
    random_state: int = 42
    # Re-running a TrainingRun: metrics must match exactly with one worker; with more,
    # thread scheduling in some libraries (e.g. LightGBM) may change the last digits.
    parallel_metric_tolerance: float = 0.005


@dataclass(frozen=True)
class ScanConfig:
    """Port scanner limits (plan S16, step 0.6)."""

    max_hosts: int = 256  # one /24 network
    max_probes: int = 65_536  # host x port checks; TCP and UDP count separately


@dataclass(frozen=True)
class WebConfig:
    """Web trainer settings (plan S6, S9, D7, D23; one place since v5.1 step 1.2a)."""

    max_upload_mb: int = 2048  # large enough for the full CIC-IDS2017 file (~1 GB), S9
    workspace_idle_minutes: int = 240  # unused workspaces are dropped (not while training)
    max_workspaces: int = 4  # open workspaces (browser sessions with a dataset or bundle)
    max_data_mb: int = 4096  # loaded datasets of all workspaces together (models not counted)
    port: int = 5000
    https: bool = False  # Secure session cookie: only behind HTTPS (D23)
    debug: bool = False  # Flask's debug server: development only, its debugger can run code
    secret_key: str = ""  # session signing key; '' = random per start


def _int_setting(environ: Mapping[str, str], variable: str, default: int) -> int:
    raw = environ.get(variable, "").strip()
    return int(raw) if raw else default


def web_config(environ: Mapping[str, str] | None = None) -> WebConfig:
    """``TOOLKIT_MAX_UPLOAD_MB``, ``TOOLKIT_WORKSPACE_IDLE_MIN``, ``TOOLKIT_MAX_WORKSPACES``,
    ``TOOLKIT_MAX_DATA_MB``, ``TOOLKIT_PORT``, ``TOOLKIT_HTTPS=1``, ``TOOLKIT_DEBUG=1`` and
    ``TOOLKIT_SECRET_KEY``, else the defaults."""
    environ = os.environ if environ is None else environ
    return WebConfig(
        max_upload_mb=_int_setting(environ, "TOOLKIT_MAX_UPLOAD_MB", WebConfig.max_upload_mb),
        workspace_idle_minutes=_int_setting(environ, "TOOLKIT_WORKSPACE_IDLE_MIN",
                                            WebConfig.workspace_idle_minutes),
        max_workspaces=_int_setting(environ, "TOOLKIT_MAX_WORKSPACES", WebConfig.max_workspaces),
        max_data_mb=_int_setting(environ, "TOOLKIT_MAX_DATA_MB", WebConfig.max_data_mb),
        port=_int_setting(environ, "TOOLKIT_PORT", WebConfig.port),
        https=environ.get("TOOLKIT_HTTPS") == "1",
        debug=environ.get("TOOLKIT_DEBUG") == "1",
        secret_key=environ.get("TOOLKIT_SECRET_KEY", ""),
    )


def scan_limits(environ: Mapping[str, str] | None = None) -> ScanConfig:
    """``TOOLKIT_SCAN_MAX_HOSTS`` / ``TOOLKIT_SCAN_MAX_PROBES``, else the defaults.

    A value that isn't a positive whole number is an error, so a typo can't lift a limit.
    """
    environ = os.environ if environ is None else environ
    values = {}
    for field, variable in (("max_hosts", "TOOLKIT_SCAN_MAX_HOSTS"),
                            ("max_probes", "TOOLKIT_SCAN_MAX_PROBES")):
        raw = environ.get(variable, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(f"{variable} must be a whole number >= 1, not {raw!r}")
        values[field] = value
    return ScanConfig(**values)
