"""Tunable settings in one place (plan D17). Tests rely on these defaults."""

from __future__ import annotations

import logging
import os
import platform
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


# The project folder (when run from it) holds toolkit/: as many folders above this file as
# the module's name has dots (toolkit.core.data.settings -> parents[3]), so moving the file
# changes both together (v5.2 step 2.3b; a fixed parents[2] would have pointed at toolkit/
# after the move, and the tool would have taken itself for an installed copy). Under any
# other name (the file run as a script) the count would be wrong: refuse instead.
if not __name__.startswith("toolkit."):
    raise RuntimeError(f"{__name__}: settings.py is part of the toolkit package; import it "
                       "as toolkit.core.data.settings")
REPO_ROOT = Path(__file__).resolve().parents[__name__.count(".")]
APP_NAME = "network-security-toolkit"


def is_checkout() -> bool:
    """True when ``toolkit/`` sits in the project folder (a clone, or ``pip install -e .``);
    False for an installed copy (``pip install .``: site-packages has no pyproject.toml)."""
    return (REPO_ROOT / "pyproject.toml").is_file()


def user_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """An installed copy's folder for ``instance/`` and ``.env``: ``%LOCALAPPDATA%`` on
    Windows, ``~/Library/Application Support`` on macOS, ``$XDG_DATA_HOME`` (if absolute) or
    ``~/.local/share`` elsewhere; then ``network-security-toolkit``."""
    environ = os.environ if environ is None else environ
    system = platform.system()
    if system == "Windows":
        local = environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local" / "share"
    return base / APP_NAME


def _data_root(environ: Mapping[str, str] | None = None) -> Path:
    """The project folder in a checkout, else the installed copy's ``user_data_dir``."""
    return REPO_ROOT if is_checkout() else user_data_dir(environ)


def instance_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Local, gitignored data of this install (runs, bundles, signing key, logs):
    ``TOOLKIT_INSTANCE_DIR``, else ``instance/`` in the project folder (an installed copy:
    in its ``user_data_dir``)."""
    environ = os.environ if environ is None else environ
    value = environ.get("TOOLKIT_INSTANCE_DIR", "").strip()
    return Path(value) if value else _data_root(environ) / "instance"


def env_file(environ: Mapping[str, str] | None = None) -> Path:
    """The ``.env`` file: in the project folder (an installed copy: its ``user_data_dir``)."""
    return _data_root(environ) / ".env"


ENV_FILE = env_file()
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


def whole_number(environ: Mapping[str, str], variable: str, default: int) -> int:
    """``variable`` as a whole number >= 1; ``default`` when it is unset.

    Anything else is a ValueError naming the setting (v5.1 step 1.5e), so a typo can't lift
    a limit, turn one into 0 or silently use every core.
    """
    raw = environ.get(variable, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{variable} must be a whole number >= 1, not {raw!r}")
    return value


def switch(environ: Mapping[str, str], variable: str) -> bool:
    """An on/off setting: ``1`` is on, ``0`` or unset is off.

    Anything else (``true``, ``yes``) is a ValueError naming the setting (v5.1 step 1.5e);
    before, it silently meant off.
    """
    raw = environ.get(variable, "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{variable} must be 1 (on) or 0 (off), not {raw!r}")
    return raw == "1"


def max_workers(environ: Mapping[str, str] | None = None) -> int:
    """CPU workers for training and CV: ``TOOLKIT_MAX_WORKERS``, else all cores but one (D8).

    Never more than the machine has; a value that isn't a positive whole number is an error.
    """
    environ = os.environ if environ is None else environ
    cores = os.cpu_count() or 1
    workers = whole_number(environ, "TOOLKIT_MAX_WORKERS", 0)
    return min(workers, cores) if workers else max(1, cores - 1)


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
class ExplainConfig:
    """What a model relies on (v5.3 F3): permutation importance on the test split, and why one
    prediction came out as it did."""

    # Above this many test rows, a stratified sample of them (at least min_per_class each).
    importance_max_rows: int = 5_000
    importance_min_per_class: int = 3
    importance_repeats: int = 5
    # One feature with more than this share of the summed positive drops: the importance
    # concentration warning (a copy of the label is one possible cause).
    concentration_share: float = 0.5
    # One prediction: the features that push most toward the class, and most away from it.
    local_top: int = 5


@dataclass(frozen=True)
class ScanConfig:
    """Port scanner limits (plan S16, step 0.6)."""

    max_hosts: int = 256  # one /24 network
    max_probes: int = 65_536  # host x port checks; TCP and UDP count separately
    max_minutes: int = 60  # the nmap engine's time for one host (its --host-timeout)


@dataclass(frozen=True)
class WebConfig:
    """Web trainer settings (plan S6, S9, D7, D23; one place since v5.1 step 1.2a)."""

    max_upload_mb: int = 2048  # large enough for the full CIC-IDS2017 file (~1 GB), S9
    max_predict_mb: int = 512  # one "Predict a CSV file" upload (it is read into memory)
    workspace_idle_minutes: int = 240  # unused workspaces are dropped (not while training)
    max_workspaces: int = 4  # open workspaces (browser sessions with a dataset or bundle)
    max_data_mb: int = 4096  # loaded datasets of all workspaces together (models not counted)
    job_max_minutes: int = 180  # a training job is stopped (cancelled) after this long
    port: int = 5000
    host: str = "127.0.0.1"  # where it listens: --host only, never an environment variable (2.4f)
    https: bool = False  # Secure session cookie: only behind HTTPS (D23)
    debug: bool = False  # Flask's debug server: development only, its debugger can run code
    secret_key: str = ""  # session signing key; '' = random per start
    open_browser: bool = True  # open the trainer in the default browser (TOOLKIT_NO_BROWSER=1: no)


def port_number(text: str, source: str = "TOOLKIT_PORT") -> int:
    """A TCP port (1-65535) from ``text``; ValueError naming ``source`` otherwise."""
    try:
        port = int(text.strip())
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise ValueError(f"{source} must be a port number from 1 to 65535, not {text!r}")
    return port


def web_config(environ: Mapping[str, str] | None = None) -> WebConfig:
    """``TOOLKIT_MAX_UPLOAD_MB``, ``TOOLKIT_MAX_PREDICT_MB``, ``TOOLKIT_WORKSPACE_IDLE_MIN``,
    ``TOOLKIT_MAX_WORKSPACES``,
    ``TOOLKIT_MAX_DATA_MB``, ``TOOLKIT_JOB_MAX_MIN``, ``TOOLKIT_PORT``, ``TOOLKIT_HTTPS=1``, ``TOOLKIT_DEBUG=1``,
    ``TOOLKIT_SECRET_KEY`` and ``TOOLKIT_NO_BROWSER=1``, else the defaults. A bad value is a
    ValueError naming the setting."""
    environ = os.environ if environ is None else environ
    return WebConfig(
        max_upload_mb=whole_number(environ, "TOOLKIT_MAX_UPLOAD_MB", WebConfig.max_upload_mb),
        max_predict_mb=whole_number(environ, "TOOLKIT_MAX_PREDICT_MB", WebConfig.max_predict_mb),
        workspace_idle_minutes=whole_number(environ, "TOOLKIT_WORKSPACE_IDLE_MIN",
                                            WebConfig.workspace_idle_minutes),
        max_workspaces=whole_number(environ, "TOOLKIT_MAX_WORKSPACES", WebConfig.max_workspaces),
        max_data_mb=whole_number(environ, "TOOLKIT_MAX_DATA_MB", WebConfig.max_data_mb),
        job_max_minutes=whole_number(environ, "TOOLKIT_JOB_MAX_MIN", WebConfig.job_max_minutes),
        port=port_number(environ["TOOLKIT_PORT"]) if environ.get("TOOLKIT_PORT", "").strip()
        else WebConfig.port,
        https=switch(environ, "TOOLKIT_HTTPS"),
        debug=switch(environ, "TOOLKIT_DEBUG"),
        secret_key=environ.get("TOOLKIT_SECRET_KEY", ""),
        open_browser=not switch(environ, "TOOLKIT_NO_BROWSER"),
    )


def scan_limits(environ: Mapping[str, str] | None = None) -> ScanConfig:
    """``TOOLKIT_SCAN_MAX_HOSTS``, ``TOOLKIT_SCAN_MAX_PROBES`` and ``TOOLKIT_SCAN_MAX_MIN``,
    else the defaults.

    A value that isn't a positive whole number is an error, so a typo can't lift a limit.
    """
    environ = os.environ if environ is None else environ
    return ScanConfig(
        max_hosts=whole_number(environ, "TOOLKIT_SCAN_MAX_HOSTS", ScanConfig.max_hosts),
        max_probes=whole_number(environ, "TOOLKIT_SCAN_MAX_PROBES", ScanConfig.max_probes),
        max_minutes=whole_number(environ, "TOOLKIT_SCAN_MAX_MIN", ScanConfig.max_minutes))


def nmap_path(environ: Mapping[str, str] | None = None) -> str:
    """``TOOLKIT_NMAP_PATH`` (quotes and spaces around it dropped), '' when unset; ValueError
    naming it when it is not a full path (a relative one would be read from the current
    folder, where a planted program could wait)."""
    environ = os.environ if environ is None else environ
    raw = environ.get("TOOLKIT_NMAP_PATH", "").strip().strip('"')
    if raw and not Path(raw).is_absolute():
        raise ValueError(f"TOOLKIT_NMAP_PATH must be nmap's full path, not {raw!r}")
    return raw


# The TOOLKIT_* settings each command reads, checked before it starts (v5.1 step 1.5e).
# TOOLKIT_INSTANCE_DIR and TOOLKIT_SECRET_KEY take any value.
SHARED_SETTINGS = ("TOOLKIT_LOG_LEVEL",)
COMMAND_SETTINGS = {
    "scan": (*SHARED_SETTINGS, "TOOLKIT_SCAN_MAX_HOSTS", "TOOLKIT_SCAN_MAX_PROBES",
             "TOOLKIT_SCAN_MAX_MIN", "TOOLKIT_NMAP_PATH"),
    "ids": SHARED_SETTINGS,
    "web": (*SHARED_SETTINGS, "TOOLKIT_MAX_UPLOAD_MB", "TOOLKIT_MAX_PREDICT_MB",
            "TOOLKIT_WORKSPACE_IDLE_MIN", "TOOLKIT_MAX_WORKSPACES", "TOOLKIT_MAX_DATA_MB",
            "TOOLKIT_JOB_MAX_MIN", "TOOLKIT_MAX_WORKERS", "TOOLKIT_PORT", "TOOLKIT_HTTPS",
            "TOOLKIT_DEBUG", "TOOLKIT_NO_BROWSER"),
}
SWITCHES = ("TOOLKIT_HTTPS", "TOOLKIT_DEBUG", "TOOLKIT_NO_BROWSER")


def _check(environ: Mapping[str, str], variable: str) -> None:
    """Reads ``variable`` the way its command does; ValueError naming it when it is bad."""
    if variable == "TOOLKIT_LOG_LEVEL":
        log_level(environ)
    elif variable == "TOOLKIT_PORT":
        if environ.get(variable, "").strip():
            port_number(environ[variable])
    elif variable in SWITCHES:
        switch(environ, variable)
    elif variable == "TOOLKIT_NMAP_PATH":
        nmap_path(environ)
    else:
        whole_number(environ, variable, 0)


def check_settings(command: str, environ: Mapping[str, str] | None = None) -> list[str]:
    """One message per bad ``TOOLKIT_*`` setting that ``command`` (scan, ids, web) reads,
    in a fixed order; [] when all are fine. A setting another command reads is not checked,
    so a typo in a web setting doesn't stop a scan."""
    environ = os.environ if environ is None else environ
    problems = []
    for variable in COMMAND_SETTINGS[command]:
        try:
            _check(environ, variable)
        except ValueError as error:
            problems.append(str(error))
    return problems
