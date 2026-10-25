"""Every rule the project's code is allowed to break, with the reason (v5.2 step 2.3a).

``tests/test_exceptions_inventory.py`` reads the code and the configuration and fails when
they don't match this file exactly: a new suppression, one more of a listed kind, or an entry
whose suppression is gone. An exception is allowed when it has a reason; it is never hidden.

Not listed here, because they are the architecture's own rules rather than exceptions to a
lint: what each layer may import, which data modules controllers may use, where Flask and
argparse may be imported. They are defined (and explained) in ``tests/test_architecture.py``.
"""

# (file, rule, count, reason): a comment that switches a rule off in that file.
SUPPRESSIONS = [
    ("scripts/baseline_v4.py", "noqa[E402]", 12,
     "the imports follow the single-core environment, set before numpy and sklearn load"),
    ("scripts/menu_walkthrough.py", "noqa[E402]", 1,
     "the toolkit import follows the scratch TOOLKIT_INSTANCE_DIR and the sys.path line"),
    ("scripts/ml_report.py", "noqa[E402]", 13,
     "the imports follow the single-core environment (set before numpy and sklearn load) "
     "and the sys.path line"),
    ("tests/fixtures/golden/make_goldens.py", "noqa[E402]", 1,
     "the import follows the line that puts the project and tests folders on sys.path"),
    ("tests/fixtures/legacy/make_legacy_bundle.py", "noqa[E402]", 2,
     "the old code's imports follow the line that puts the project folder on sys.path"),
    ("toolkit/controllers/web/schemas.py", "type: ignore[arg-type]", 1,
     "load() takes any schema class (a TypeVar); mypy cannot know it is a dataclass"),
    ("toolkit/services/ids/profiles.py", "pragma: no cover", 1,
     "unreachable: the generic profile matches every file, the line only documents that"),
    ("toolkit/services/ids/registry.py", "type: ignore[misc]", 1,
     "SelectKBest has no type information (an untyped base class)"),
]

# (table, key, value, reason): a setting in pyproject.toml that switches checking off.
CONFIGURATION = [
    ("tool.mypy.overrides", "ignore_missing_imports",
     ("pandas", "pandas.*", "sklearn", "sklearn.*", "imblearn", "imblearn.*", "joblib",
      "waitress", "threadpoolctl", "xgboost", "xgboost.*", "lightgbm", "lightgbm.*"),
     "these libraries ship no type information; their names are Any for mypy"),
]

# Ruff's per-file-ignores (pattern -> codes); empty since v5.2 step 2.3a, and only shrinks.
PER_FILE_IGNORES: dict[str, tuple[str, ...]] = {}
