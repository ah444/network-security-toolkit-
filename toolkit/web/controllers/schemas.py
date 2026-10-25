"""Request schemas (S9): one small frozen dataclass per route, checked by ``load``.

``load(Schema, source)`` reads a JSON body or a query string: it converts each field
to its declared type, refuses missing required fields and wrong types with a message
naming the field, clamps numbers that have a range, and caps lists and texts. Checks
that need the dataset (does this column exist?) stay in the routes.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from ...core.data.settings import SplitConfig
from .api import ApiError

MAX_ITEMS = 10_000  # list items or dictionary entries in one field
MAX_TEXT = 1_000  # characters in one text value

T = TypeVar("T")


def clamp(low: float, high: float, message: str = "") -> dict[str, Any]:
    """Field metadata: numbers outside [low, high] are moved to the nearest end."""
    return {"clamp": (low, high), "message": message}


def load(schema: type[T], source: Mapping[str, Any]) -> T:
    """``schema`` filled from ``source``; ApiError (BAD_REQUEST) for an invalid field."""
    hints = get_type_hints(schema)
    values: dict[str, Any] = {}
    for field in dataclasses.fields(schema):  # type: ignore[arg-type]
        value = source.get(field.name)
        if value is None:
            required = (field.default is dataclasses.MISSING
                        and field.default_factory is dataclasses.MISSING)
            if required:
                raise ApiError(f"'{field.name}' is required.")
            continue
        values[field.name] = _convert(field.name, value, hints[field.name], field.metadata)
    return schema(**values)


def _convert(name: str, value: Any, kind: Any, metadata: Mapping[str, Any]) -> Any:
    origin = get_origin(kind)
    if origin in (Union, types.UnionType):  # "X | None": None was handled by load()
        kind = next(arg for arg in get_args(kind) if arg is not type(None))
        origin = get_origin(kind)
    if kind is int or kind is float:
        return _number(name, value, kind, metadata)
    if kind is str:
        return _text(name, value)
    if origin is list:
        if not isinstance(value, list):
            raise ApiError(f"'{name}' must be a list.")
        if len(value) > MAX_ITEMS:
            raise ApiError(f"'{name}' has too many items (max {MAX_ITEMS:,}).")
        return [_text(name, _scalar(name, item)) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise ApiError(f"'{name}' must be an object.")
        if len(value) > MAX_ITEMS:
            raise ApiError(f"'{name}' has too many entries (max {MAX_ITEMS:,}).")
        checked = {}
        for key, item in value.items():
            _text(name, key)
            item = _scalar(name, item, allow_none=True)
            checked[key] = _text(name, item) if isinstance(item, str) else item
        return checked
    raise TypeError(f"schema field {name!r}: unsupported type {kind!r}")  # a programming error


def _number(name: str, value: Any, kind: type, metadata: Mapping[str, Any]) -> int | float:
    whole = kind is int
    message = metadata.get("message") or (f"'{name}' must be a whole number." if whole
                                          else f"'{name}' must be a number.")
    if isinstance(value, bool):
        raise ApiError(message)
    try:
        number: int | float = int(value) if whole else float(value)
    except (TypeError, ValueError, OverflowError):
        raise ApiError(message) from None
    if not whole and number != number:  # NaN
        raise ApiError(message)
    if "clamp" in metadata:
        low, high = metadata["clamp"]
        number = max(low, min(high, number))
    return number


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ApiError(f"'{name}' must be text.")
    if len(value) > MAX_TEXT:
        raise ApiError(f"'{name}' is too long (max {MAX_TEXT:,} characters).")
    return value


def _scalar(name: str, value: Any, allow_none: bool = False) -> Any:
    """A list item or dictionary value: text or a number (numbers in lists become text)."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ApiError(f"'{name}' may only hold text or numbers.")
    return value if allow_none else str(value)


TEST_SIZE = clamp(0.1, 0.5, "'test_size' must be a number between 0.1 and 0.5.")
DEFAULT_TEST_SIZE = SplitConfig().test_size


# --- One schema per route --------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PreviewRequest:
    rows: int = dataclasses.field(default=10, metadata=clamp(1, 100))


@dataclasses.dataclass(frozen=True)
class ClassesRequest:
    target: str = ""


@dataclasses.dataclass(frozen=True)
class FilterRequest:
    target: str = ""
    remove: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class DistributionRequest:
    target: str = ""
    test_size: float = dataclasses.field(default=DEFAULT_TEST_SIZE, metadata=TEST_SIZE)


@dataclasses.dataclass(frozen=True)
class BalancingPreviewRequest:
    target: str = ""
    balancing: str = ""
    features: list[str] | None = None
    test_size: float = dataclasses.field(default=DEFAULT_TEST_SIZE, metadata=TEST_SIZE)


@dataclasses.dataclass(frozen=True)
class FeatureScoresRequest:
    target: str = ""
    k: int = dataclasses.field(default=10, metadata=clamp(1, 10_000))
    test_size: float = dataclasses.field(default=DEFAULT_TEST_SIZE, metadata=TEST_SIZE)


@dataclasses.dataclass(frozen=True)
class TrainRequest:
    target: str = ""
    features: list[str] | None = None
    models: list[str] | None = None
    balancing: str = ""
    test_size: float = dataclasses.field(default=DEFAULT_TEST_SIZE, metadata=TEST_SIZE)


@dataclasses.dataclass(frozen=True)
class ConfusionRequest:
    model: str = ""


@dataclasses.dataclass(frozen=True)
class PredictRequest:
    values: dict[str, str | int | float | None] | None = None
    model: str = ""


@dataclasses.dataclass(frozen=True)
class ExportRequest:
    name: str = ""
