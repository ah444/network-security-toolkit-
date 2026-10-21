"""v5.1 step 1.2b: request schemas (S9) - types, required fields, clamps and size caps."""

import dataclasses

import pytest

from toolkit.web.api import ApiError
from toolkit.web.schemas import (
    MAX_ITEMS,
    MAX_TEXT,
    PredictRequest,
    PreviewRequest,
    TrainRequest,
    clamp,
    load,
)


@dataclasses.dataclass(frozen=True)
class Needs:
    name: str
    count: int = dataclasses.field(default=3, metadata=clamp(1, 10))
    share: float = 0.5


def _refused(schema, source):
    with pytest.raises(ApiError) as caught:
        load(schema, source)
    assert caught.value.code == "BAD_REQUEST" and caught.value.status == 400
    return caught.value.message


def test_defaults_and_conversion():
    assert load(Needs, {"name": "x"}) == Needs("x", 3, 0.5)
    assert load(Needs, {"name": "x", "count": "7", "share": "0.25"}) == Needs("x", 7, 0.25)


def test_required_field_named():
    assert _refused(Needs, {}) == "'name' is required."


def test_wrong_types_name_the_field():
    assert "'count' must be a whole number" in _refused(Needs, {"name": "x", "count": "lots"})
    assert "'share' must be a number" in _refused(Needs, {"name": "x", "share": "half"})
    assert "'name' must be text" in _refused(Needs, {"name": 5})
    assert "'count'" in _refused(Needs, {"name": "x", "count": True}), "a bool is not a number"
    assert "'share'" in _refused(Needs, {"name": "x", "share": "nan"})


def test_numbers_are_clamped_not_refused():
    assert load(Needs, {"name": "x", "count": 10**9}).count == 10
    assert load(Needs, {"name": "x", "count": -5}).count == 1
    assert load(PreviewRequest, {"num_rows": 0}).num_rows == 1
    assert load(TrainRequest, {"test_size": 0.9}).test_size == 0.5


def test_lists_hold_text_and_are_capped():
    assert load(TrainRequest, {"input_features": ["a", 2, 3.5]}).input_features == ["a", "2", "3.5"]
    assert "must be a list" in _refused(TrainRequest, {"input_features": "a"})
    assert "only hold text or numbers" in _refused(TrainRequest, {"input_features": [["a"]]})
    assert "too many items" in _refused(TrainRequest, {"input_features": ["a"] * (MAX_ITEMS + 1)})
    assert "too long" in _refused(TrainRequest, {"input_features": ["a" * (MAX_TEXT + 1)]})


def test_texts_are_capped():
    assert "too long" in _refused(TrainRequest, {"target": "x" * (MAX_TEXT + 1)})
    assert load(TrainRequest, {"target": "x" * MAX_TEXT}).target == "x" * MAX_TEXT


def test_prediction_values():
    values = {"a": "1", "b": 2, "c": 2.5, "d": None}
    assert load(PredictRequest, {"values": values}).values == values
    assert "must be an object" in _refused(PredictRequest, {"values": [1, 2]})
    assert "only hold text or numbers" in _refused(PredictRequest, {"values": {"a": [1]}})
    too_many = {str(i): 1 for i in range(MAX_ITEMS + 1)}
    assert "too many entries" in _refused(PredictRequest, {"values": too_many})


def test_none_means_missing():
    assert load(TrainRequest, {"target": None, "input_features": None}) == TrainRequest()
