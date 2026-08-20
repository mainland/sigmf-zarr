"""Tests for JSON validation, read-only views, and canonical hashing."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import zarr

from sigmf_zarr.integrity import canonical_json_bytes
from sigmf_zarr.json import json_object, json_object_list, json_value
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup


def test_json_value_detaches_nested_containers() -> None:
    """Validated JSON must not share containers with the input."""
    source: dict[str, Any] = {"list": [1, {"x": None}], "flag": True}
    result = json_value(source, name="value")
    assert result == source
    assert isinstance(result, dict)
    result_list = result["list"]
    assert isinstance(result_list, list)
    assert result_list is not source["list"]
    assert result_list[1] is not source["list"][1]


@pytest.mark.parametrize(
    "value",
    [
        {1: "integer key"},
        {"x": float("nan")},
        [float("inf")],
        ("tuple",),
        {"nested": [object()]},
    ],
)
def test_json_value_rejects_non_json_values(value: object) -> None:
    """Non-string keys, non-finite numbers, and other types are rejected.

    Args:
        value: Invalid candidate.
    """
    with pytest.raises(ValueError, match="Expected value"):
        json_value(value, name="value")


def test_json_object_helpers_require_objects() -> None:
    """Object and object-list helpers reject other JSON values."""
    assert json_object({"a": 1}, name="value") == {"a": 1}
    assert json_object_list([{"a": 1}], name="value") == [{"a": 1}]
    with pytest.raises(ValueError, match="JSON object"):
        json_object([], name="value")
    with pytest.raises(ValueError, match="list of JSON objects"):
        json_object_list({}, name="value")
    with pytest.raises(ValueError, match=r"value\[0\]"):
        json_object_list([1], name="value")


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        ({"b": 1, "a": 2}, b'{"a":2,"b":1}'),
        # Code-point order places U+FF5E before U+1F600. UTF-16 order would
        # place the surrogate pair first.
        (
            {"\U0001f600": 2, "\uff5e": 1},
            '{"\uff5e":1,"\U0001f600":2}'.encode(),
        ),
        (
            '"\\\b\f\n\r\t\x01/\x7fé',
            '"\\"\\\\\\b\\f\\n\\r\\t\\u0001/\x7fé"'.encode(),
        ),
        ([1, 1.0, -0.0, 0.0001, 1e-05, 2.5e-07, 1e16], None),
        ([True, False, None], b"[true,false,null]"),
    ],
)
def test_canonical_json_matches_format_specification(
    value: object, encoded: bytes | None
) -> None:
    """Canonical JSON must follow the rules in the format specification.

    Args:
        value: JSON value to encode.
        encoded: Expected bytes, or None for the number vector below.
    """
    expected = (
        b"[1,1.0,-0.0,0.0001,1e-05,2.5e-07,1e+16]"
        if encoded is None
        else encoded
    )
    assert canonical_json_bytes(value) == expected


def test_canonical_json_rejects_non_finite_numbers() -> None:
    """Non-finite numbers have no canonical encoding."""
    with pytest.raises(ValueError, match="not JSON compliant"):
        canonical_json_bytes([math.inf])


def test_read_only_array_rejects_writes() -> None:
    """Array views must reject writes and unreviewed members."""
    array = zarr.create_array(
        zarr.storage.MemoryStore(), data=np.arange(4), chunks=(2,)
    )
    array.attrs["nested"] = {"values": [1]}
    view = ReadOnlyArray(array)
    np.testing.assert_array_equal(view[:], np.arange(4))
    write: Callable[[], None]
    for write in (
        lambda: view.__setitem__(0, 1),
        lambda: view.oindex.__setitem__([0], 1),
        lambda: view.vindex.__setitem__([0], 1),
    ):
        with pytest.raises(TypeError, match="read-only"):
            write()
    with pytest.raises(AttributeError, match="mutation context"):
        view.resize((8,))  # type: ignore[operator]
    detached = view.attrs["nested"]
    detached["values"].append(2)
    assert array.attrs["nested"] == {"values": [1]}
    materialized = np.asarray(view)
    materialized[0] = 99
    assert array[0] == 0


def test_read_only_group_wraps_descendants() -> None:
    """Group views must reject changes and wrap every descendant."""
    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    child = group.create_group("child")
    child.create_array("values", data=np.arange(3), chunks=(3,))
    view = ReadOnlyGroup(group)
    assert isinstance(view["child"], ReadOnlyGroup)
    assert isinstance(view["child/values"], ReadOnlyArray)
    assert all(
        isinstance(member, ReadOnlyArray | ReadOnlyGroup)
        for _, member in view.members(max_depth=None)
    )
    with pytest.raises(TypeError, match="read-only"):
        view["new"] = np.arange(2)
    with pytest.raises(TypeError, match="read-only"):
        del view["child"]
    with pytest.raises(AttributeError):
        view.create_group("new")  # type: ignore[operator]
