"""Tests for JSON validation, read-only views, and canonical hashing."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import zarr

import sigmf_zarr.integrity as integrity
from sigmf_zarr.integrity import (
    array_slab_length,
    calculate_array_sha512,
    canonical_json_bytes,
)
from sigmf_zarr.json import json_object, json_object_list, json_value
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.store import ZarrFormat

ARANGE_I4_AXIS0_SHA512 = (
    "50b018660452b2c2c5e6a72784a575c5dbdda743409a07875b47ef83ad404269"
    "d2f768ce240a3ccbbc738e725948d6f4c5b4e232a81ab17f053f3e518a4f3587"
)
"""Digest of `arange(6, dtype="<i4").reshape(2, 3)` with slab axis 0."""

ARANGE_I4_AXIS1_SHA512 = (
    "ade9001ae0a49a6745ed0c19f968b8cab79e6ed92f981cdc6f1270dd9dedb754"
    "d4b6db91830d0ae66ef40c45cd89d3bddf672baa0f21159039cd5c6e611f2c06"
)
"""Digest of the same array with slab axis 1."""

SCALAR_F8_SHA512 = (
    "38b64e08bb33bde4c5a9b33b525a55c58aa57d7e32aaa1a099f689e0c35bb7a3"
    "c55870ddd138a890b613775986607c5cbd41caed661d9acb1c4cba1b34fafd7f"
)
"""Digest of a zero-dimensional float64 array containing 1.5."""

EMPTY_F4_SHA512 = (
    "34954acb23e398e340e38fa5372b0a9c9ddf7a449db7f8cc5abb5c24fe52c7eb"
    "e27d7639f339a2cb61c309378e7d18e6d2e2250206c4ee3d1fe094774a0b3c6b"
)
"""Digest of a float32 array with shape `(0, 3)`."""

STRINGS_SHA512 = (
    "a79fab06523df9ba2e141baf44a3063f09b96c177fc5fd24c6bcdd2f0b2ec2ae"
    "cebeb1094a920a24f7a59290da697600a9756b0254e079016716ef605f64b777"
)
"""Digest of the object string array `["a", "bc"]`."""


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


def test_array_hash_known_answers() -> None:
    """Array digests must match fixed vectors for each hashing rule."""
    values = np.arange(6, dtype="<i4").reshape(2, 3)
    assert calculate_array_sha512(values) == ARANGE_I4_AXIS0_SHA512
    assert calculate_array_sha512(values, axis=1) == ARANGE_I4_AXIS1_SHA512
    assert calculate_array_sha512(values, axis=-1) == ARANGE_I4_AXIS1_SHA512
    assert (
        calculate_array_sha512(values.astype(">i4")) == ARANGE_I4_AXIS0_SHA512
    )
    assert calculate_array_sha512(np.full((), 1.5)) == SCALAR_F8_SHA512
    assert (
        calculate_array_sha512(np.zeros((0, 3), dtype=np.float32))
        == EMPTY_F4_SHA512
    )
    strings = np.array(["a", "bc"], dtype=object)
    assert calculate_array_sha512(strings) == STRINGS_SHA512


def test_array_hash_matches_independent_construction() -> None:
    """A one-slab digest must equal its construction from the definition."""
    values = np.arange(6, dtype="<i4").reshape(2, 3)
    descriptor = {
        "dtype": "<i4",
        "shape": [2, 3],
        "slab_axis": 0,
        "slab_length": 2**24 // 12,
        "version": 1,
    }
    slab_digest = hashlib.sha512(values.tobytes()).digest()
    expected = hashlib.sha512(
        b"sigmf-zarr-array-v1\0"
        + canonical_json_bytes(descriptor)
        + b"\0"
        + slab_digest
    ).hexdigest()
    assert calculate_array_sha512(values) == expected


def test_array_slab_length_follows_logical_layout() -> None:
    """Slab length depends on dtype and the other dimensions only."""
    assert array_slab_length(np.dtype("<f4"), (2, 2**30), 1) == 2**21
    assert array_slab_length(np.dtype("<f4"), (2**30, 2), 0) == 2**21
    assert array_slab_length(np.dtype("<c16"), (2**25, 2**21), 0) == 1
    assert array_slab_length(np.dtype("O"), (10,), 0) == 2**20
    assert array_slab_length(np.dtype("<f4"), (0, 8), 1) == 2**24 // 4


def test_array_hash_rejects_axis_outside_shape() -> None:
    """A slab axis must name an array dimension."""
    with pytest.raises(ValueError, match="axis"):
        calculate_array_sha512(np.zeros((2, 3)), axis=2)


class RecordingArray:
    """NumPy-backed array that records the slab-axis ranges it reads."""

    def __init__(self, values: np.ndarray, *, chunks: tuple[int, ...]) -> None:
        """Initialize a recording array.

        Args:
            values: Backing values.
            chunks: Reported chunk shape.
        """
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.chunks = chunks
        self.reads: list[tuple[int, int]] = []

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        """Return values and record the range read along axis 1.

        Args:
            key: Basic selection.

        Returns:
            Selected values.
        """
        self.reads.append((key[1].start, key[1].stop))
        return self.values[key]


def test_array_hash_reads_each_chunk_once_and_ignores_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-plane slabs must not reread chunks or depend on chunk layout.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    # Small slabs exercise slab boundaries that do not align with chunks.
    monkeypatch.setattr(integrity, "ARRAY_SLAB_BYTES", 40)
    values = np.arange(4 * 37, dtype="<f4").reshape(4, 37)
    assert array_slab_length(values.dtype, values.shape, 1) == 2
    expected = calculate_array_sha512(values, axis=1)
    for chunk in (1, 3, 5, 37, 64):
        array = RecordingArray(values, chunks=(4, chunk))
        # RecordingArray is a duck-typed stand-in for a Zarr array.
        assert calculate_array_sha512(array, axis=1) == expected  # type: ignore[arg-type]
        starts = [start for start, _ in array.reads]
        stops = [stop for _, stop in array.reads]
        # Reads tile the axis once, in order, on chunk boundaries.
        assert starts == [0, *stops[:-1]]
        assert stops[-1] == 37
        assert all(stop == 37 or stop % chunk == 0 for stop in stops)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_hash_is_independent_of_zarr_chunks(
    monkeypatch: pytest.MonkeyPatch, zarr_format: ZarrFormat
) -> None:
    """Equivalent Zarr arrays must hash equally for any chunk shape.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        zarr_format: Physical Zarr format.
    """
    monkeypatch.setattr(integrity, "ARRAY_SLAB_BYTES", 64)
    values = np.arange(3 * 50, dtype=">i2").reshape(3, 50)
    expected = calculate_array_sha512(values, axis=1)
    for chunks in ((3, 7), (1, 50), (2, 16)):
        array = zarr.create_array(
            zarr.storage.MemoryStore(),
            data=values,
            chunks=chunks,
            zarr_format=zarr_format,
        )
        assert calculate_array_sha512(array, axis=1) == expected
        assert calculate_array_sha512(ReadOnlyArray(array), axis=1) == (
            expected
        )


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
