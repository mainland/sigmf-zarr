"""Canonical integrity helpers for logical SigMF-Zarr content."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from itertools import product
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from zarr.core.array import Array
from zarr.core.group import Group

INTEGRITY_ATTR = "integrity"
"""Group attribute containing SigMF-Zarr-native integrity hashes."""

INTEGRITY_ALGORITHM = "sha512"
"""Hash algorithm used by the initial integrity format."""

INTEGRITY_VERSION = 1
"""Canonical integrity payload version."""


def integrity_is_supported(value: object) -> bool:
    """Return whether an integrity object uses this format's algorithm.

    Args:
        value: Candidate integrity metadata object.

    Returns:
        True for a dictionary declaring the supported algorithm and version.
    """
    if not isinstance(value, dict):
        return False
    version = value.get("version")
    return (
        value.get("algorithm") == INTEGRITY_ALGORITHM
        and type(version) is int
        and version == INTEGRITY_VERSION
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON-compatible value in the format's canonical form.

    Args:
        value: JSON-compatible value to encode.

    Returns:
        UTF-8 encoded JSON with sorted keys and no insignificant whitespace.

    Raises:
        TypeError: If the value is not JSON serializable.
        ValueError: If the value contains a non-finite float.
    """
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_dtype(dtype: object) -> str:
    """Return a stable logical dtype identifier.

    Args:
        dtype: Zarr or NumPy dtype object.

    Returns:
        Canonical dtype string.
    """
    try:
        numpy_dtype = np.dtype(cast(Any, dtype))
    except TypeError:
        return str(dtype)
    # Numeric hashes use little-endian values, so the descriptor must use the
    # same byte order instead of exposing the host or store byte order.
    if numpy_dtype.kind in "biufcmM":
        return str(numpy_dtype.newbyteorder("<").str)
    return str(numpy_dtype.str)


def _iter_array_blocks(array: Array) -> Iterator[npt.NDArray[Any]]:
    """Yield array values in bounded C-order blocks.

    Args:
        array: Zarr array to read.

    Yields:
        NumPy arrays whose concatenated elements follow logical C order.
    """
    shape = tuple(int(size) for size in array.shape)
    if not shape:
        yield np.asarray(array[()])
        return
    if any(size == 0 for size in shape):
        return

    try:
        itemsize = max(int(np.dtype(cast(Any, array.dtype)).itemsize), 1)
    except TypeError:
        itemsize = 16
    # Bound temporary allocations without making the digest depend on Zarr's
    # physical chunks, which may differ across equivalent stores.
    target_bytes = 16 * 1024 * 1024
    trailing_elements = math.prod(shape[1:])
    trailing_bytes = trailing_elements * itemsize
    if trailing_bytes <= target_bytes:
        rows_per_block = max(target_bytes // max(trailing_bytes, 1), 1)
        for start in range(0, shape[0], rows_per_block):
            key: tuple[slice, ...] = (
                slice(start, min(start + rows_per_block, shape[0])),
                *(slice(None) for _ in shape[1:]),
            )
            yield np.asarray(array[key])
        return

    # A complete trailing plane may exceed the target. Read bounded segments
    # of the final axis while preserving logical C order.
    chunks = getattr(array, "chunks", None)
    block_size = shape[-1]
    if isinstance(chunks, tuple) and chunks:
        block_size = max(int(chunks[-1]), 1)
    prefixes = product(*(range(size) for size in shape[:-1]))
    for prefix in prefixes:
        for start in range(0, shape[-1], block_size):
            block_key: tuple[int | slice, ...] = (
                *prefix,
                slice(start, min(start + block_size, shape[-1])),
            )
            yield np.asarray(array[block_key])


def _update_array_values(
    digest: Any,
    values: npt.NDArray[Any],
) -> None:
    """Add canonical array values to a digest.

    Args:
        digest: Hash object to update.
        values: Array values in logical C order.

    Raises:
        TypeError: If an object value is not JSON serializable.
        ValueError: If an object value contains a non-finite float.
    """
    if values.dtype.kind in "biufcmM":
        dtype = values.dtype.newbyteorder("<")
        digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
        return
    if values.dtype.kind == "S":
        digest.update(np.ascontiguousarray(values).tobytes())
        return

    # Length prefixes keep adjacent variable-length encodings unambiguous.
    for value in values.reshape(-1):
        scalar = value.item() if isinstance(value, np.generic) else value
        encoded = canonical_json_bytes(scalar)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def calculate_array_sha512(array: Array) -> str:
    """Hash one logical Zarr array independently of chunk encoding.

    The digest includes logical shape and dtype, then values in C order.
    Numeric values use little-endian bytes so the result is independent of
    host byte order and physical Zarr format.

    Args:
        array: Zarr array to hash.

    Returns:
        Lowercase SHA-512 hexadecimal digest.
    """
    descriptor = {
        "dtype": _canonical_dtype(array.dtype),
        "shape": [int(size) for size in array.shape],
        "version": INTEGRITY_VERSION,
    }
    digest = hashlib.sha512()
    # Domain separation prevents an array payload from colliding with another
    # canonical object that happens to contain the same bytes.
    digest.update(b"sigmf-zarr-array-v1\0")
    digest.update(canonical_json_bytes(descriptor))
    digest.update(b"\0")
    for block in _iter_array_blocks(array):
        _update_array_values(digest, block)
    return digest.hexdigest()


def _group_metadata_payload(
    group: Group,
    *,
    exclude_array_values: frozenset[str],
    exclude_groups: frozenset[str],
    prefix: str = "",
) -> dict[str, object]:
    """Build a canonical logical metadata payload for a Zarr group.

    Args:
        group: Group to describe.
        exclude_array_values: Relative array paths whose values are omitted.
        exclude_groups: Immediate child groups omitted from the payload.
        prefix: Relative path used during recursion.

    Returns:
        JSON-compatible metadata payload.
    """
    attrs = dict(group.attrs)
    # A resource hash cannot include the field that stores that hash.
    attrs.pop(INTEGRITY_ATTR, None)
    arrays: dict[str, object] = {}
    # Zarr does not guarantee member iteration order across store backends.
    for name in sorted(group.array_keys()):
        array = group[name]
        assert isinstance(array, Array)
        path = f"{prefix}/{name}" if prefix else name
        entry: dict[str, object] = {
            "attrs": dict(array.attrs),
            "dtype": _canonical_dtype(array.dtype),
            "shape": [int(size) for size in array.shape],
        }
        if path not in exclude_array_values:
            entry["sha512"] = calculate_array_sha512(array)
        arrays[name] = entry

    groups: dict[str, object] = {}
    for name in sorted(group.group_keys()):
        # Group exclusions are intentionally relative to the requested root.
        if not prefix and name in exclude_groups:
            continue
        child = group[name]
        assert isinstance(child, Group)
        child_prefix = f"{prefix}/{name}" if prefix else name
        groups[name] = _group_metadata_payload(
            child,
            exclude_array_values=exclude_array_values,
            exclude_groups=frozenset(),
            prefix=child_prefix,
        )
    return {
        "arrays": arrays,
        "attrs": attrs,
        "groups": groups,
    }


def calculate_group_metadata_sha512(
    group: Group,
    *,
    exclude_array_values: frozenset[str] = frozenset(),
    exclude_groups: frozenset[str] = frozenset(),
) -> str:
    """Hash the canonical logical metadata represented by a Zarr group.

    Array descriptors and attributes are always included. Array values are
    included unless their relative path appears in ``exclude_array_values``.

    Args:
        group: Zarr group to hash.
        exclude_array_values: Relative array paths whose values are omitted.
        exclude_groups: Immediate child groups omitted from the payload.

    Returns:
        Lowercase SHA-512 hexadecimal digest.
    """
    payload = _group_metadata_payload(
        group,
        exclude_array_values=exclude_array_values,
        exclude_groups=exclude_groups,
    )
    digest = hashlib.sha512()
    digest.update(b"sigmf-zarr-metadata-v1\0")
    digest.update(canonical_json_bytes(payload))
    return digest.hexdigest()


__all__ = [
    "INTEGRITY_ALGORITHM",
    "INTEGRITY_ATTR",
    "INTEGRITY_VERSION",
    "calculate_array_sha512",
    "calculate_group_metadata_sha512",
    "canonical_json_bytes",
    "integrity_is_supported",
]
