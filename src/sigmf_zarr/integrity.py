"""Canonical integrity helpers for logical SigMF-Zarr content."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from zarr.core.array import Array
from zarr.core.group import Group

from sigmf_zarr._consolidated import without_consolidated_metadata
from sigmf_zarr.readonly import ReadOnlyArray

INTEGRITY_ATTR = "integrity"
"""Group attribute containing SigMF-Zarr-native integrity hashes."""

INTEGRITY_ALGORITHM = "sha512"
"""Hash algorithm used by the initial integrity format."""

INTEGRITY_VERSION = 1
"""Canonical integrity payload version."""

ARRAY_SLAB_BYTES = 16 * 1024 * 1024
"""Nominal uncompressed size of one hashed array slab."""

VARIABLE_LENGTH_ITEMSIZE = 16
"""Nominal element size used to size slabs of variable-length values."""

STRING_DTYPE = "string"
"""Hash descriptor for arrays of variable-length UTF-8 strings."""

METADATA_DOMAIN = b"sigmf-zarr-metadata-v1\0"
"""Domain separator for recording and collection metadata hashes."""

STORE_DOMAIN = b"sigmf-zarr-store-v1\0"
"""Domain separator for the store-wide metadata hash."""


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
    # Zarr reads variable-length UTF-8 strings as StringDType in both formats,
    # but NumPy spells that dtype `|T16` in 2.0 and `StringDType()` later.
    if numpy_dtype.kind == "T":
        return STRING_DTYPE
    # Numeric hashes use little-endian values, so the descriptor must use the
    # same byte order instead of exposing the host or store byte order.
    # Unicode values are hashed as text, so their byte order is irrelevant.
    if numpy_dtype.kind in "biufcmMU":
        return str(numpy_dtype.newbyteorder("<").str)
    return str(numpy_dtype.str)


def _nominal_itemsize(dtype: object) -> int:
    """Return the element size used to choose a slab length.

    Args:
        dtype: Zarr or NumPy dtype object.

    Returns:
        Fixed element size in bytes, or the nominal size for variable-length
        strings and objects.
    """
    try:
        numpy_dtype = np.dtype(cast(Any, dtype))
    except TypeError:
        return VARIABLE_LENGTH_ITEMSIZE
    # NumPy kinds O and T are Python objects and variable-width strings.
    if numpy_dtype.kind in {"O", "T"} or numpy_dtype.itemsize == 0:
        return VARIABLE_LENGTH_ITEMSIZE
    return int(numpy_dtype.itemsize)


def _decoded_chunk_nbytes(array: Array[Any] | ReadOnlyArray) -> int:
    """Return the nominal decoded size of one stored chunk or shard.

    Reading any element decodes at least one whole chunk, and a sharded
    array can allocate a buffer for a whole shard. A chunk can be much
    larger than the array, so this size can exceed the logical array size.
    Variable-length strings and objects count with the nominal element
    size, so the result bounds the number of decoded elements but not their
    total length.

    Args:
        array: Zarr or read-only array.

    Returns:
        Shard size for a sharded array, otherwise chunk size, in bytes.
    """
    # A read-only view types delegated members as `object`.
    shards = cast(tuple[int, ...] | None, array.shards)
    shape = shards if shards is not None else array.chunks
    return math.prod(int(size) for size in shape) * _nominal_itemsize(
        array.dtype
    )


def array_slab_length(dtype: object, shape: tuple[int, ...], axis: int) -> int:
    """Return the number of slab-axis positions hashed per slab.

    The length depends only on the logical dtype and shape, so equivalent
    arrays use the same slabs regardless of their physical chunks.

    Args:
        dtype: Zarr or NumPy dtype object.
        shape: Logical array shape.
        axis: Nonnegative slab axis.

    Returns:
        Positive slab length.
    """
    position_elements = math.prod(
        size for dimension, size in enumerate(shape) if dimension != axis
    )
    position_bytes = _nominal_itemsize(dtype) * max(position_elements, 1)
    return max(1, ARRAY_SLAB_BYTES // position_bytes)


def _axis_key(
    ndim: int, axis: int, start: int, stop: int
) -> tuple[slice, ...]:
    """Select a contiguous range along one axis and all of every other axis.

    Args:
        ndim: Array rank.
        axis: Selected axis.
        start: First selected position.
        stop: Exclusive stop position.

    Returns:
        Basic selection key.
    """
    key = [slice(None)] * ndim
    key[axis] = slice(start, stop)
    return tuple(key)


def _iter_slabs(
    array: Array[Any] | ReadOnlyArray | npt.NDArray[Any],
    *,
    axis: int,
    length: int,
) -> Iterator[npt.NDArray[Any]]:
    """Yield consecutive slabs along one axis, reading each chunk once.

    Reads extend to the next physical chunk boundary along the slab axis.
    The unused remainder is carried into the next slab, so a chunk that
    straddles a slab boundary is decoded only once. Memory is bounded by one
    slab plus one chunk.

    Args:
        array: Zarr, read-only, or NumPy array with at least one dimension.
        axis: Nonnegative slab axis.
        length: Positions per slab.

    Yields:
        Slab values, each with the complete extent of every other axis.
    """
    shape = tuple(int(size) for size in array.shape)
    ndim = len(shape)
    total = shape[axis]
    chunks = getattr(array, "chunks", None)
    chunk = (
        max(int(chunks[axis]), 1)
        if isinstance(chunks, tuple) and len(chunks) == ndim
        else length
    )
    buffered: npt.NDArray[Any] | None = None
    buffered_start = 0
    buffered_stop = 0
    for start in range(0, total, length):
        stop = min(start + length, total)
        if buffered_stop < stop:
            read_stop = min(total, -(-stop // chunk) * chunk)
            block = np.asarray(
                array[_axis_key(ndim, axis, buffered_stop, read_stop)]
            )
            buffered = (
                block
                if buffered is None
                else np.concatenate((buffered, block), axis=axis)
            )
            buffered_stop = read_stop
        assert buffered is not None
        yield buffered[
            _axis_key(
                ndim, axis, start - buffered_start, stop - buffered_start
            )
        ]
        buffered = buffered[
            _axis_key(
                ndim,
                axis,
                stop - buffered_start,
                buffered_stop - buffered_start,
            )
        ]
        buffered_start = stop


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


def _scalar_values(
    array: Array[Any] | ReadOnlyArray | npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Return the value of a zero-dimensional array with its declared dtype.

    Zarr returns a scalar for this selection. Converting that scalar with
    `np.asarray` would shorten byte strings to their content and expand an
    object value that is a list, so the value is placed in a new array.

    Args:
        array: Zarr, read-only, or NumPy array with shape `()`.

    Returns:
        Zero-dimensional NumPy array.
    """
    if isinstance(array, np.ndarray):
        return array
    values = np.empty((), dtype=array.dtype)
    values[()] = array[()]
    return values


def calculate_array_sha512(
    array: Array[Any] | ReadOnlyArray | npt.NDArray[Any],
    *,
    axis: int = 0,
) -> str:
    """Hash one logical array independently of chunk encoding.

    The array is divided into consecutive slabs along `axis`. Each slab
    holds `array_slab_length()` positions with the complete extent of every
    other axis, and its values are hashed in C order. The digest covers the
    logical dtype, shape, slab layout, and the ordered slab digests. Numeric
    values use little-endian bytes, so the result is independent of host
    byte order, physical chunks, and Zarr format.

    Args:
        array: Zarr, read-only, or NumPy array to hash.
        axis: Slab axis. Sample arrays use their growth axis, which is
            `item` for batched recordings and `time` otherwise.

    Returns:
        Lowercase SHA-512 hexadecimal digest.

    Raises:
        ValueError: If `axis` is outside a nonscalar array.
    """
    shape = tuple(int(size) for size in array.shape)
    slab_axis: int | None = None
    slab_length = 1
    if shape:
        if not -len(shape) <= axis < len(shape):
            raise ValueError(f"axis {axis} is outside array shape {shape}")
        slab_axis = axis % len(shape)
        slab_length = array_slab_length(array.dtype, shape, slab_axis)
    descriptor = {
        "dtype": _canonical_dtype(array.dtype),
        "shape": list(shape),
        "slab_axis": slab_axis,
        "slab_length": slab_length,
        "version": INTEGRITY_VERSION,
    }
    digest = hashlib.sha512()
    # Domain separation prevents an array payload from colliding with another
    # canonical object that happens to contain the same bytes.
    digest.update(b"sigmf-zarr-array-v1\0")
    digest.update(canonical_json_bytes(descriptor))
    digest.update(b"\0")
    if slab_axis is None:
        slabs: Iterator[npt.NDArray[Any]] = iter((_scalar_values(array),))
    elif math.prod(shape) == 0:
        slabs = iter(())
    else:
        slabs = _iter_slabs(array, axis=slab_axis, length=slab_length)
    for slab in slabs:
        slab_digest = hashlib.sha512()
        _update_array_values(slab_digest, slab)
        digest.update(slab_digest.digest())
    return digest.hexdigest()


def group_metadata_payload(
    group: Group,
    *,
    exclude_array_values: frozenset[str] = frozenset(),
    exclude_groups: frozenset[str] = frozenset(),
    exclude_integrity: bool = True,
    prefix: str = "",
) -> dict[str, object]:
    """Build a canonical logical metadata payload for a Zarr group.

    Args:
        group: Group to describe.
        exclude_array_values: Relative array paths whose values are omitted.
        exclude_groups: Immediate child groups omitted from the payload.
        exclude_integrity: Whether to omit this group's own `integrity`
            attribute. Child groups always keep theirs.
        prefix: Relative path used during recursion.

    Returns:
        JSON-compatible metadata payload.
    """
    # Hashes cover the node metadata documents, never a consolidated copy.
    group = without_consolidated_metadata(group)
    attrs = dict(group.attrs)
    # A resource hash cannot include the field that stores that hash. Any
    # other `integrity` attribute is ordinary content.
    if exclude_integrity:
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
        groups[name] = group_metadata_payload(
            child,
            exclude_array_values=exclude_array_values,
            exclude_groups=frozenset(),
            exclude_integrity=False,
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
    payload = group_metadata_payload(
        group,
        exclude_array_values=exclude_array_values,
        exclude_groups=exclude_groups,
    )
    return calculate_payload_sha512(payload, domain=METADATA_DOMAIN)


def calculate_payload_sha512(payload: object, *, domain: bytes) -> str:
    """Hash a canonical JSON payload under a domain separator.

    Args:
        payload: JSON-compatible payload.
        domain: Prefix identifying the payload kind.

    Returns:
        Lowercase SHA-512 hexadecimal digest.
    """
    digest = hashlib.sha512()
    digest.update(domain)
    digest.update(canonical_json_bytes(payload))
    return digest.hexdigest()


__all__ = [
    "ARRAY_SLAB_BYTES",
    "INTEGRITY_ALGORITHM",
    "INTEGRITY_ATTR",
    "INTEGRITY_VERSION",
    "METADATA_DOMAIN",
    "STORE_DOMAIN",
    "STRING_DTYPE",
    "VARIABLE_LENGTH_ITEMSIZE",
    "array_slab_length",
    "calculate_array_sha512",
    "calculate_group_metadata_sha512",
    "calculate_payload_sha512",
    "canonical_json_bytes",
    "group_metadata_payload",
    "integrity_is_supported",
]
