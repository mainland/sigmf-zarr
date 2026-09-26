"""Types and helpers shared by SigMF-Zarr store implementations."""

import re
import unicodedata
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from zarr.core.array import Array
from zarr.core.group import Group

type ChecksumName = Literal["crc32c"]
"""Supported chunk-level sample checksum names."""

type ZarrFormat = Literal[2, 3]
"""Supported physical Zarr storage formats."""

DEFAULT_SAMPLE_CHUNK_BYTES = 1024 * 1024
"""Target logical chunk size for recordings created without explicit chunks."""

ZARR_METADATA_KEYS = frozenset(
    {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}
)
"""Zarr format 2 and Zarr format 3 metadata document names."""

_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]+(?:://|::)")
"""Explicit scheme that starts a store URL, as in `s3://` or `zip::`.

A scheme has at least two characters, so a Windows drive letter is part of
a path.
"""


def is_store_url(location: str) -> bool:
    """Return whether a store location string is a URL.

    A string is a URL only when it starts with an explicit scheme followed
    by `://`, as in `s3://bucket/key`, `file:///data/a.zarr`, or
    `memory://store`, or by `::`, as in the chained URL
    `simplecache::s3://bucket/key`. Every other string is a local path.

    Args:
        location: Store path or URL.

    Returns:
        True when Zarr should parse `location` as a URL.
    """
    return _URL_SCHEME.match(location) is not None


def store_location(location: str | PathLike[str]) -> str | Path:
    """Return a store location in the form that Zarr interprets as intended.

    Zarr parses every string as a URL and drops a fragment or query from
    it. A local path therefore goes to Zarr as a `Path`, which keeps `#`
    and `?` literal.

    Args:
        location: Store path or URL.

    Returns:
        A URL string unchanged, or a local path as a `Path`.
    """
    if isinstance(location, str) and is_store_url(location):
        return location
    return Path(location)


_INVALID_NAME_CATEGORIES = frozenset({"Cc", "Cs"})
"""Unicode categories that names must not contain.

Category `Cc` holds the C0 and C1 control characters, including NUL, line
feed, and escape. Category `Cs` holds surrogate code points, which Python
uses to represent undecodable bytes of a file name.
"""


def has_invalid_name_character(name: str) -> bool:
    """Return whether a name contains a control character or a surrogate.

    Resource names and array-name segments must be valid Unicode text that
    a terminal displays as written. A lone surrogate has no UTF-8 encoding,
    and a control character can forge command-line output.

    Args:
        name: Resource name, array name, or name segment.

    Returns:
        True when `name` contains a code point in Unicode category `Cc` or
        `Cs`.
    """
    return any(
        unicodedata.category(character) in _INVALID_NAME_CATEGORIES
        for character in name
    )


INDEX_VALID_FIELD = "sigmf-zarr:valid"
"""Index attribute that marks an index as invalid when false."""

INDEX_INVALID_REASON_FIELD = "sigmf-zarr:invalid-reason"
"""Index attribute that explains why an index is invalid."""


def index_marker_error(name: str, attrs: Mapping[str, object]) -> str | None:
    """Describe why the mutation markers of an index prevent its use.

    Args:
        name: Index name used in the message.
        attrs: Index array attributes.

    Returns:
        Error message for a malformed marker or an index marked invalid, or
        ``None`` when the index is not marked invalid.
    """
    valid = attrs.get(INDEX_VALID_FIELD, True)
    # A JSON number or string is not a Boolean, even when it is truthy.
    if not isinstance(valid, bool):
        return (
            f"Index {name!r} attribute {INDEX_VALID_FIELD!r} must be a "
            f"Boolean, got {valid!r}"
        )
    reason = attrs.get(INDEX_INVALID_REASON_FIELD, "unknown reason")
    if not isinstance(reason, str):
        return (
            f"Index {name!r} attribute {INDEX_INVALID_REASON_FIELD!r} must "
            f"be a string, got {reason!r}"
        )
    if not valid:
        return f"Index {name!r} is invalid: {reason}"
    return None


def _python_dimensions[T](dimensions: T) -> T:
    """Convert NumPy integers in a shape, chunk, or shard tuple to `int`.

    Zarr format 3 treats a NumPy integer dimension as a nested sequence and
    fails with `TypeError`. Other values pass through unchanged, so Zarr
    reports them as before.

    Args:
        dimensions: Tuple or list of dimensions, or any other shape option.

    Returns:
        A sequence of the same type with NumPy integers replaced, or
        `dimensions` unchanged when it is not a tuple or list.
    """
    if not isinstance(dimensions, tuple | list):
        return dimensions
    converted = [
        int(value) if isinstance(value, np.integer) else value
        for value in dimensions
    ]
    return cast(
        T, tuple(converted) if isinstance(dimensions, tuple) else converted
    )


def _validate_resource_name(name: str, *, kind: str) -> None:
    """Require a resource identifier to name one child group.

    Args:
        name: Recording or collection identifier.
        kind: Resource kind used in the error message.

    Raises:
        ValueError: If the identifier is empty, contains path components, is
            a reserved Zarr metadata document name, or contains a control
            character or lone surrogate.
    """
    # Untyped callers can pass any value.
    if (
        not isinstance(name, str)  # type: ignore[redundant-expr]
        or not name
        or name in {".", ".."}
        or any(character in name for character in ("/", "\\", "\x00"))
    ):
        raise ValueError(
            f"{kind} name must be a single nonempty Zarr group name"
        )
    if name in ZARR_METADATA_KEYS:
        raise ValueError(
            f"{kind} name {name!r} is reserved for Zarr metadata documents"
        )
    if has_invalid_name_character(name):
        raise ValueError(
            f"{kind} name {name!r} must not contain control characters or "
            "lone surrogates"
        )


def _validate_array_name(name: str, *, kind: str) -> None:
    """Require a name to identify one array below its parent group.

    Names may contain nested groups separated by `/`. An empty name or empty
    segment would name the parent group itself. A segment equal to a Zarr
    metadata document name would name that document.

    Args:
        name: Index, split, or extension-array name relative to its group.
        kind: Name kind used in the error message.

    Raises:
        ValueError: If the name is empty, has an empty, `.`, or `..`
            segment, starts or ends with `/`, contains NUL, has a segment
            that is a reserved Zarr metadata document name, or contains a
            control character or lone surrogate.
    """
    # Untyped callers can pass any value.
    if (
        not isinstance(name, str)  # type: ignore[redundant-expr]
        or "\x00" in name
        or any(segment in {"", ".", ".."} for segment in name.split("/"))
    ):
        raise ValueError(
            f"{kind} name {name!r} must be one or more nonempty segments "
            "separated by '/', without '.' or '..' segments or NUL"
        )
    reserved = sorted(set(name.split("/")) & ZARR_METADATA_KEYS)
    if reserved:
        raise ValueError(
            f"{kind} name {name!r} has segment {reserved[0]!r}, which is "
            "reserved for Zarr metadata documents"
        )
    if has_invalid_name_character(name):
        raise ValueError(
            f"{kind} name {name!r} must not contain control characters or "
            "lone surrogates"
        )


def _check_array_dtype(dtype: np.dtype[Any]) -> None:
    """Reject an array dtype that a store must not contain.

    Array hashes do not cover the field names, offsets, or field dtypes of a
    structured dtype, so the format forbids structured and void dtypes.

    Args:
        dtype: Dtype of the values to store.

    Raises:
        ValueError: If `dtype` is a structured or void dtype.
    """
    if dtype.kind == "V":
        raise ValueError(
            f"Structured and void dtypes are not supported, got {dtype.str!r}."
            " Store each field as a separate array"
        )


def _require_array(group: Group, name: str, *, kind: str) -> Array[Any]:
    """Return the array that a name identifies below a group.

    Args:
        group: Parent group.
        name: Array name relative to `group`.
        kind: Name kind used in error messages.

    Returns:
        Existing array.

    Raises:
        ValueError: If the name is malformed.
        KeyError: If no array has this name. A group is not an array.
    """
    _validate_array_name(name, kind=kind)
    if name not in group:
        raise KeyError(name)
    member = group[name]
    if not isinstance(member, Array):
        raise KeyError(f"{kind} {name!r} names a group, not an array")
    return member


def _check_array_target(
    group: Group, name: str, *, kind: str, overwrite: bool
) -> None:
    """Reject a name that cannot be created or replaced as one array.

    Callers check before any write, including hash invalidation, because
    replacing a group would delete every array nested under it, and a
    rejected call must leave every hash intact.

    A name below an existing array is rejected. Zarr stores the chunks of an
    array under the array's own prefix, as in `ids/c/0` for array `ids` in
    Zarr format 3, so such a name addresses that array's chunks.

    Args:
        group: Parent group.
        name: Array name relative to `group`.
        kind: Name kind used in error messages.
        overwrite: Whether the caller may replace an existing array.

    Raises:
        ValueError: If the name is malformed, an ancestor of the name is an
            existing array, the target exists and `overwrite` is false, or
            an overwrite targets an existing group.
    """
    _validate_array_name(name, kind=kind)
    segments = name.split("/")
    for depth in range(1, len(segments)):
        ancestor = "/".join(segments[:depth])
        if ancestor in group and isinstance(group[ancestor], Array):
            raise ValueError(
                f"{kind} {name!r} is below the existing array "
                f"{ancestor!r}. An array cannot contain other nodes"
            )
    if name not in group:
        return
    if not overwrite:
        raise ValueError(
            f"{kind} {name!r} already exists. Pass overwrite=True to "
            "replace it"
        )
    if not isinstance(group[name], Array):
        raise ValueError(
            f"{kind} {name!r} names an existing group. Overwrite replaces "
            "only an array"
        )


__all__ = ["ChecksumName", "ZarrFormat"]
