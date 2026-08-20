"""JSON type aliases and validation helpers for SigMF-Zarr metadata."""

from __future__ import annotations

import math
from typing import cast

type JSONScalar = None | bool | int | float | str
"""Scalar value representable in JSON metadata."""

type JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
"""Recursive JSON-compatible metadata value."""

type JSONObject = dict[str, JSONValue]
"""JSON object used for SigMF-style metadata dictionaries."""


_MAX_DEPTH = 128
"""Greatest accepted container nesting depth.

This is an implementation limit rather than a format rule. It keeps nested
values well within the interpreter recursion limit, so every recursive reader
in this package can process any value that the writer accepts.
"""


def _value_name(base: str, parts: list[str | int]) -> str:
    """Format the human-readable name of a nested value.

    Names are built only for error messages. Building one for every value
    would repeat each ancestor key, so a long key path above a long list
    would make validation quadratic.

    Args:
        base: Name of the outermost value.
        parts: Object keys and list positions from the outermost value.

    Returns:
        Name such as `attributes.global[3]`.
    """
    return base + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in parts
    )


def _has_utf8_encoding(text: str) -> bool:
    """Return whether a string has a UTF-8 encoding.

    Args:
        text: Candidate string or object key.

    Returns:
        False if `text` contains a lone surrogate code point.
    """
    if text.isascii():
        return True
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _json_value(
    value: object, *, base: str, parts: list[str | int]
) -> JSONValue:
    """Validate one value nested `len(parts)` containers deep.

    Args:
        value: Candidate JSON value.
        base: Name of the outermost value, for error messages.
        parts: Object keys and list positions from the outermost value to
            `value`. The list is extended and restored during traversal.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is not finite, encodable JSON within the
            nesting limit.
    """
    if value is None or isinstance(value, bool | int):
        return cast(JSONScalar, value)
    if isinstance(value, str):
        if not _has_utf8_encoding(value):
            raise ValueError(
                f"Expected {_value_name(base, parts)} strings to contain no "
                "lone surrogates"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Expected {_value_name(base, parts)} numbers to be finite"
            )
        return value
    if isinstance(value, list | dict) and len(parts) >= _MAX_DEPTH:
        raise ValueError(
            f"Expected {_value_name(base, parts)} to be nested at most "
            f"{_MAX_DEPTH} levels deep"
        )
    if isinstance(value, list):
        return _json_list(value, base=base, parts=parts)
    if isinstance(value, dict):
        return _json_object(value, base=base, parts=parts)
    raise ValueError(
        f"Expected {_value_name(base, parts)} to contain only JSON values"
    )


def _json_list(
    value: list[object], *, base: str, parts: list[str | int]
) -> list[JSONValue]:
    """Validate the items of one list.

    Args:
        value: Candidate list.
        base: Name of the outermost value, for error messages.
        parts: Path from the outermost value to `value`.

    Returns:
        The validated items.

    Raises:
        ValueError: If an item is not finite, encodable JSON within the
            nesting limit.
    """
    items: list[JSONValue] = []
    for index, item in enumerate(value):
        parts.append(index)
        items.append(_json_value(item, base=base, parts=parts))
        parts.pop()
    return items


def _json_object(
    value: dict[object, object], *, base: str, parts: list[str | int]
) -> dict[str, JSONValue]:
    """Validate the keys and values of one object.

    Args:
        value: Candidate mapping.
        base: Name of the outermost value, for error messages.
        parts: Path from the outermost value to `value`.

    Returns:
        The validated object.

    Raises:
        ValueError: If a key is not an encodable string, or a value is not
            finite, encodable JSON within the nesting limit.
    """
    result: dict[str, JSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Expected {_value_name(base, parts)} object keys to be "
                "strings"
            )
        if not _has_utf8_encoding(key):
            raise ValueError(
                f"Expected {_value_name(base, parts)} object key strings to "
                "contain no lone surrogates"
            )
        parts.append(key)
        result[key] = _json_value(item, base=base, parts=parts)
        parts.pop()
    return result


def json_value(value: object, *, name: str) -> JSONValue:
    """Validate a recursively JSON-compatible value.

    Args:
        value: Candidate JSON value.
        name: Human-readable value name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If a mapping key is not a string, a number is not finite,
            a string or key contains a lone surrogate, lists and objects are
            nested more than 128 levels deep, or a value has no JSON
            representation.
    """
    try:
        return _json_value(value, base=name, parts=[])
    except RecursionError as exc:
        # The explicit limit normally applies first. A caller that is already
        # deep in the stack can still exhaust it on an accepted depth.
        raise ValueError(f"Expected {name} to be nested less deeply") from exc


def json_object(value: object, *, name: str) -> JSONObject:
    """Validate that a decoded metadata value is a JSON object.

    Args:
        value: Decoded JSON value to validate.
        name: Human-readable value name for error messages.

    Returns:
        The decoded value typed as a JSON object.

    Raises:
        ValueError: If `value` is not a JSON object or contains a value
            that `json_value` rejects.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Expected {name} to be a JSON object")
    return cast(JSONObject, json_value(value, name=name))


def json_object_list(value: object, *, name: str) -> list[JSONObject]:
    """Validate that a decoded metadata value is a JSON object list.

    Args:
        value: Decoded JSON value to validate.
        name: Human-readable value name for error messages.

    Returns:
        The decoded value typed as a list of JSON objects.

    Raises:
        ValueError: If `value` is not a list of JSON objects or contains
            a value that `json_value` rejects.
    """
    if not isinstance(value, list):
        raise ValueError(f"Expected {name} to be a list of JSON objects")
    result: list[JSONObject] = []
    for index, item in enumerate(value):
        result.append(json_object(item, name=f"{name}[{index}]"))
    return result


__all__ = [
    "JSONObject",
    "JSONScalar",
    "JSONValue",
    "json_object",
    "json_object_list",
    "json_value",
]
