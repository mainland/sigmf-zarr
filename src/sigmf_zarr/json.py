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


def json_value(value: object, *, name: str) -> JSONValue:
    """Validate a recursively JSON-compatible value.

    Args:
        value: Candidate JSON value.
        name: Human-readable value name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If a mapping key is not a string, a number is not finite,
            or a value has no JSON representation.
    """
    if value is None or isinstance(value, bool | int | str):
        return cast(JSONScalar, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Expected {name} numbers to be finite")
        return value
    if isinstance(value, list):
        return [
            json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"Expected {name} object keys to be strings")
        return {
            cast(str, key): json_value(item, name=f"{name}.{key}")
            for key, item in value.items()
        }
    raise ValueError(f"Expected {name} to contain only JSON values")


def json_object(value: object, *, name: str) -> JSONObject:
    """Validate that a decoded metadata value is a JSON object.

    Args:
        value: Decoded JSON value to validate.
        name: Human-readable value name for error messages.

    Returns:
        The decoded value typed as a JSON object.

    Raises:
        ValueError: If `value` is not a JSON object.
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
        ValueError: If `value` is not a list of JSON objects.
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
