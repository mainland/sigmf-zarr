"""Explicit categorical validation for independent metadata indexes."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as integer_index
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from sigmf_zarr.json import JSONValue, json_value
from sigmf_zarr.readonly import ReadOnlyArray


def validate_categorical_values(
    values: npt.ArrayLike, *, labels: object
) -> list[JSONValue]:
    """Validate categorical IDs and return a detached JSON lookup table.

    This checks the supplied values only. It neither reads an entire index nor
    compares its values with recording metadata. Domain consumers can apply
    additional constraints such as unique string labels.

    Args:
        values: One-dimensional array of non-Boolean integer category IDs.
        labels: JSON list of lookup values.

    Returns:
        Validated lookup table, detached from the input's mutable objects.

    Raises:
        ValueError: If the lookup table is malformed, values are not a vector
            of integer IDs, or an ID is outside the lookup table.
    """
    if not isinstance(labels, list):
        raise ValueError("Categorical index labels must be a JSON list")
    table = cast(list[JSONValue], json_value(labels, name="index labels"))
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "Categorical values must be one-dimensional integer IDs"
        )
    if np.any(ids < 0) or np.any(ids >= len(table)):
        raise ValueError("Categorical ID is outside the labels lookup table")
    return table


def _read_index_selection(
    array: ReadOnlyArray, selection: int | slice | Sequence[int]
) -> npt.NDArray[Any]:
    """Read a bounded one-dimensional selection in the requested order.

    Args:
        array: Structurally validated index array.
        selection: Integer position, slice, or sequence of integer positions.

    Returns:
        One-dimensional selected values, including for scalar selections.

    Raises:
        TypeError: If positions are not integers or are Boolean values.
        IndexError: If a position is outside the index.
        ValueError: If a slice step is zero.
    """
    if isinstance(selection, slice):
        start, stop, step = selection.indices(array.shape[0])
        if step > 0:
            return np.asarray(array[slice(start, stop, step)])
        # Zarr basic slicing does not support negative steps. Explicit
        # positions retain Python's reverse-slice order through oindex.
        positions = list(range(start, stop, step))
    elif isinstance(selection, int | np.integer):
        positions = [selection]
    else:
        positions = list(selection)
    normalized = []
    length = array.shape[0]
    for position in positions:
        if isinstance(position, bool | np.bool_):
            raise TypeError("Index selection positions must not be Boolean")
        value = integer_index(position)
        if value < -length or value >= length:
            raise IndexError(f"Index position {value} is out of range")
        # Bounds must be checked before modulo, or invalid positions would
        # wrap into valid items instead of raising IndexError.
        normalized.append(value % length)
    return np.asarray(array.oindex[np.asarray(normalized, dtype=np.intp)])


__all__ = ["validate_categorical_values"]
