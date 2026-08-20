"""Kinds and logical byte order of stored sample dtypes."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def logical_sample_dtype(dtype: npt.DTypeLike) -> np.dtype[Any]:
    """Return a sample dtype in the SigMF-Zarr logical byte order.

    SigMF-Zarr records `sigmf-zarr:dtype` and creates sample arrays in
    little-endian byte order. A source in another byte order therefore
    produces the same metadata in both Zarr formats and on every host.
    Dtypes without a byte order, such as one-byte types, are unchanged.

    Args:
        dtype: Requested or source sample dtype.

    Returns:
        Little-endian or byte-order-free form of `dtype`.

    Raises:
        TypeError: If `dtype` does not describe a NumPy dtype.
    """
    return np.dtype(dtype).newbyteorder("<")


def is_logical_sample_dtype(dtype: npt.DTypeLike) -> bool:
    """Return whether a sample dtype uses the logical byte order.

    Args:
        dtype: Declared or stored sample dtype.

    Returns:
        Whether `dtype` is little-endian or has no byte order.

    Raises:
        TypeError: If `dtype` does not describe a NumPy dtype.
    """
    resolved = np.dtype(dtype)
    return resolved == resolved.newbyteorder("<")


SAMPLE_DTYPE_KINDS = "biufc"
"""NumPy kinds of sample dtypes: Boolean, integer, floating point, complex."""


def is_numeric_sample_dtype(dtype: npt.DTypeLike) -> bool:
    """Return whether a dtype is one that sample arrays may use.

    Text, bytes, date-time, time-delta, object, void, and structured dtypes
    have no numeric sample values, so no sample write could fill them.

    Args:
        dtype: Declared or requested sample dtype.

    Returns:
        Whether `dtype` is a Boolean, integer, floating-point, or complex
        dtype.

    Raises:
        TypeError: If `dtype` does not describe a NumPy dtype.
    """
    return np.dtype(dtype).kind in SAMPLE_DTYPE_KINDS
