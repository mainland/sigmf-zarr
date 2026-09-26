"""Read Chad Spooner's ``.tim`` signal files."""

from __future__ import annotations

import struct
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

type TimSamples = npt.NDArray[np.float32] | npt.NDArray[np.complex64]
"""Real or complex single-precision samples decoded from ``.tim``."""

_INVALID_TIM = (
    "Invalid .tim header or payload length. Expected two int32 header "
    "values followed by the declared float32 samples"
)


def _header_fields(header: bytes) -> tuple[str, int, int]:
    """Detect byte order and decode the fixed ``.tim`` header.

    Args:
        header: At least the first eight bytes of a ``.tim`` file.

    Returns:
        Byte-order prefix, real/complex flag, and sample count.

    Raises:
        ValueError: If the header is too short or its fields are invalid.
    """
    if len(header) < 8:
        raise ValueError(_INVALID_TIM)

    # The real/complex flag has only two valid values, so it is sufficient to
    # identify the byte order without a caller-supplied format option.
    little_flag = struct.unpack("<i", header[:4])[0]
    big_flag = struct.unpack(">i", header[:4])[0]
    if little_flag in {1, 2}:
        prefix = "<"
    elif big_flag in {1, 2}:
        prefix = ">"
    else:
        raise ValueError(_INVALID_TIM)

    real_complex, sample_count = cast(
        tuple[int, int],
        struct.unpack(f"{prefix}ii", header[:8]),
    )
    if sample_count < 0:
        raise ValueError(_INVALID_TIM)
    return prefix, real_complex, sample_count


def _tim_file_size(header: bytes) -> int:
    """Return the exact file size that a ``.tim`` header declares.

    Readers can compare this size with an archive member's declared size
    before reading the complete payload.

    Args:
        header: At least the first eight bytes of a ``.tim`` file.

    Returns:
        Header size plus the declared float32 payload size, in bytes.

    Raises:
        ValueError: If the header is invalid.
    """
    _, real_complex, sample_count = _header_fields(header)
    return 8 + sample_count * real_complex * 4


def _decode_header(data: bytes) -> tuple[str, int, int]:
    """Detect byte order and decode a ``.tim`` header.

    Args:
        data: Complete file contents.

    Returns:
        Byte-order prefix, real/complex flag, and sample count.

    Raises:
        ValueError: If the header is invalid or the payload size is wrong.
    """
    prefix, real_complex, sample_count = _header_fields(data)
    # Requiring the exact declared size also rejects trailing data and guards
    # against a plausible flag decoded with the wrong byte order.
    if len(data) != _tim_file_size(data):
        raise ValueError(_INVALID_TIM)
    return prefix, real_complex, sample_count


def decode_tim(
    data: bytes,
) -> TimSamples:
    """Decode complete ``.tim`` file contents.

    Spooner's format begins with two 32-bit integers. The first is ``1``
    for real samples or ``2`` for complex samples, and the second is the
    logical sample count. The payload contains float32 values. Complex
    payloads alternate I and Q components.

    Args:
        data: Complete file contents.

    Returns:
        One-dimensional float32 or complex64 sample array.

    Raises:
        ValueError: If the header or payload length is invalid.
    """
    prefix, real_complex, sample_count = _decode_header(data)
    components = np.frombuffer(
        data,
        dtype=np.dtype(f"{prefix}f4"),
        count=sample_count * real_complex,
        offset=8,
    ).astype(np.float32, copy=True)
    # The copy above normalizes native byte order and detaches from immutable
    # input bytes before the components are assembled.

    if real_complex == 1:
        return components

    # Direct component assignment avoids an intermediate complex promotion.
    samples = np.empty((sample_count,), dtype=np.complex64)
    samples.real = components[0::2]
    samples.imag = components[1::2]
    return samples


def read_tim(
    path: str | PathLike[str],
) -> TimSamples:
    """Read a real or complex signal from a ``.tim`` file.

    Args:
        path: Source ``.tim`` file.

    Returns:
        One-dimensional float32 or complex64 sample array.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file encoding is invalid.
    """
    return decode_tim(Path(path).read_bytes())


__all__ = [
    "TimSamples",
    "decode_tim",
    "read_tim",
]
