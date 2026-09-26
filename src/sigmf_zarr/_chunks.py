"""Sequential reads that decode each stored chunk once.

Zarr decodes a whole chunk to return any part of it. A loop that reads a
long array in fixed batches shorter than its chunks therefore decodes each
chunk once per batch, and the total cost grows with the square of the chunk
length. The helpers here align reads to chunk boundaries instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class _ChunkedVector(Protocol):
    """One-dimensional array that supports contiguous slicing."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        ...

    @property
    def chunks(self) -> tuple[int, ...]:
        """Stored chunk shape."""
        ...

    def __getitem__(self, selection: slice) -> Any:
        """Read a contiguous span."""
        ...


def chunk_spans(
    start: int, stop: int, chunk_length: int, minimum: int
) -> Iterator[tuple[int, int]]:
    """Split an interval into spans whose interior bounds are chunk-aligned.

    Every stored chunk that intersects the interval lies within exactly one
    span, so reading the spans in turn decodes each chunk once. Each span is
    at most the larger of one chunk and `minimum` rounded up to a whole
    number of chunks.

    Args:
        start: Inclusive start of the interval.
        stop: Exclusive end of the interval.
        chunk_length: Stored chunk length along the interval's axis.
        minimum: Preferred span length when chunks are shorter than it.

    Yields:
        Consecutive half-open spans that cover the interval.
    """
    chunk = max(chunk_length, 1)
    step = -(-max(minimum, 1) // chunk) * chunk
    position = start
    while position < stop:
        # Round down so that every later span starts on a chunk boundary.
        end = min(position // chunk * chunk + step, stop)
        yield position, end
        position = end


class SpanReader:
    """Read consecutive spans of a one-dimensional array chunk by chunk.

    A read extends to the end of the last chunk it touches, and later reads
    within that range use the retained values. For reads in ascending order,
    each stored chunk decodes once. Retained memory is at most one requested
    span plus one chunk.
    """

    _array: _ChunkedVector
    """Source array."""

    _length: int
    """Source length."""

    _chunk: int
    """Stored chunk length."""

    _start: int
    """Inclusive start of the retained values."""

    _stop: int
    """Exclusive end of the retained values."""

    _values: npt.NDArray[Any]
    """Retained decoded values."""

    def __init__(self, array: _ChunkedVector) -> None:
        """Wrap an array without reading it.

        Args:
            array: One-dimensional source array.
        """
        self._array = array
        self._length = int(array.shape[0])
        self._chunk = max(int(array.chunks[0]), 1)
        self._start = self._stop = 0
        self._values = np.empty(0)

    def read(self, start: int, stop: int) -> npt.NDArray[Any]:
        """Return values in one span.

        Args:
            start: Inclusive start, at least zero.
            stop: Exclusive end, at most the array length.

        Returns:
            Decoded values. The result may share memory with retained values,
            so callers must not modify it.
        """
        if self._start <= start and stop <= self._stop:
            return self._values[start - self._start : stop - self._start]
        end = min(-(-stop // self._chunk) * self._chunk, self._length)
        if self._start <= start < self._stop:
            # Keep the retained tail rather than decoding its chunk again.
            values = np.concatenate(
                (
                    self._values[start - self._start :],
                    np.asarray(self._array[self._stop : end]),
                )
            )
        else:
            values = np.asarray(self._array[start:end])
        self._start, self._stop, self._values = start, end, values
        return values[: stop - start]


__all__ = ["SpanReader", "chunk_spans"]
