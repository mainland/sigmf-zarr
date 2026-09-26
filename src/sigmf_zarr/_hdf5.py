"""Bounded reads from untrusted HDF5 datasets.

HDF5's deflate filter enlarges its output buffer until the compressed stream
ends, so one small chunk can inflate to gigabytes. Readers in this module
accept only the gzip (deflate), shuffle, and Fletcher-32 filters. A dataset
that uses gzip is read one raw chunk at a time, and this module undoes its
filters itself, inflating at most one byte past the size that the chunk must
decode to. Other datasets are read through HDF5, because shuffle and
Fletcher-32 never enlarge a chunk.
"""

from __future__ import annotations

import itertools
import math
import threading
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from h5py import h5pl, h5t, h5z

_FILTER_NAMES = {
    h5z.FILTER_DEFLATE: "gzip",
    h5z.FILTER_SHUFFLE: "shuffle",
    h5z.FILTER_FLETCHER32: "fletcher32",
}
"""HDF5 filters that the bounded reader can undo, by filter identifier."""

_FLETCHER32_BYTES = 4
"""Size of the checksum that the Fletcher-32 filter appends to a chunk."""

_FLETCHER32_BLOCK_WORDS = 1 << 20
"""Number of 16-bit words checksummed per NumPy operation."""

_plugin_lock = threading.Lock()
_plugin_users = 0
_plugin_paths: list[bytes] = []


@contextmanager
def hdf5_plugins_disabled() -> Iterator[None]:
    """Prevent HDF5 from loading filter plugins while the context is active.

    HDF5 loads a dynamic plugin for an unregistered filter from the
    directories in its plugin search path. h5py does not expose
    `H5PLset_loading_state`, so the context empties the search path and
    restores it when the last concurrent user exits. Filters that are
    already registered, including those built into HDF5, remain available.

    Yields:
        Nothing.
    """
    global _plugin_users, _plugin_paths
    with _plugin_lock:
        if _plugin_users == 0:
            _plugin_paths = [h5pl.get(index) for index in range(h5pl.size())]
            for _ in _plugin_paths:
                h5pl.remove(0)
        _plugin_users += 1
    try:
        yield
    finally:
        with _plugin_lock:
            _plugin_users -= 1
            if _plugin_users == 0:
                for path in _plugin_paths:
                    h5pl.append(path)
                _plugin_paths = []


@dataclass(frozen=True)
class HDF5Filter:
    """One stage of an HDF5 dataset's filter pipeline."""

    code: int
    """HDF5 filter identifier."""

    values: tuple[int, ...]
    """Filter client data."""


def dataset_filters(
    dataset: h5py.Dataset,
    *,
    description: str,
) -> tuple[HDF5Filter, ...]:
    """Return a dataset's filter pipeline after checking that it is bounded.

    Args:
        dataset: HDF5 dataset to inspect.
        description: Dataset description for error messages.

    Returns:
        Filters in the order that HDF5 applies them when writing.

    Raises:
        ValueError: If the pipeline uses a filter other than gzip, shuffle,
            and Fletcher-32, uses one filter twice, has invalid shuffle
            parameters, or belongs to a dataset that is not chunked.
    """
    plist = dataset.id.get_create_plist()
    filters: list[HDF5Filter] = []
    for position in range(plist.get_nfilters()):
        code, _flags, values, name = plist.get_filter(position)
        if code not in _FILTER_NAMES:
            label = name.decode("utf-8", "replace") or "unnamed"
            raise ValueError(
                f"{description} uses HDF5 filter {code} ({label!r}). "
                "Import reads only gzip, shuffle, and fletcher32 filters"
            )
        if any(code == other.code for other in filters):
            raise ValueError(
                f"{description} applies the HDF5 {_FILTER_NAMES[code]} "
                "filter more than once"
            )
        if code == h5z.FILTER_SHUFFLE and (len(values) != 1 or values[0] < 1):
            raise ValueError(
                f"{description} has invalid HDF5 shuffle parameters"
            )
        filters.append(HDF5Filter(code=int(code), values=tuple(values)))
    if filters and dataset.chunks is None:
        raise ValueError(
            f"{description} declares HDF5 filters without chunked storage"
        )
    return tuple(filters)


def _fold_checksum(total: int, *, positive: bool) -> int:
    """Reduce a Fletcher-32 sum to HDF5's 16-bit representation.

    HDF5 adds carries back into the low 16 bits, which preserves the sum
    modulo 65535 and maps a positive multiple of 65535 to `0xffff`.

    Args:
        total: Sum, exact or reduced modulo 65535.
        positive: Whether the exact sum is positive.

    Returns:
        Sum in the range HDF5 produces.
    """
    return (total - 1) % 0xFFFF + 1 if positive else 0


def fletcher32(data: bytes) -> int:
    """Return HDF5's Fletcher-32 checksum of a byte string.

    HDF5 sums big-endian 16-bit words, padding an odd final byte with a zero
    low byte. The second sum adds every prefix of the first, so word `i` of
    `n` contributes `n - i` times.

    Args:
        data: Checksummed bytes.

    Returns:
        Checksum as HDF5 computes it.
    """
    words = np.frombuffer(data, dtype=">u2", count=len(data) // 2)
    count = len(words) + len(data) % 2
    first = 0
    second = 0
    for begin in range(0, len(words), _FLETCHER32_BLOCK_WORDS):
        block = words[begin : begin + _FLETCHER32_BLOCK_WORDS].astype(
            np.uint64
        )
        # Reduce the weights so that one block's dot product fits in 64 bits.
        weights = np.arange(
            count - begin, count - begin - len(block), -1, dtype=np.uint64
        )
        first += int(block.sum())
        second += int(np.dot(block, weights % 0xFFFF))
    if len(data) % 2:
        first += data[-1] << 8
        second += data[-1] << 8
    positive = first > 0
    return _fold_checksum(second, positive=positive) << 16 | _fold_checksum(
        first, positive=positive
    )


def _strip_fletcher32(data: bytes, *, description: str) -> bytes:
    """Verify and remove a chunk's Fletcher-32 checksum.

    Args:
        data: Chunk bytes followed by a little-endian checksum.
        description: Dataset description for error messages.

    Returns:
        Chunk bytes without the checksum.

    Raises:
        ValueError: If the chunk is too short or the checksum differs.
    """
    if len(data) < _FLETCHER32_BYTES:
        raise ValueError(f"{description} has a truncated HDF5 chunk")
    body = data[:-_FLETCHER32_BYTES]
    stored = int.from_bytes(data[-_FLETCHER32_BYTES:], "little")
    computed = fletcher32(body)
    # HDF5 before 1.6.3 byte-swapped each half on little-endian hosts, and
    # HDF5 still accepts that form.
    swapped = ((computed & 0x00FF00FF) << 8) | ((computed >> 8) & 0x00FF00FF)
    if stored not in {computed, swapped}:
        raise ValueError(
            f"{description} has an HDF5 chunk that fails its Fletcher-32 "
            "checksum"
        )
    return body


def _unshuffle(data: bytes, element_size: int) -> bytes:
    """Undo HDF5's shuffle filter.

    The filter stores byte `j` of every element together, and leaves any
    trailing partial element in place.

    Args:
        data: Shuffled chunk bytes.
        element_size: Element size in bytes.

    Returns:
        Chunk bytes in element order.
    """
    count = len(data) // element_size
    if element_size <= 1 or count <= 1:
        return data
    planes = np.frombuffer(data, dtype=np.uint8, count=count * element_size)
    return (
        planes.reshape(element_size, count).T.tobytes()
        + data[count * element_size :]
    )


def _inflate(data: bytes, size: int, *, description: str) -> bytes:
    """Inflate one zlib stream that must decode to exactly `size` bytes.

    Args:
        data: zlib stream, possibly followed by unused bytes as HDF5 allows.
        size: Required decoded size.
        description: Dataset description for error messages.

    Returns:
        Decoded bytes.

    Raises:
        ValueError: If the stream is corrupt or truncated, or decodes to any
            other size. Inflation stops one byte past `size`.
    """
    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(data, size + 1)
    except zlib.error as exc:
        raise ValueError(
            f"{description} has a corrupt gzip HDF5 chunk: {exc}"
        ) from exc
    if len(decoded) > size:
        raise ValueError(
            f"{description} has a gzip HDF5 chunk that inflates past its "
            f"{size}-byte decoded size"
        )
    if not decompressor.eof:
        raise ValueError(f"{description} has a truncated gzip HDF5 chunk")
    if len(decoded) != size:
        raise ValueError(
            f"{description} has a gzip HDF5 chunk that decodes to "
            f"{len(decoded)} bytes instead of {size}"
        )
    return decoded


def _max_stored_bytes(chunk_bytes: int) -> int:
    """Return the largest stored size accepted for one filtered chunk.

    Args:
        chunk_bytes: Decoded chunk size.

    Returns:
        Bound above zlib's worst-case expansion of incompressible data, plus
        the zlib wrapper and a Fletcher-32 checksum.
    """
    return chunk_bytes + chunk_bytes // 1024 + 64


class HDF5RowReader:
    """Read rows of an HDF5 dataset without unbounded decompression.

    A row is one index along the first axis. Each read returns full rows, in
    the dataset's own NumPy dtype, as HDF5 slicing does.
    """

    def __init__(self, dataset: h5py.Dataset, *, description: str) -> None:
        """Check a dataset's filters and prepare to read it.

        Args:
            dataset: HDF5 dataset to read.
            description: Dataset description for error messages.

        Raises:
            ValueError: If the dataset's filters are not allowed, or a
                gzip dataset stores a type that NumPy cannot represent
                byte for byte.
        """
        self._dataset = dataset
        self._description = description
        self._filters = dataset_filters(dataset, description=description)
        self._direct = any(
            item.code == h5z.FILTER_DEFLATE for item in self._filters
        )
        self._cached_offset: tuple[int, ...] | None = None
        self._cached_chunk: npt.NDArray[Any] | None = None
        if self._direct and not dataset.id.get_type().equal(
            h5t.py_create(dataset.dtype)
        ):
            raise ValueError(
                f"{description} stores an HDF5 type that differs from NumPy "
                f"{dataset.dtype}, which gzip import cannot decode"
            )

    def read(self, start: int, stop: int) -> npt.NDArray[Any]:
        """Read rows `start` through `stop - 1`.

        Args:
            start: First row.
            stop: One past the last row.

        Returns:
            Rows with the dataset's dtype.

        Raises:
            ValueError: If a gzip chunk is corrupt, fails its checksum, or
                does not decode to exactly the chunk size.
        """
        if not self._direct:
            return np.asarray(self._dataset[start:stop])
        dataset = self._dataset
        chunks = dataset.chunks
        assert chunks is not None  # Guaranteed by dataset_filters.
        shape = dataset.shape
        result = np.empty((stop - start, *shape[1:]), dtype=dataset.dtype)
        first = start - start % chunks[0]
        grid = [range(first, stop, chunks[0])] + [
            range(0, length, size)
            for length, size in zip(shape[1:], chunks[1:], strict=True)
        ]
        for offset in itertools.product(*grid):
            chunk = self._chunk(offset)
            low = max(start, offset[0])
            high = min(stop, offset[0] + chunks[0])
            source = [slice(low - offset[0], high - offset[0])]
            target = [slice(low - start, high - start)]
            for begin, size, length in zip(
                offset[1:], chunks[1:], shape[1:], strict=True
            ):
                end = min(begin + size, length)
                source.append(slice(0, end - begin))
                target.append(slice(begin, end))
            result[tuple(target)] = chunk[tuple(source)]
        return result

    def _chunk(self, offset: tuple[int, ...]) -> npt.NDArray[Any]:
        """Decode the chunk at one chunk-aligned offset.

        The reader keeps the most recent chunk, because consecutive reads
        whose boundaries do not align with chunks share one.

        Args:
            offset: Dataset coordinates of the chunk's first element.

        Returns:
            Decoded chunk with the full chunk shape.
        """
        if offset == self._cached_offset and self._cached_chunk is not None:
            return self._cached_chunk
        dataset = self._dataset
        chunks = dataset.chunks
        assert chunks is not None  # Guaranteed by dataset_filters.
        info = dataset.id.get_chunk_info_by_coord(offset)
        if info.byte_offset is None:
            # HDF5 reads an unwritten chunk as the fill value.
            chunk = np.full(chunks, dataset.fillvalue, dtype=dataset.dtype)
        else:
            chunk_bytes = math.prod(chunks) * dataset.dtype.itemsize
            if info.size > _max_stored_bytes(chunk_bytes):
                raise ValueError(
                    f"{self._description} stores a {info.size}-byte HDF5 "
                    f"chunk, more than a {chunk_bytes}-byte chunk can need"
                )
            filter_mask, raw = dataset.id.read_direct_chunk(offset)
            data = self._decode(bytes(raw), filter_mask, chunk_bytes)
            chunk = np.frombuffer(data, dtype=dataset.dtype).reshape(chunks)
        self._cached_offset = offset
        self._cached_chunk = chunk
        return chunk

    def _decode(
        self, data: bytes, filter_mask: int, chunk_bytes: int
    ) -> bytes:
        """Undo a chunk's filters in reverse pipeline order.

        Args:
            data: Stored chunk bytes.
            filter_mask: HDF5 mask whose bit `i` marks filter `i` as skipped.
            chunk_bytes: Required decoded chunk size.

        Returns:
            Decoded chunk bytes.

        Raises:
            ValueError: If any stage fails or the result has the wrong size.
        """
        active = [
            item
            for position, item in enumerate(self._filters)
            if not filter_mask & (1 << position)
        ]
        for position in reversed(range(len(active))):
            item = active[position]
            if item.code == h5z.FILTER_FLETCHER32:
                data = _strip_fletcher32(data, description=self._description)
            elif item.code == h5z.FILTER_SHUFFLE:
                data = _unshuffle(data, item.values[0])
            else:
                # Checksums applied before compression are undone after it,
                # so the inflated data still carries them.
                checksums = sum(
                    1
                    for earlier in active[:position]
                    if earlier.code == h5z.FILTER_FLETCHER32
                )
                data = _inflate(
                    data,
                    chunk_bytes + checksums * _FLETCHER32_BYTES,
                    description=self._description,
                )
        if len(data) != chunk_bytes:
            raise ValueError(
                f"{self._description} has an HDF5 chunk that decodes to "
                f"{len(data)} bytes instead of {chunk_bytes}"
            )
        return data


__all__ = [
    "HDF5Filter",
    "HDF5RowReader",
    "dataset_filters",
    "fletcher32",
    "hdf5_plugins_disabled",
]
