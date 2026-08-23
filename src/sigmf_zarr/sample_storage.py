"""Resolve sample storage layouts used by dataset importers."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from zarr.core.array import ShardsLike

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat

DEFAULT_IMPORT_CHUNK_BYTES = 1024 * 1024
"""Target logical chunk size for sharded unbatched imports."""

DEFAULT_BATCHED_IMPORT_CHUNK_BYTES = 256 * 1024
"""Target logical chunk size for sharded batched imports."""

DEFAULT_IMPORT_SHARD_BYTES = 4 * 1024 * 1024
"""Target physical shard size for Zarr format 3 imports."""


def _default_sample_shards(
    sample_dtype: npt.DTypeLike,
    array_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    *,
    batched: bool,
    target_shard_bytes: int,
    time_axis: int = -1,
) -> tuple[int, ...] | None:
    """Choose a shard shape by grouping chunks along the growing axis.

    Args:
        sample_dtype: Sample array data type.
        array_shape: Complete sample array shape.
        chunk_shape: Logical chunk shape.
        batched: Whether the leading array axis is an item axis.
        target_shard_bytes: Target uncompressed shard size.
        time_axis: Position of the time axis in an unbatched array.

    Returns:
        A shard shape, or `None` when the array has only one chunk along the
        growing axis.
    """
    dtype = np.dtype(sample_dtype)
    chunk_elements = int(np.prod(chunk_shape, dtype=np.int64))
    chunk_bytes = dtype.itemsize * chunk_elements
    chunks_per_shard = max(1, target_shard_bytes // chunk_bytes)
    # Only the append-oriented axis grows. Keeping all other shard dimensions
    # equal to the chunk dimensions preserves complete I/Q and channel planes.
    growth_axis = 0 if batched else time_axis % len(array_shape)
    available_chunks = math.ceil(
        array_shape[growth_axis] / chunk_shape[growth_axis]
    )
    chunks_per_shard = min(chunks_per_shard, available_chunks)
    if chunks_per_shard <= 1:
        # A one-chunk shard adds indirection without reducing object count.
        return None

    # Zarr requires each shard dimension to be an integer multiple of its
    # corresponding chunk dimension.
    shard_shape = list(chunk_shape)
    shard_shape[growth_axis] *= chunks_per_shard
    return tuple(shard_shape)


def _validate_storage_shape(
    name: str, shape: object, array_ndim: int
) -> tuple[int, ...]:
    """Require an explicit chunk or shard shape of positive integers.

    Args:
        name: Option name for error messages.
        shape: Explicit shape.
        array_ndim: Number of dimensions of the sample array.

    Returns:
        Shape as a tuple.

    Raises:
        ValueError: If `shape` is not a sequence of `array_ndim` positive
            integers.
    """
    if not isinstance(shape, tuple | list) or not all(
        isinstance(extent, int | np.integer)
        and not isinstance(extent, bool)
        and extent > 0
        for extent in shape
    ):
        raise ValueError(
            f"{name} must contain positive integers, got {shape!r}"
        )
    if len(shape) != array_ndim:
        raise ValueError(
            f"{name} {tuple(shape)!r} must have {array_ndim} dimensions to "
            "match the sample array"
        )
    return tuple(int(extent) for extent in shape)


def resolve_import_sample_storage(
    sample_dtype: npt.DTypeLike,
    sample_shape: tuple[int, ...],
    num_samples: int,
    *,
    batched: bool,
    zarr_format: ZarrFormat,
    sample_chunks: tuple[int, ...] | None = None,
    sample_shards: ShardsLike | None = None,
    automatic_sharding: bool = True,
    time_axis: int = -1,
) -> tuple[tuple[int, ...], ShardsLike | None]:
    """Resolve chunk and shard shapes for an imported sample array.

    Zarr format 3 groups batched items into approximately 256 KiB logical
    chunks and 4 MiB physical shards. Unbatched format-3 recordings use
    approximately 1 MiB chunks inside 4 MiB shards. Zarr format 2 uses
    approximately 4 MiB chunks because it cannot use shards.

    Args:
        sample_dtype: Sample array data type.
        sample_shape: Per-item shape for batched data or complete array shape
            for unbatched data.
        num_samples: Number of items for batched data. Use one for unbatched
            data.
        batched: Whether the leading array axis is an item axis.
        zarr_format: Physical Zarr format.
        sample_chunks: Optional explicit logical chunk shape.
        sample_shards: Optional explicit physical shard shape.
        automatic_sharding: Whether to derive shards when none are supplied.
        time_axis: Position of the `time` axis in `sample_shape` for
            unbatched data. Chunks and shards split only this axis. Defaults
            to the final axis.

    Returns:
        Resolved logical chunk shape and optional physical shard shape.

    Raises:
        ValueError: If explicit sharding is requested for Zarr format 2,
            `time_axis` is outside an unbatched `sample_shape`, or an
            explicit chunk or shard shape does not contain one positive
            integer per sample-array dimension.
    """
    if not batched and not -len(sample_shape) <= time_axis < len(sample_shape):
        raise ValueError(
            f"time_axis {time_axis} is outside sample shape {sample_shape}"
        )
    array_ndim = len(sample_shape) + (1 if batched else 0)
    if sample_chunks is not None:
        sample_chunks = _validate_storage_shape(
            "sample_chunks", sample_chunks, array_ndim
        )
    if isinstance(sample_shards, tuple | list):
        _validate_storage_shape("sample_shards", sample_shards, array_ndim)
    if zarr_format == 2 and sample_shards is not None:
        raise ValueError(
            "Sample sharding requires Zarr format 3. Omit sample_shards or "
            "create a Zarr format 3 store"
        )

    should_shard = zarr_format == 3 and (
        automatic_sharding or sample_shards is not None
    )
    resolved_chunks = sample_chunks
    if resolved_chunks is None:
        if batched and sample_shards is not None:
            # One complete item per chunk divides every valid explicit shard
            # batch while avoiding chunks that split a signal tensor.
            resolved_chunks = (1, *sample_shape)
        else:
            # Without shards, use the shard-size target for chunks to avoid
            # creating many small store objects.
            target_chunk_bytes = DEFAULT_IMPORT_SHARD_BYTES
            if should_shard:
                target_chunk_bytes = (
                    DEFAULT_BATCHED_IMPORT_CHUNK_BYTES
                    if batched
                    else DEFAULT_IMPORT_CHUNK_BYTES
                )
            resolved_chunks = SigMFZarrStore.default_sample_chunks(
                sample_dtype,
                sample_shape,
                num_samples,
                batched=batched,
                target_chunk_bytes=target_chunk_bytes,
                time_axis=time_axis,
            )

    resolved_shards = sample_shards
    if should_shard and resolved_shards is None:
        # Derive automatic shards from the resolved chunks so explicit chunk
        # choices remain authoritative.
        array_shape = (num_samples, *sample_shape) if batched else sample_shape
        resolved_shards = _default_sample_shards(
            sample_dtype,
            array_shape,
            resolved_chunks,
            batched=batched,
            target_shard_bytes=DEFAULT_IMPORT_SHARD_BYTES,
            time_axis=time_axis,
        )
    return resolved_chunks, resolved_shards


__all__ = [
    "DEFAULT_BATCHED_IMPORT_CHUNK_BYTES",
    "DEFAULT_IMPORT_CHUNK_BYTES",
    "DEFAULT_IMPORT_SHARD_BYTES",
    "resolve_import_sample_storage",
]
