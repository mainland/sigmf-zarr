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
) -> tuple[int, ...] | None:
    """Choose a shard shape by grouping chunks along the growing axis.

    Args:
        sample_dtype: Sample array data type.
        array_shape: Complete sample array shape.
        chunk_shape: Logical chunk shape.
        batched: Whether the leading array axis is an item axis.
        target_shard_bytes: Target uncompressed shard size.

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
    growth_axis = 0 if batched else len(array_shape) - 1
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

    Returns:
        Resolved logical chunk shape and optional physical shard shape.

    Raises:
        ValueError: If explicit sharding is requested for Zarr format 2.
    """
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
            )

    resolved_shards = sample_shards
    if should_shard and resolved_shards is None:
        # Derive automatic shards from the resolved chunks so explicit chunk
        # choices remain authoritative.
        array_shape = (
            (num_samples, *sample_shape) if batched else sample_shape
        )
        resolved_shards = _default_sample_shards(
            sample_dtype,
            array_shape,
            resolved_chunks,
            batched=batched,
            target_shard_bytes=DEFAULT_IMPORT_SHARD_BYTES,
        )
    return resolved_chunks, resolved_shards


__all__ = [
    "DEFAULT_BATCHED_IMPORT_CHUNK_BYTES",
    "DEFAULT_IMPORT_CHUNK_BYTES",
    "DEFAULT_IMPORT_SHARD_BYTES",
    "resolve_import_sample_storage",
]
