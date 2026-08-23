"""Tests for imported sample storage layout selection."""

from __future__ import annotations

from sigmf_zarr.sample_storage import resolve_import_sample_storage


def test_radioml2016_uses_four_mibibyte_shards() -> None:
    """RadioML items should use 256 KiB chunks within 4 MiB shards."""
    chunks, shards = resolve_import_sample_storage(
        "float32",
        (2, 128),
        220_000,
        batched=True,
        zarr_format=3,
    )

    assert chunks == (256, 2, 128)
    assert shards == (4096, 2, 128)


def test_larger_radioml_items_use_smaller_shard_batches() -> None:
    """Chunk and shard batches should adapt to per-item byte size."""
    chunks, shards = resolve_import_sample_storage(
        "float32",
        (2, 1024),
        2_555_904,
        batched=True,
        zarr_format=3,
    )

    assert chunks == (32, 2, 1024)
    assert shards == (512, 2, 1024)


def test_zarr_format_2_uses_larger_chunks_without_shards() -> None:
    """Format-2 imports should avoid one-file-per-item layouts."""
    chunks, shards = resolve_import_sample_storage(
        "float32",
        (2, 128),
        220_000,
        batched=True,
        zarr_format=2,
    )

    assert chunks == (4096, 2, 128)
    assert shards is None


def test_disabling_sharding_uses_larger_format_3_chunks() -> None:
    """Unsharded format-3 imports should still avoid tiny files."""
    chunks, shards = resolve_import_sample_storage(
        "float32",
        (2, 128),
        220_000,
        batched=True,
        zarr_format=3,
        automatic_sharding=False,
    )

    assert chunks == (4096, 2, 128)
    assert shards is None


def test_unbatched_imports_shard_along_the_final_axis() -> None:
    """Unbatched recordings should chunk and shard their time axis."""
    chunks, shards = resolve_import_sample_storage(
        "float32",
        (2, 2_000_000),
        1,
        batched=False,
        zarr_format=3,
    )

    assert chunks == (2, 131_072)
    assert shards == (2, 524_288)


def test_explicit_shards_override_automatic_sizing() -> None:
    """Callers should retain control over an explicit shard shape."""
    chunks, shards = resolve_import_sample_storage(
        "float32",
        (2, 128),
        100,
        batched=True,
        zarr_format=3,
        sample_shards=(32, 2, 128),
    )

    assert chunks == (1, 2, 128)
    assert shards == (32, 2, 128)


def test_zarr_format_2_rejects_explicit_shards() -> None:
    """Format 2 should reject an explicit format-3 shard shape.

    Raises:
        AssertionError: If the explicit shard shape is accepted.
    """
    try:
        resolve_import_sample_storage(
            "float32",
            (2, 128),
            100,
            batched=True,
            zarr_format=2,
            sample_shards=(32, 2, 128),
        )
    except ValueError as exc:
        assert "requires Zarr format 3" in str(exc)
    else:
        raise AssertionError("Expected format-2 sharding to fail")
