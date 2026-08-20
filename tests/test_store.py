"""Tests for store-level helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from sigmf_zarr.store import SigMFZarrStore


def test_default_sample_chunks_preserves_existing_default() -> None:
    """Default chunking should stay sample-major without a byte target."""
    chunks = SigMFZarrStore.default_sample_chunks(
        "float32",
        (2, 128),
        5000,
        batched=True,
    )

    assert chunks == (1024, 2, 128)


def test_default_sample_chunks_targets_chunk_bytes_for_batches() -> None:
    """Batch chunking should aim for the requested byte size."""
    chunks = SigMFZarrStore.default_sample_chunks(
        "float32",
        (2, 128),
        5000,
        batched=True,
        target_chunk_bytes=1_048_576,
    )

    assert chunks == (1024, 2, 128)


def test_default_sample_chunks_targets_chunk_bytes_for_unbatched() -> None:
    """Unbatched chunking should keep the original sample shape."""
    chunks = SigMFZarrStore.default_sample_chunks(
        "float32",
        (4096, 512),
        1,
        batched=False,
        target_chunk_bytes=1_048_576,
    )

    assert chunks == (4096, 512)


def test_default_sample_chunks_rejects_non_positive_target() -> None:
    """Chunk byte targets must be positive."""
    with pytest.raises(
        ValueError, match="target_chunk_bytes must be positive"
    ):
        SigMFZarrStore.default_sample_chunks(
            "float32",
            (2, 128),
            1,
            target_chunk_bytes=0,
        )


def test_schema_version_is_initial_draft() -> None:
    """Schema version should remain the initial pre-draft format."""
    assert SigMFZarrStore.SCHEMA_VERSION == 1


def test_create_defaults_to_zarr_format_3(tmp_path: Path) -> None:
    """New stores should use Zarr format 3 unless requested otherwise.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)

    assert store.zarr_format == 3
    assert (store_path / "zarr.json").is_file()
