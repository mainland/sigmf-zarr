"""Regression tests for sample mutation preflight validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigmf_zarr.store import SigMFZarrStore


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("replacement", [["bad"], []])
def test_rejected_sample_replacement_preserves_recording(
    tmp_path: Path,
    zarr_format: int,
    replacement: list[Any],
) -> None:
    """Rejected replacements must preserve samples, metadata, and hashes.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        replacement: Invalid replacement sample values.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
    )
    original = np.arange(4, dtype=np.float32)
    recording.set_samples(original)
    recording.add_index(
        "quality", original, axis="time", field="test:quality"
    )
    store.update_integrity()
    metadata = recording.metadata()

    with pytest.raises(ValueError):
        recording.set_samples(replacement)

    np.testing.assert_array_equal(recording.samples[:], original)
    np.testing.assert_array_equal(recording.index("quality")[:], original)
    assert recording.metadata() == metadata
    assert store.verify_integrity()
    assert store.recordings.open("rec").sample_shape == (4,)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("invalid_field", ["samples", "capture"])
def test_rejected_sample_append_preserves_recording(
    tmp_path: Path,
    zarr_format: int,
    batched: bool,
    invalid_field: str,
) -> None:
    """Invalid append values must be rejected before resizing storage.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        batched: Whether appends extend the item axis.
        invalid_field: Append input containing an invalid value.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=batched,
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
    )
    original = np.arange(4, dtype=np.float32)
    if batched:
        recording.append_samples(
            original.reshape(1, 4), item_metadata=[{"global": {"test:x": 1}}]
        )
    else:
        recording.set_samples(original)
    recording.add_index(
        "quality",
        np.arange(len(recording) if batched else 4),
        axis="item" if batched else "time",
        field="test:quality",
    )
    store.update_integrity()
    original_samples = recording.samples[:]
    metadata = recording.metadata()
    appended = np.full(
        (1, 4) if batched else (1,),
        "bad" if invalid_field == "samples" else 1.0,
    )
    capture = {"test:value": object()} if invalid_field == "capture" else None

    with pytest.raises(ValueError):
        recording.append_samples(appended, capture=capture)

    np.testing.assert_array_equal(recording.samples[:], original_samples)
    assert recording.metadata() == metadata
    assert store.verify_integrity()
    if batched:
        assert recording.item_metadata_array.shape == (1,)
        assert recording.get_item_metadata(0) == {"global": {"test:x": 1}}


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_rejected_item_metadata_preserves_batch(
    tmp_path: Path, zarr_format: int
) -> None:
    """Per-item metadata validation must precede sample mutation.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    recording = store.recordings.open(
        "rec", create=True, batched=True, sample_shape=(4,)
    )
    recording.append_samples(np.ones((1, 4)), item_metadata=[None])
    store.update_integrity()
    metadata = recording.metadata()

    with pytest.raises(ValueError, match="unsupported keys"):
        recording.append_samples(
            np.zeros((1, 4)), item_metadata=[{"unsupported": True}]
        )

    np.testing.assert_array_equal(recording.samples[:], np.ones((1, 4)))
    assert recording.item_metadata_array.shape == (1,)
    assert recording.metadata() == metadata
    assert store.verify_integrity()
