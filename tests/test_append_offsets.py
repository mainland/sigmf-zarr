"""Unbatched capture defaults use absolute source sample coordinates."""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_append_capture_respects_source_offset(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Add the stored time count to the source offset for a new capture.

    Args:
        tmp_path: Temporary directory fixture.
        zarr_format: Physical Zarr format.
    """
    with SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    ) as store:
        recording = store.recordings.open(
            "rec", sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:offset": 1000},
        )
        recording.append_samples(np.ones(2), capture={"core:frequency": 100})
        assert recording.captures == [{
            "core:sample_start": 1004, "core:frequency": 100,
        }]
        assert recording.sample_count == 6


@pytest.mark.parametrize("offset", [-1, True, 1.5])
def test_invalid_append_offset_preserves_samples(
    tmp_path: Path, offset: object
) -> None:
    """Reject invalid offsets before resizing or invalidating source data.

    Args:
        tmp_path: Temporary directory fixture.
        offset: Invalid source sample offset.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:offset": offset},
        )
        before = recording.metadata()
        with pytest.raises(ValueError, match="core:offset"):
            recording.append_samples(np.ones(2), capture={})
        assert recording.sample_count == 4
        assert recording.captures == []
        assert recording.metadata() == before
