"""One signal has explicit coordinates and independent acquisition anchors."""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat


@pytest.mark.parametrize("axes", [("iq", "time"), ("time", "iq")])
def test_signal_view_resolves_metadata_and_reads_bounded_samples(
    tmp_path: Path,
    axes: tuple[str, ...],
) -> None:
    """Resolve scopes without changing source JSON or reading adjacent items.

    Args:
        tmp_path: Temporary directory.
        axes: Per-item sample layout.
    """
    shape = (2, 4) if axes[0] == "iq" else (4, 2)
    values = np.arange(16, dtype="f4").reshape(2, *shape)
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.create(
            "rec",
            batched=True,
            sample_dtype="f4",
            sample_shape=shape,
            sample_axes=axes,
            global_metadata={"core:offset": 100},
        )
        recording.append_samples(
            values,
            capture={
                "core:frequency": 20,
                "core:global_index": 1000,
                "core:datetime": "2026-01-01T00:00:00Z",
            },
            item_captures=[
                None,
                [
                    {"core:sample_start": 102, "core:global_index": 9000},
                ],
            ],
        )
        before = recording.metadata()
        view = recording.signal(1)
        assert view.source_offset == 100
        assert view.sample_rate is None
        assert view.metadata["captures"] == [
            {"core:sample_start": 100, "core:frequency": 20},
            {
                "core:sample_start": 102,
                "core:frequency": 20,
                "core:global_index": 9000,
            },
        ]
        expected = values[1, :, 1:3] if axes[0] == "iq" else values[1, 1:3]
        np.testing.assert_array_equal(view.read_samples(1, 3), expected)
        assert all(segment.timestamp is None for segment in view.segments)
        assert recording.signal(0).segments[0].timestamp is not None
        assert recording.metadata() == before
        metadata = view.metadata
        metadata["global"] = {}
        assert view.source_offset == 100
        with pytest.raises(ValueError, match="interval"):
            view.read_samples(0, 5)
        with pytest.raises(ValueError, match="batch item"):
            recording.signal()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_signal_view_omits_numbers_outside_float_range(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """JSON integers beyond the float range must be unavailable values.

    Args:
        tmp_path: Temporary directory.
        zarr_format: Physical Zarr format.
    """
    with SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    ) as store:
        recording = store.recordings.create(
            "rec",
            sample_dtype="complex64",
            sample_shape=(4,),
            sample_axes=("time",),
            global_metadata={"core:sample_rate": 10**400},
            captures=[
                {"core:sample_start": 0, "core:frequency": 10**400},
                {"core:sample_start": 2, "core:frequency": 2**64},
            ],
        )
        view = recording.signal()
        assert view.sample_rate is None
        assert [s.frequency for s in view.segments] == [None, 2.0**64]
        recording.set_global_field("core:sample_rate", 10**12)
        assert recording.signal().sample_rate == 1e12


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_signal_names_per_item_capture_without_start(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """A per-item capture without a start must be named in the error.

    Args:
        tmp_path: Temporary directory.
        zarr_format: Physical Zarr format to create.
    """
    with SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    ) as store:
        recording = store.recordings.create(
            "rec", batched=True, sample_shape=(4,)
        )
        recording.append_samples(
            np.zeros((2, 4), dtype=np.float32),
            item_metadata=[
                None,
                {"captures": [{"core:datetime": "2026-08-19T00:00:00Z"}]},
            ],
        )
        recording.signal(0)
        with pytest.raises(
            ValueError,
            match=r"Item 1 capture 0 has no nonnegative integer "
            r"core:sample_start\. Signal resolution requires an explicit "
            "start",
        ):
            recording.signal(1)
