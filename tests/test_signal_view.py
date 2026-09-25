"""One signal has explicit coordinates and independent acquisition anchors."""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.store import SigMFZarrStore


@pytest.mark.parametrize("axes", [("iq", "time"), ("time", "iq")])
def test_signal_view_resolves_metadata_and_reads_bounded_samples(
    tmp_path: Path, axes: tuple[str, ...],
) -> None:
    """Resolve scopes without changing source JSON or reading adjacent items.

    Args:
        tmp_path: Temporary directory.
        axes: Per-item sample layout.
    """
    shape = (2, 4) if axes[0] == "iq" else (4, 2)
    values = np.arange(16, dtype="f4").reshape(2, *shape)
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_dtype="f4", sample_shape=shape,
            sample_axes=axes, global_metadata={"core:offset": 100},
        )
        recording.append_samples(
            values,
            capture={
                "core:frequency": 20, "core:global_index": 1000,
                "core:datetime": "2026-01-01T00:00:00Z",
            },
            item_captures=[None, [
                {"core:sample_start": 102, "core:global_index": 9000},
            ]],
        )
        before = recording.metadata()
        view = recording.signal(1)
        assert view.source_offset == 100
        assert view.sample_rate is None
        assert view.metadata["captures"] == [
            {"core:sample_start": 100, "core:frequency": 20},
            {"core:sample_start": 102, "core:frequency": 20,
             "core:global_index": 9000},
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
