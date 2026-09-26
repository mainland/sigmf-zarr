"""Per-item capture appends preserve independent acquisition metadata."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_append_distinct_item_captures(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Capture lists align with new items and survive subsequent appends.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    path = tmp_path / "store.zarr"
    captures = [
        [{"core:datetime": "2025-01-01T00:00:10.000000001Z"}],
        [
            {"core:datetime": "2025-01-01T00:00:00Z"},
            {"core:sample_start": 4, "core:datetime": "2025-01-01T00:01:00Z"},
        ],
        None,
    ]
    metadata = [
        {"global": {"test:source": "receiver-a"}},
        None,
        {"captures": [{"core:sample_start": 2, "core:frequency": 200}]},
    ]
    original_captures, original_metadata = deepcopy((captures, metadata))
    samples = np.arange(24, dtype=np.float32).reshape(3, 8)
    with SigMFZarrStore.create(path, zarr_format=zarr_format) as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(8,), sample_axes=("time",)
        )
        recording.append_samples(np.zeros((2, 8)))
        recording.append_samples(
            samples,
            capture={"core:frequency": 100},
            item_captures=captures,
            item_metadata=metadata,
        )
        assert recording.get_item_metadata(0) == {}
        assert recording.get_item_metadata(1) == {}
        assert recording.get_item_metadata(2) == {
            **metadata[0],
            "captures": [{**captures[0][0], "core:sample_start": 0}],
        }
        assert recording.get_item_metadata(3) == {
            "captures": [
                {**captures[1][0], "core:sample_start": 0}, captures[1][1]
            ]
        }
        assert recording.get_item_metadata(4) == metadata[2]
        assert recording.captures == [
            {"core:sample_start": 2, "core:frequency": 100}
        ]
        recording.append_samples(np.zeros((1, 8)))
        recording.append_samples(np.zeros((2, 8)), item_captures=[None, []])
        recording.append_samples(np.empty((0, 8)), item_captures=[])
        assert recording.get_item_metadata(5) == {}
        assert recording.get_item_metadata(6) == {}
        assert recording.get_item_metadata(7) == {"captures": []}
        assert recording.item_metadata_array.shape == (8,)
        store.update_integrity()
        assert store.verify_integrity()
    assert captures == original_captures
    assert metadata == original_metadata
    with SigMFZarrStore(path, mode="r") as store:
        recording = store.recordings.open("rec")
        np.testing.assert_array_equal(recording.samples[2:5], samples)
        assert recording.get_item_metadata(2)["captures"][0][
            "core:datetime"
        ] == "2025-01-01T00:00:10.000000001Z"
        assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_capture_start_defaults_to_resolved_item_offset(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Initial defaults respect source offsets and explicit starts survive.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    with SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    ) as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(8,), sample_axes=("time",),
            global_metadata={"core:offset": 100},
        )
        recording.append_samples(
            np.ones((3, 8)),
            item_captures=[
                [{}], [{}, {"core:sample_start": 204}],
                [{"core:sample_start": 103}],
            ],
            item_metadata=[None, {"global": {"core:offset": 200}}, None],
        )
        assert recording.get_item_metadata(0)["captures"] == [
            {"core:sample_start": 100}
        ]
        assert recording.get_item_metadata(1)["captures"] == [
            {"core:sample_start": 200}, {"core:sample_start": 204}
        ]
        assert recording.get_item_metadata(2)["captures"] == [
            {"core:sample_start": 103}
        ]


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    ("captures", "metadata", "error"),
    [
        ([None], None, "Expected captures for 2 appended items"),
        ([None, None], [None], "Expected metadata for 2 appended items"),
        ({0: [], 1: []}, None, "item_captures to be a sequence"),
        ([None, {}], None, "sequence of captures"),
        ([None, [None]], None, "JSON object"),
        ([None, [{"test:value": object()}]], None, "only JSON values"),
        ([None, [{"core:sample_start": -1}]], None, "nonnegative integer"),
        ([None, [{"core:sample_start": 0.5}]], None, "nonnegative integer"),
        ([None, [{"core:sample_start": True}]], None, "nonnegative integer"),
        ([None, [{}, {}]], None, "nonnegative integer"),
        ([None, []], [None, {"captures": []}], "Captures supplied in both"),
        ([None, [{}]], [None, {"unsupported": True}], "unsupported keys"),
        (
            [None, [{}]], [None, {"global": {"core:offset": "bad"}}],
            "nonnegative integer",
        ),
    ],
)
def test_invalid_item_captures_leave_recording_unchanged(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    captures: Any,
    metadata: Any,
    error: str,
) -> None:
    """All capture validation precedes writes and integrity invalidation.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        captures: Invalid capture argument, or valid input with bad metadata.
        metadata: Optional metadata that conflicts or is malformed.
        error: Expected validation error text.
    """
    with SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    ) as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(8,), sample_axes=("time",)
        )
        original = np.arange(8, dtype=np.float32).reshape(1, 8)
        recording.append_samples(original, item_metadata=[None])
        recording.add_index(
            "quality", np.array([1]), axis="item", field="test:quality"
        )
        store.update_integrity()
        original_metadata = recording.metadata()
        with pytest.raises(ValueError, match=error):
            recording.append_samples(
                np.zeros((2, 8)), capture={"core:frequency": 100},
                item_captures=captures, item_metadata=metadata,
            )
        np.testing.assert_array_equal(recording.samples[:], original)
        np.testing.assert_array_equal(recording.index("quality")[:], [1])
        assert recording.metadata() == original_metadata
        assert recording.item_metadata_array.shape == (1,)
        assert recording.get_item_metadata(0) == {}
        assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_unbatched_append_rejects_item_captures(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Unbatched appends cannot assign metadata to independent items.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    with SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    ) as store:
        recording = store.recordings.open(
            "rec", sample_shape=(8,), sample_axes=("time",)
        )
        original = np.arange(8, dtype=np.float32)
        recording.set_samples(original)
        store.update_integrity()
        with pytest.raises(ValueError, match="only supported.*batched"):
            recording.append_samples(np.zeros(8), item_captures=[[{}]])
        np.testing.assert_array_equal(recording.samples[:], original)
        assert store.verify_integrity()
