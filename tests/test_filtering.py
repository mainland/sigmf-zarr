"""Filtering semantics and bounded reads over independent metadata
snapshots.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr.filtering import IndexFilter, filter_items
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat


@pytest.fixture(params=[2, 3])
def recording(
    tmp_path: Path, request: pytest.FixtureRequest
) -> SigMFRecording:
    """Create an indexed recording whose JSON deliberately disagrees.

    Args:
        tmp_path: Temporary directory.
        request: Zarr format parameter.

    Returns:
        Five-item test recording.
    """
    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=cast(ZarrFormat, request.param)
    )
    rec = store.recordings.open(
        "rec",
        batched=True,
        sample_shape=(2, 8),
        sample_axes=("iq", "time"),
        global_metadata={"test:shared": True},
    )
    rec.append_samples(np.arange(80, dtype=np.float32).reshape(5, 2, 8))
    rec.set_item_metadata(
        [
            {"global": {"test:class": "JSON", "test:keep": i % 2 == 0}}
            for i in range(5)
        ]
    )
    rec.add_index(
        "class",
        [0, 1, 0, 1, 0],
        axis="item",
        field="test:class",
        labels=["BPSK", "QPSK"],
    )
    rec.add_index(
        "quality/snr",
        [-10.0, 0.0, 10.0, 20.0, np.nan],
        axis="item",
        field="test:snr",
    )
    return rec


@pytest.mark.parametrize(
    ("operation", "value", "expected"),
    [
        ("==", 10, [2]),
        ("!=", 10, [0, 1, 3]),
        ("<", 0, [0]),
        ("<=", 0, [0, 1]),
        (">", 10, [3]),
        (">=", 10, [2, 3]),
        ("in", [0, 20], [1, 3]),
        ("in", [], []),
    ],
)
def test_numeric_filters(
    recording: SigMFRecording, operation: str, value: Any, expected: list[int]
) -> None:
    """Operators should exclude nonfinite values and preserve source IDs.

    Args:
        recording: Example indexed recording.
        operation: Comparison under test.
        value: Comparison operand.
        expected: Expected original item positions.
    """
    actual = filter_items(
        recording, [IndexFilter("quality/snr", operation, value)], batch_size=2
    )
    assert actual.tolist() == expected


def test_indexes_do_not_read_json_or_samples(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Index filtering should never resolve conflicting JSON or read samples.

    Args:
        recording: Example indexed recording.
        monkeypatch: Patch fixture.
    """
    original = Array.__getitem__
    reads = []

    def read(array: Array, selection: Any, *args: Any, **kwargs: Any) -> Any:
        """Observe access during the test.

        Args:
            array: Array being observed.
            selection: Array selection.

        Returns:
            Observed operation result.
        """
        assert "samples" not in array.path
        assert "item_metadata" not in array.path
        reads.append(selection)
        return original(array, selection, *args, **kwargs)

    monkeypatch.setattr(Array, "__getitem__", read)
    result = filter_items(
        recording, [IndexFilter("class", "==", "QPSK", True)], batch_size=2
    )
    assert result.tolist() == [1, 3]
    assert all(s.stop - s.start <= 2 for s in reads)


def test_combined_query_reads_only_candidates(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON queries should examine survivors and retain JSON independence.

    Args:
        recording: Example indexed recording.
        monkeypatch: Patch fixture.
    """
    pytest.importorskip("jmespath")
    original = SigMFRecording.resolved_item_metadata_batch
    reads = []

    def metadata(rec: SigMFRecording, item: list[int]) -> Any:
        """Observe access during the test.

        Args:
            rec: Recording being observed.
            item: Original candidate positions.

        Returns:
            Observed operation result.
        """
        reads.extend(item)
        return original(rec, item)

    monkeypatch.setattr(
        SigMFRecording, "resolved_item_metadata_batch", metadata
    )
    progress = []
    result = filter_items(
        recording,
        [IndexFilter("quality/snr", ">=", 0)],
        metadata_query='global."test:class" == \'JSON\' && global."test:keep"',
        batch_size=2,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert result.tolist() == [2]
    assert reads == [1, 2, 3]
    assert progress == [(0, 5), (2, 5), (4, 5), (5, 5)]


@pytest.mark.parametrize("query", ["[", "global", "length(missing)", "`0`"])
def test_query_errors_are_not_silent_nonmatches(
    recording: SigMFRecording, query: str
) -> None:
    """Bad syntax, types, and non-Boolean output should fail explicitly.

    Args:
        recording: Example indexed recording.
        query: Metadata expression.
    """
    pytest.importorskip("jmespath")
    with pytest.raises(ValueError):
        filter_items(recording, metadata_query=query)


def test_missing_json_and_cancellation(recording: SigMFRecording) -> None:
    """Null is a nonmatch and cancellation never returns partial results.

    Args:
        recording: Example indexed recording.
    """
    pytest.importorskip("jmespath")
    assert not len(filter_items(recording, metadata_query="missing"))
    with pytest.raises(CancelledError):
        filter_items(recording, cancelled=lambda: True)
    progress = []
    with pytest.raises(CancelledError):
        filter_items(
            recording,
            batch_size=2,
            cancelled=lambda: len(progress) > 1,
            progress=lambda *args: progress.append(args),
        )


def test_integer_precision_and_bad_index(recording: SigMFRecording) -> None:
    """Large integer IDs must retain exact comparisons without float coercion.

    Args:
        recording: Example indexed recording.
    """
    big = 2**63 + 1
    recording.add_index(
        "large",
        np.array([big, big + 1, 0, 1, 2], dtype="u8"),
        axis="item",
        field="test:id",
    )
    assert filter_items(
        recording, [IndexFilter("large", "==", big)]
    ).tolist() == [0]
    assert filter_items(
        recording, [IndexFilter("large", "in", [big])]
    ).tolist() == [0]
    with pytest.raises(ValueError, match="type"):
        filter_items(recording, [IndexFilter("large", "==", "1")])
    recording._indexes_group["class"].attrs["sigmf-zarr:valid"] = False
    with pytest.raises(ValueError, match="invalid"):
        filter_items(recording, [IndexFilter("class", "==", 0)])


@pytest.mark.parametrize(
    ("op", "value"),
    [
        ("bad", 1),
        ("in", 1),
        ("==", []),
        ("==", None),
        ("==", float("inf")),
    ],
)
def test_bad_operands(op: str, value: Any) -> None:
    """Reject malformed filter operands before accessing a store.

    Args:
        op: Comparison under test.
        value: Comparison operand.
    """
    with pytest.raises(ValueError):
        IndexFilter("a", op, value)


def test_axis_masks_labels_and_batch_validation(
    recording: SigMFRecording,
) -> None:
    """Filtering should reject unsupported axes, masks, and label encodings.

    Args:
        recording: Example indexed recording.
    """
    recording.add_index("time", np.arange(8), axis="time", field="test:time")
    with pytest.raises(ValueError, match="dense item"):
        filter_items(recording, [IndexFilter("time", "==", 0)])
    recording._indexes_group["class"].attrs["validity"] = "mask"
    with pytest.raises(ValueError, match="dense item"):
        filter_items(recording, [IndexFilter("class", "==", 0)])
    del recording._indexes_group["class"].attrs["validity"]
    recording._indexes_group["class"][0] = 9
    with pytest.raises(ValueError, match="lookup"):
        filter_items(recording, [IndexFilter("class", "==", "QPSK", True)])
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            filter_items(recording, batch_size=size)


def test_optional_query_dependency(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Index filtering remains usable when JMESPath is unavailable.

    Args:
        recording: Example indexed recording.
        monkeypatch: Patch fixture.
    """
    import sys

    monkeypatch.setitem(sys.modules, "jmespath", None)
    assert filter_items(
        recording, [IndexFilter("class", "==", 1)]
    ).tolist() == [1, 3]
    with pytest.raises(ImportError, match="sigmf-zarr\\[query\\]"):
        filter_items(recording, metadata_query="`true`")


def test_metadata_batch_preserves_order_and_independence(
    recording: SigMFRecording,
) -> None:
    """Batched resolution preserves duplicates without sharing mutable JSON.

    Args:
        recording: Indexed recording with independent item metadata.
    """
    rec = recording
    values = rec.resolved_item_metadata_batch([2, 0, 0, -3])
    assert values[1] == rec.resolved_item_metadata(0)
    assert values[0] == values[3]
    values[1]["global"]["test:class"] = "changed"
    assert values[2]["global"]["test:class"] == "JSON"
    assert rec.resolved_item_metadata_batch([]) == []
    for positions in ([True], [1.5]):
        with pytest.raises(TypeError):
            rec.resolved_item_metadata_batch(positions)
    with pytest.raises(IndexError):
        rec.resolved_item_metadata_batch([5])
