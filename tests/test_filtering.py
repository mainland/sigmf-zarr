"""Filtering semantics and bounded reads over independent snapshots."""

from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array

import sigmf_zarr.store._recording as store_recording
from sigmf_zarr.filtering import FilterOperator, IndexFilter, filter_items
from sigmf_zarr.json import JSONObject, json_object
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat
from tests.chunk_reads import count_chunk_reads


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
    rec = store.recordings.create(
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
    recording: SigMFRecording,
    operation: FilterOperator,
    value: Any,
    expected: list[int],
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

    def read(
        array: Array[Any], selection: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Observe access during the test.

        Args:
            array: Array being observed.
            selection: Array selection.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

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
    # The index is one stored chunk, which is read once for every batch.
    assert reads == [slice(0, 5)]


def test_combined_query_reads_only_candidates(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON queries should examine survivors and retain JSON independence.

    Args:
        recording: Example indexed recording.
        monkeypatch: Patch fixture.
    """
    pytest.importorskip("jmespath")
    original = SigMFRecording._resolved_item_metadata_views
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
        SigMFRecording, "_resolved_item_metadata_views", metadata
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


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_json_filter_decodes_each_chunk_once(
    tmp_path: Path, zarr_format: ZarrFormat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidates from many batches must share one item JSON chunk read.

    Args:
        tmp_path: Temporary directory.
        zarr_format: Physical Zarr format.
        monkeypatch: Patch fixture.
    """
    pytest.importorskip("jmespath")
    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=zarr_format
    )
    rec = store.recordings.create(
        "rec", batched=True, sample_shape=(2, 1), sample_axes=("iq", "time")
    )
    rec.append_samples(np.zeros((30, 2, 1), np.float32))
    rec.add_index("snr", np.arange(30), axis="item", field="t:v")
    rec.set_item_metadata(
        [{"global": {"t:keep": i % 2 == 0}} for i in range(30)]
    )
    assert rec.item_metadata_array.chunks[0] >= 30
    reads = count_chunk_reads(monkeypatch)
    result = filter_items(
        rec,
        [IndexFilter("snr", ">=", 4)],
        metadata_query='global."t:keep"',
        batch_size=4,
    )
    assert result.tolist() == list(range(4, 30, 2))
    item_reads = [n for k, n in reads.items() if "/item_metadata/" in k]
    assert item_reads == [1]


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
    with pytest.raises(ValueError, match=r"[Mm]etadata query"):
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
    progress: list[tuple[int, ...]] = []
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
    with pytest.raises(
        ValueError, match=r"filter operand|filter operator|scalar operands"
    ):
        IndexFilter("a", op, value)  # type: ignore[arg-type]


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
    class_index = recording._indexes_group["class"]
    assert isinstance(class_index, Array)
    class_index[0] = 9
    with pytest.raises(ValueError, match="lookup"):
        filter_items(recording, [IndexFilter("class", "==", "QPSK", True)])
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            filter_items(recording, batch_size=size)  # type: ignore[arg-type]


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
    changed = values[1]["global"]
    assert isinstance(changed, dict)
    changed["test:class"] = "changed"
    unchanged = json_object(values[2]["global"], name="global")
    assert unchanged["test:class"] == "JSON"
    assert rec.resolved_item_metadata_batch([]) == []
    for positions in ([True], [1.5]):
        with pytest.raises(TypeError):
            rec.resolved_item_metadata_batch(positions)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        rec.resolved_item_metadata_batch([5])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_metadata_query_does_not_copy_shared_metadata(
    tmp_path: Path, zarr_format: ZarrFormat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shared annotations must not be copied for each candidate item.

    Args:
        tmp_path: Temporary directory.
        zarr_format: Physical Zarr format.
        monkeypatch: Patch fixture.
    """
    pytest.importorskip("jmespath")
    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=zarr_format
    )
    shared: list[JSONObject] = [
        {"core:sample_start": 0, "core:sample_count": 1, "core:label": "s"}
    ] * 50
    rec = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 8),
        sample_axes=("iq", "time"),
        global_metadata={"test:shared": 1},
        annotations=shared,
    )
    rec.append_samples(np.zeros((4, 2, 8), dtype=np.float32))
    rec.set_item_metadata(
        [
            {},
            {
                "global": {"test:shared": 2},
                "annotations": [
                    {"core:sample_start": 1, "core:label": "local"}
                ],
            },
            {},
            {"global": {"test:local": True}},
        ]
    )
    before = rec.metadata()

    def no_copy(value: object) -> object:
        """Reject deep copies during filtering.

        Args:
            value: Value that would be copied.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("Shared metadata was copied")

    monkeypatch.setattr(store_recording, "deepcopy", no_copy)
    for query, expected in [
        ("length(annotations) == `51`", [1]),
        ("annotations[-1].\"core:label\" == 's'", [0, 2, 3]),
        ('global."test:shared" == `1`', [0, 2, 3]),
        ('global."test:local" && global."test:shared" == `1`', [3]),
    ]:
        assert (
            filter_items(rec, metadata_query=query, batch_size=3).tolist()
            == expected
        )
    monkeypatch.undo()
    assert rec.metadata() == before
    resolved = rec.resolved_item_metadata(1)
    assert len(cast(list[Any], resolved["annotations"])) == 51
