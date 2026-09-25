"""Viewer data access must remain bounded, read-only, and independent of Qt."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array
from zarr.storage import LocalStore

from sigmf_zarr.json import JSONObject
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.viewer import DatasetSource
from sigmf_zarr.viewer.overview import summarize_peaks
from tests.chunk_reads import count_chunk_reads


def nested(metadata: JSONObject | None, key: str) -> JSONObject:
    """Return a nested metadata object without copying it.

    Args:
        metadata: Metadata object that must be present.
        key: Member that must contain a JSON object.

    Returns:
        The member object itself.
    """
    assert metadata is not None
    value = metadata[key]
    assert isinstance(value, dict)
    return value


def test_peak_selection_decodes_named_axes(source: DatasetSource) -> None:
    """Envelope reads must honor item, channel, and time-axis selections.

    Args:
        source: Recording with reversed time/IQ order and two channels.
    """
    raw = source.read_window(
        "rec", item=2, start=3, count=4, coordinates={"channel": 1}
    )
    peaks = source.read_peaks(
        "rec", item=2, start=3, count=4, coordinates={"channel": 1}
    )
    np.testing.assert_array_equal(peaks, np.abs(raw.samples))


@pytest.fixture(params=[2, 3])
def source(tmp_path: Path, request: pytest.FixtureRequest) -> DatasetSource:
    """Create a recording with reversed time/IQ order and two channels.

    Args:
        tmp_path: Temporary directory.
        request: Physical Zarr format.

    Returns:
        Read-only viewer data source.
    """
    path = tmp_path / "data.zarr"
    store = SigMFZarrStore.create(
        path, zarr_format=cast(ZarrFormat, request.param)
    )
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_dtype="float32",
        sample_shape=(8, 2, 2),
        sample_axes=("time", "channel", "iq"),
        global_metadata={"test:class": "shared"},
        channel_metadata=[{"test:receiver": "A"}, {"test:receiver": "B"}],
    )
    recording.append_samples(
        np.arange(96, dtype=np.float32).reshape(3, 8, 2, 2)
    )
    recording.add_index(
        "class",
        [0, 1, 0],
        axis="item",
        field="test:class",
        labels=["BPSK", "QPSK"],
    )
    recording.set_item_metadata(
        [{"global": {"test:class": "JSON"}}, None, None]
    )
    unbatched = store.recordings.create(
        "continuous",
        batched=False,
        sample_dtype="complex64",
        sample_shape=(8,),
        sample_axes=("time",),
    )
    unbatched.set_samples(np.full(8, 3 + 4j, dtype=np.complex64))
    collection = store.collections.create(
        "group", metadata={"description": "Example collection"}
    )
    # Browsing must tolerate dangling references from external writers.
    with collection.mutate() as group:
        group.attrs["recording_ids"] = ["rec", "continuous", "rec", "missing"]
    store.collections.create("empty", recording_ids=[])
    return DatasetSource(path)


@pytest.mark.parametrize(("chunk", "start"), [(140_000, 0), (1000, 500)])
def test_interval_reads_decode_each_chunk_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int, start: int
) -> None:
    """Envelope and overview scans must decode each time chunk once.

    Args:
        tmp_path: Temporary directory.
        monkeypatch: Patch fixture.
        chunk: Stored chunk length along the time axis.
        start: Interval start.
    """
    path = tmp_path / "data.zarr"
    store = SigMFZarrStore.create(path)
    length = 140_000
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_dtype="float32",
        sample_shape=(2, length),
        sample_axes=("iq", "time"),
        sample_chunks=(2, chunk),
    )
    values = np.arange(2 * length, dtype=np.float32).reshape(2, length)
    recording.set_samples(values)
    store.close()
    source = DatasetSource(path)
    count = length - start
    stored = values.astype(np.float64)
    expected = np.abs(stored[0, start:] + 1j * stored[1, start:])
    reads = count_chunk_reads(monkeypatch)
    peaks = source.read_peaks("rec", start=start, count=count, bins=4096)
    blocks = (expected[i : i + 65536] for i in range(0, count, 65536))
    np.testing.assert_allclose(
        peaks, summarize_peaks(blocks, count, bins=4096)
    )
    chunk_reads = [n for k, n in reads.items() if "/samples/" in k]
    assert chunk_reads == [1] * -(-length // chunk)
    reads.clear()
    progress: list[tuple[int, int]] = []
    source.read_overview(
        "rec",
        start=start,
        count=count,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert progress[-1] == (count, count)
    assert all(a[0] < b[0] for a, b in pairwise(progress))
    # The overview also reads one sample of context before its scan.
    chunk_reads = [n for k, n in reads.items() if "/samples/" in k]
    assert sum(chunk_reads) <= -(-length // chunk) + 1


def test_path_location_is_never_parsed_as_url(tmp_path: Path) -> None:
    """A local location keeps `#` as part of the directory name.

    Args:
        tmp_path: Temporary directory.
    """
    path = tmp_path / "run#2.zarr"
    # An explicit backend creates the store at the literal path.
    SigMFZarrStore.create(LocalStore(path)).recordings.create(
        "rec", batched=True, sample_shape=(4,)
    )
    assert DatasetSource(path).recordings() == ("rec",)
    assert DatasetSource(str(path)).path == path
    assert DatasetSource(str(path)).recordings() == ("rec",)


def test_descriptor_reads_do_not_scan_payload(
    source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the browser must read only descriptors, even with item JSON.

    Args:
        source: Example data source.
        monkeypatch: Patch fixture.
    """

    def no_read(*args: Any, **kwargs: Any) -> Any:
        """Observe access during the test."""
        raise AssertionError("Unexpected payload scan")

    monkeypatch.setattr(Array, "__getitem__", no_read)
    assert source.recordings() == ("continuous", "rec")
    catalog = source.catalog()
    assert catalog.recordings == ("continuous", "rec")
    assert catalog.collections == {
        "empty": (),
        "group": ("rec", "continuous", "rec", "missing"),
    }
    metadata = source.collection_metadata("group")
    assert metadata["metadata"] == {"description": "Example collection"}
    assert metadata["recording_ids"] == ["rec", "continuous", "rec", "missing"]
    info = source.describe("rec")
    assert info.item_count == 3
    assert info.axes == ("item", "time", "channel", "iq")
    assert info.indexes["class"]["field"] == "test:class"


def test_window_uses_named_axes_and_preserves_metadata(
    source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read one channel and a bounded time range without projecting labels.

    Args:
        source: Example data source.
        monkeypatch: Patch fixture.
    """
    original = Array.__getitem__
    reads: list[object] = []

    def read(
        array: Array[Any], selection: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Observe access during the test.

        Args:
            array: Array being observed.
            selection: Array selection.
            *args: Additional positional selection arguments.
            **kwargs: Additional keyword selection arguments.

        Returns:
            Observed operation result.
        """
        if array.path.endswith("samples"):
            reads.append(selection)
        return original(array, selection, *args, **kwargs)

    monkeypatch.setattr(Array, "__getitem__", read)
    window = source.read_window(
        "rec",
        item=0,
        start=2,
        count=3,
        coordinates={"channel": 1},
        indexes=["class"],
    )
    np.testing.assert_array_equal(
        window.samples, [10 + 11j, 14 + 15j, 18 + 19j]
    )
    assert reads == [(0, slice(2, 5), 1, slice(None))]
    assert nested(window.metadata, "global")["test:class"] == "JSON"
    assert (
        nested(window.recording_metadata, "global")["test:class"] == "shared"
    )
    assert window.item_metadata == {"global": {"test:class": "JSON"}}
    assert window.channel_metadata is not None
    assert window.channel_metadata["test:receiver"] == "B"
    assert "indexes" not in window.metadata
    assert "test:receiver" not in nested(window.metadata, "global")
    nested(window.metadata, "global")["test:class"] = "changed"
    assert nested(window.item_metadata, "global")["test:class"] == "JSON"
    assert (
        nested(window.recording_metadata, "global")["test:class"] == "shared"
    )
    other = source.read_window("rec", item=1, coordinates={"channel": 0})
    assert other.item_metadata == {}
    assert other.channel_metadata is not None
    assert other.channel_metadata["test:receiver"] == "A"
    continuous = source.read_window("continuous")
    assert continuous.item_metadata is None
    assert continuous.channel_metadata is None
    assert window.indexes["class"]["label"] == "BPSK"
    assert not window.samples.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        window.samples[0] = 0
    assert source.read_window(
        "rec", item=0, start=7, count=99, coordinates={"channel": 0}
    ).samples.shape == (1,)


def test_overview_preserves_named_axes_and_metadata(
    source: DatasetSource,
) -> None:
    """Reduced data must preserve item, channel, and index provenance.

    Args:
        source: Example source in each physical Zarr format.
    """
    view = source.read_overview(
        "rec",
        item=0,
        start=2,
        count=3,
        coordinates={"channel": 1},
        indexes=["class"],
        fft_size=2,
    )
    np.testing.assert_array_equal(
        view.signal.iq, [10 + 11j, 14 + 15j, 18 + 19j]
    )
    assert view.context.samples.size == 0
    assert nested(view.context.metadata, "global")["test:class"] == "JSON"
    context_global = nested(view.context.recording_metadata, "global")
    assert context_global["test:class"] == "shared"
    assert view.context.channel_metadata is not None
    assert view.context.channel_metadata["test:receiver"] == "B"
    assert view.context.indexes["class"]["label"] == "BPSK"


def test_page_preserves_original_positions_and_avoids_json(
    source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pages preserve order and repeats without scanning JSON or samples.

    Args:
        source: Example data source.
        monkeypatch: Patch fixture.
    """
    original = Array.__getitem__

    def read(
        array: Array[Any], selection: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Observe access during the test.

        Args:
            array: Array being observed.
            selection: Array selection.
            *args: Additional positional selection arguments.
            **kwargs: Additional keyword selection arguments.

        Returns:
            Observed operation result.
        """
        assert not array.path.endswith(("samples", "item_metadata"))
        return original(array, selection, *args, **kwargs)

    monkeypatch.setattr(Array, "__getitem__", read)
    rows = source.page("rec", [2, 0, 2, 1], ["class"])
    assert [row["item"] for row in rows] == [2, 0, 2, 1]
    assert nested(rows[-1], "indexes")["class"] == {
        "value": 1,
        "label": "QPSK",
    }
    with pytest.raises(IndexError):
        source.page("rec", [-1])
    with pytest.raises(ValueError, match="at most 4096 rows"):
        source.page("rec", [0] * 4097)


def test_unbatched_and_selection_errors(source: DatasetSource) -> None:
    """Continuous recordings use bounded windows.

    Args:
        source: Example data source.
    """
    assert source.describe("continuous").item_count == 1
    np.testing.assert_array_equal(
        source.read_window("continuous").samples, [3 + 4j] * 8
    )
    with pytest.raises(ValueError, match="batched"):
        source.filter("continuous")
    for kwargs, message in (
        ({"item": 0}, "Select an item only"),
        ({"count": 65537}, "Window count must be between"),
        ({"count": 0}, "Window count must be between"),
        ({"coordinates": {"channel": 0}}, "Select exactly these"),
    ):
        with pytest.raises(ValueError, match=message):
            source.read_window("continuous", **kwargs)
    with pytest.raises(IndexError):
        source.read_window("continuous", start=8)
    with pytest.raises(ValueError, match="Select an item"):
        source.read_window("rec", coordinates={"channel": 0})
