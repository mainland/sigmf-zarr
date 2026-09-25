"""Read-only, bounded dataset access independent of any GUI framework."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from os import PathLike, fspath
from typing import Any

import numpy as np
import numpy.typing as npt

from sigmf_zarr.filtering import IndexFilter, filter_items
from sigmf_zarr.indexes import (
    _read_index_selection,
    validate_categorical_values,
)
from sigmf_zarr.json import JSONObject, json_object
from sigmf_zarr.query import IndexQuery
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore
from sigmf_zarr.viewer.captures import capture_segments, item_capture_metadata
from sigmf_zarr.viewer.overview import (
    SignalOverview,
    summarize_peaks,
    summarize_signal,
)
from sigmf_zarr.viewer.windows import WindowFunction


@dataclass(frozen=True)
class DatasetCatalog:
    """Dataset hierarchy without recording payloads or item metadata."""

    recordings: tuple[str, ...]
    """All recording names in sorted order."""

    collections: dict[str, tuple[str, ...]]
    """Collection names mapped to recording references in collection order."""


@dataclass(frozen=True)
class RecordingInfo:
    """Recording descriptors without reading samples or item JSON."""

    name: str
    """Recording name."""

    shape: tuple[int, ...]
    """Full sample shape, including the item axis when batched."""

    axes: tuple[str, ...]
    """Runtime axis names corresponding to shape."""

    batched: bool
    """Whether records correspond to an explicit item axis."""

    indexes: dict[str, JSONObject]
    """Descriptors for item indexes, including invalid indexes."""

    metadata: JSONObject = field(default_factory=dict)
    """Stored recording attributes, without item overrides."""

    @property
    def item_count(self) -> int:
        """Number of browsable records.

        Returns:
            Item count, or one for an unbatched recording.
        """
        return self.shape[0] if self.batched else 1


@dataclass(frozen=True)
class SampleWindow:
    """Selected samples and independent metadata snapshots for consumers."""

    recording: str
    """Recording name."""

    item: int | None
    """Original item position, or None for an unbatched recording."""

    start: int
    """Start position on the item's or recording's time axis."""

    coordinates: dict[str, int]
    """Selected positions on axes other than item, time, and iq."""

    samples: npt.NDArray[Any]
    """Read-only real or complex vector.

    An iq axis is decoded as I + jQ.
    """

    metadata: JSONObject
    """Resolved item JSON, or shared metadata for an unbatched recording."""

    indexes: dict[str, JSONObject]
    """Explicitly selected indexes with raw values, attributes, and labels."""

    recording_metadata: JSONObject = field(default_factory=dict)
    """Stored recording attributes, without item overrides."""

    item_metadata: JSONObject | None = None
    """Stored item bundle, or None when the item metadata array is absent."""

    channel_metadata: JSONObject | None = None
    """Selected channel attributes, or None without a channel axis."""

    total_samples: int | None = None
    """Full stored signal length, independent of the selected interval."""

    @property
    def signal_metadata(self) -> Mapping[str, object]:
        """Metadata with capture coordinates adapted to the selected signal.

        Returns:
            Derived time-axis captures for a batched item, or the original
            metadata for an unbatched signal or a window without source
            scopes. Stored and resolved JSON snapshots remain unchanged.
        """
        if self.item is None or not self.recording_metadata:
            return self.metadata
        return item_capture_metadata(
            self.metadata, self.recording_metadata, self.item_metadata,
            self.item,
        )


@dataclass(frozen=True)
class OverviewWindow:
    """Explicit reduced data for a selected interval, without raw samples."""

    context: SampleWindow
    """Selection coordinates and metadata, with an empty sample vector."""

    signal: SignalOverview
    """Reduced data covering the entire selected interval."""


def _index_values(
    recording: SigMFRecording, names: Sequence[str], positions: Sequence[int]
) -> dict[str, list[JSONObject]]:
    """Read a page of selected indexes without reading item JSON.

    Args:
        recording: Structurally opened recording.
        names: Explicit item index names.
        positions: Original item positions in requested order.

    Returns:
        Per-index lists of raw values and optional decoded labels.

    Raises:
        ValueError: If the selected index or its labels are invalid.
    """
    result: dict[str, list[JSONObject]] = {}
    for name in names:
        array = recording.index(name)
        if array.attrs.get("axis") != "item" or "validity" in array.attrs:
            raise ValueError("The record browser requires dense item indexes")
        values = _read_index_selection(array, positions)
        labels = None
        if "labels" in array.attrs:
            labels = validate_categorical_values(
                values, labels=array.attrs["labels"]
            )
        result[name] = []
        for value in values:
            entry: JSONObject = {
                "value": value.item()
                if isinstance(value, np.generic)
                else value
            }
            if labels is not None:
                entry["label"] = labels[int(value)]
            result[name].append(entry)
    return result


class DatasetSource:
    """Read-only access with an independent store handle for each operation.

    Each call owns and closes its handle, so workers do not share a
    backend handle with the GUI thread. Callers should keep datasets
    unchanged while browsing. These methods do not create a
    transactional snapshot.
    """

    path: str
    """Store path or URL."""

    storage_options: dict[str, Any] | None
    """Backend-specific options supplied by the embedding application."""

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        """Remember a dataset location without opening it.

        Args:
            path: Store path or URL.
            storage_options: Optional backend configuration.
        """
        self.path = fspath(path)
        self.storage_options = storage_options

    @contextmanager
    def _store(self) -> Generator[SigMFZarrStore, None, None]:
        """Own one read-only store handle for the duration of an operation.

        Yields:
            Opened read-only store.
        """
        store = SigMFZarrStore.open(
            self.path, mode="r", storage_options=self.storage_options
        )
        try:
            yield store
        finally:
            store._group.store.close()

    @contextmanager
    def _recording(self, name: str) -> Generator[SigMFRecording, None, None]:
        """Open a recording without scanning all item metadata.

        Args:
            name: Recording name.

        Yields:
            Structurally validated recording.
        """
        with self._store() as store:
            yield store.recordings.open(name, validation="structural")

    def recordings(self) -> tuple[str, ...]:
        """List recording names without opening their contents.

        Returns:
            Sorted recording names.
        """
        with self._store() as store:
            return tuple(sorted(store.list_recordings()))

    def catalog(self) -> DatasetCatalog:
        """Read the recording names and collection membership for a tree.

        Returns:
            Hierarchy preserving shared, duplicate, and missing recording
            references. No recording is opened and no array values are read.
        """
        with self._store() as store:
            names = tuple(sorted(store.list_recordings()))
            return DatasetCatalog(
                names,
                {
                    name: store.collections.open(
                        name, create=False
                    ).recording_ids
                    for name in sorted(store.list_collections())
                },
            )

    def collection_metadata(self, name: str) -> JSONObject:
        """Read collection metadata and its recording references.

        Args:
            name: Collection name.

        Returns:
            Independent collection metadata and ordered recording references.
        """
        with self._store() as store:
            collection = store.collections.open(name, create=False)
            return json_object(
                {
                    "metadata": collection.metadata,
                    "recording_ids": list(collection.recording_ids),
                },
                name="collection metadata",
            )

    def describe(self, name: str) -> RecordingInfo:
        """Read recording and item index descriptors.

        Args:
            name: Recording name.

        Returns:
            Shape, axes, and item index attributes without payload reads.
        """
        with self._recording(name) as recording:
            indexes = {
                key: json_object(dict(member.attrs), name="index attributes")
                for key, member in recording.indexes.members(max_depth=None)
                if isinstance(member, ReadOnlyArray)
                and member.attrs.get("axis") == "item"
            }
            return RecordingInfo(
                name,
                recording.samples.shape,
                recording.runtime_axes,
                recording.batched,
                indexes,
                json_object(recording.metadata(), name="recording metadata"),
            )

    def filter(
        self,
        name: str,
        filters: Sequence[IndexFilter] = (),
        *,
        index_query: str | IndexQuery | None = None,
        metadata_query: str | None = None,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Find matching original item positions in a worker-owned handle.

        Args:
            name: Recording name.
            filters: Explicit index predicates combined with AND.
            index_query: Optional core index query, combined with filters.
            metadata_query: Optional JMESPath Boolean predicate.
            cancelled: Cooperative cancellation callback.
            progress: Callback receiving examined and total item counts.

        Returns:
            Ascending original item positions.
        """
        with self._recording(name) as recording:
            return filter_items(
                recording,
                filters,
                index_query=index_query,
                metadata_query=metadata_query,
                cancelled=cancelled,
                progress=progress,
            )

    def page(
        self,
        name: str,
        positions: Sequence[int],
        indexes: Sequence[str] = (),
    ) -> list[JSONObject]:
        """Read selected index columns for a browser page.

        Args:
            name: Recording name.
            positions: Original item positions, preserving order and repeats.
            indexes: Explicitly selected item index names.

        Returns:
            Rows containing the original item and independent index values.

        Raises:
            ValueError: If the page exceeds 4096 rows or an index is invalid.
            IndexError: If an item position is invalid.
        """
        if len(positions) > 4096:
            raise ValueError("Browser pages must contain at most 4096 rows")
        with self._recording(name) as recording:
            count = len(recording) if recording.batched else 1
            for position in positions:
                _position(position, count, "item")
            columns = _index_values(recording, indexes, positions)
            return [
                {
                    "item": int(item),
                    "indexes": {
                        key: values[row] for key, values in columns.items()
                    },
                }
                for row, item in enumerate(positions)
            ]

    def read_peaks(
        self,
        name: str,
        *,
        item: int | None = None,
        start: int = 0,
        count: int,
        coordinates: Mapping[str, int] | None = None,
        bins: int = 1024,
        cancelled: Callable[[], bool] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Read a magnitude envelope without computing spectra or metadata.

        Args:
            name: Recording name.
            item: Original item position for a batched recording.
            start: Inclusive time-axis coordinate.
            count: Exact positive sample count.
            coordinates: Positions for additional sample axes.
            bins: Maximum number of peak bins, from 1 through 4096.
            cancelled: Cancellation callback checked between bounded reads.

        Returns:
            Peak magnitudes over uniformly spaced integer sample bins.

        Raises:
            ValueError: If the settings or selection are invalid.
            IndexError: If the interval exceeds the recording.
            CancelledError: If cancellation is requested.
        """
        if cancelled is not None and cancelled():
            raise CancelledError()
        coordinates = dict(coordinates or {})
        with self._recording(name) as recording:
            length = recording.samples.shape[
                recording.runtime_axes.index("time")
            ]
            if type(count) is not int or count < 1:
                raise ValueError("Peak count must be a positive integer")
            _selection(recording, item, start, count, coordinates)
            if start + count > length:
                raise IndexError("Peak interval is out of range")

            def blocks() -> Generator[npt.NDArray[Any], None, None]:
                """Yield bounded decoded blocks for the envelope.

                Yields:
                    Up to 65536 decoded samples per block.

                Raises:
                    CancelledError: If cancellation is requested.
                """
                for offset in range(0, count, 65536):
                    if cancelled is not None and cancelled():
                        raise CancelledError()
                    yield _read_samples(
                        recording,
                        _selection(
                            recording,
                            item,
                            start + offset,
                            min(65536, count - offset),
                            coordinates,
                        ),
                    )
                if cancelled is not None and cancelled():
                    raise CancelledError()

            return summarize_peaks(blocks(), count, bins=bins)

    def read_overview(
        self,
        name: str,
        *,
        item: int | None = None,
        start: int = 0,
        count: int,
        coordinates: Mapping[str, int] | None = None,
        indexes: Sequence[str] = (),
        fft_size: int = 256,
        overlap: float = 0.5,
        window_function: WindowFunction = "hann",
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> OverviewWindow:
        """Scan an interval in bounded blocks to produce a reduced view.

        Args:
            name: Recording name.
            item: Original item position, required for batched recordings.
            start: Nonnegative time-axis start.
            count: Exact number of source samples, at least two.
            coordinates: Positions for every additional sample axis.
            indexes: Explicit indexes to include independently of JSON.
            fft_size: FFT length, from 2 through 8192.
            overlap: Frame overlap fraction, from zero through 0.875.
            window_function: Periodic FFT window, defaulting to Hann.
            cancelled: Cooperative cancellation callback, checked per block.
            progress: Callback receiving examined and total sample counts.

        Returns:
            Metadata and a reduced representation of the entire interval.

        Raises:
            ValueError: If the settings or selection are invalid.
            IndexError: If the interval extends outside the recording.
            CancelledError: If cancellation is requested between blocks.
        """
        if cancelled is not None and cancelled():
            raise CancelledError()
        coordinates = dict(coordinates or {})
        with self._recording(name) as recording:
            length = recording.samples.shape[
                recording.runtime_axes.index("time")
            ]
            if type(count) is not int or count < 2:
                raise ValueError(
                    "Overview count must be an integer of at least two"
                )
            _selection(recording, item, start, count, coordinates)
            if start + count > length:
                raise IndexError("Overview interval is out of range")
            context = self.read_window(
                name,
                item=item,
                start=start,
                count=1,
                coordinates=coordinates,
                indexes=indexes,
            )

            def blocks() -> Generator[npt.NDArray[Any], None, None]:
                """Yield consecutive decoded blocks from one owned handle.

                Yields:
                    Up to 65536 decoded samples per block.

                Raises:
                    CancelledError: If cancellation is requested.
                """
                for offset in range(0, count, 65536):
                    if cancelled is not None and cancelled():
                        raise CancelledError()
                    size = min(65536, count - offset)
                    yield _read_samples(
                        recording,
                        _selection(
                            recording, item, start + offset, size, coordinates
                        ),
                    )
                    if progress is not None:
                        progress(offset + size, count)
                if cancelled is not None and cancelled():
                    raise CancelledError()

            signal = summarize_signal(
                blocks(),
                count,
                fft_size=fft_size,
                window_function=window_function,
                overlap=overlap,
                # The reducer starts at zero for this read interval. Translate
                # interior capture boundaries from recording coordinates.
                boundaries=tuple(
                    segment.start - start
                    for segment in capture_segments(
                        context.signal_metadata, length
                    )
                    if start < segment.start < start + count
                ),
            )
            return OverviewWindow(
                # Keep the metadata context but remove its probe sample so a
                # consumer cannot mistake reduced data for a raw waveform.
                replace(context, samples=context.samples[:0]), signal
            )

    def read_window(
        self,
        name: str,
        *,
        item: int | None = None,
        start: int = 0,
        count: int = 1024,
        coordinates: Mapping[str, int] | None = None,
        indexes: Sequence[str] = (),
    ) -> SampleWindow:
        """Read one bounded time window and its metadata snapshots.

        Args:
            name: Recording name.
            item: Original item position. Required for batched recordings.
            start: Nonnegative time-axis start.
            count: Requested samples, from 1 through 65536. Clipped at the end.
            coordinates: Explicit positions for every additional sample axis.
            indexes: Explicit item indexes to include independently of JSON.

        Returns:
            Selected real or complex samples and metadata. Coordinates remain
            relative to the source item. Metadata is not clipped or projected.

        Raises:
            ValueError: If the selection or sample layout is unsupported.
            IndexError: If a selected coordinate is outside its axis.
        """
        if type(count) is not int or not 1 <= count <= 65536:
            raise ValueError("Window count must be between 1 and 65536")
        coordinates = dict(coordinates or {})
        with self._recording(name) as recording:
            selection = _selection(recording, item, start, count, coordinates)
            samples = _read_samples(recording, selection)
            samples.setflags(write=False)
            metadata = (
                recording.resolved_item_metadata(item)
                if item is not None
                else recording.metadata()
            )
            values = _index_values(
                recording, indexes, [item] if item is not None else []
            )
            selected = {
                key: {
                    **column[0],
                    "attributes": dict(recording.index(key).attrs),
                }
                for key, column in values.items()
            }
            return SampleWindow(
                name,
                item,
                start,
                coordinates,
                samples,
                json_object(metadata, name="metadata"),
                selected,
                recording_metadata=json_object(
                    recording.metadata(), name="recording metadata"
                ),
                item_metadata=(
                    recording.get_item_metadata(item)
                    if item is not None and recording.has_item_metadata
                    else None
                ),
                channel_metadata=(
                    json_object(
                        recording.channel_metadata(coordinates["channel"]),
                        name="channel metadata",
                    )
                    if "channel" in coordinates
                    else None
                ),
                total_samples=recording.samples.shape[
                    recording.runtime_axes.index("time")
                ],
            )


def _read_samples(
    recording: SigMFRecording, selection: tuple[int | slice, ...]
) -> npt.NDArray[Any]:
    """Decode a named-axis selection into a detached real or complex vector.

    Args:
        recording: Structurally opened recording.
        selection: Validated basic selection.

    Returns:
        Detached decoded samples.

    Raises:
        ValueError: If the stored sample type is unsupported.
    """
    samples = np.asarray(recording.samples[selection])
    # Scalar selections removed all other axes. Recover I/Q's position from
    # stored axis order, which need not place it before time.
    remaining = [a for a in recording.runtime_axes if a in {"iq", "time"}]
    if "iq" in remaining:
        if samples.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("An iq axis requires real numeric samples")
        iq = np.moveaxis(samples, remaining.index("iq"), 0)
        # Promote integer components before downstream magnitude and power
        # calculations so their arithmetic cannot overflow the stored dtype.
        samples = iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)
    elif samples.dtype.kind in {"i", "u", "f", "c"}:
        dtype = np.complex128 if np.iscomplexobj(samples) else np.float64
        samples = samples.astype(dtype)
    else:
        raise ValueError(
            "Viewer samples must be real or complex numeric values"
        )
    return samples


def _position(value: int, length: int, axis: str) -> int:
    """Validate a nonnegative integer coordinate.

    Args:
        value: Selected position.
        length: Axis length.
        axis: Axis name for diagnostics.

    Returns:
        Integer position.

    Raises:
        IndexError: If the position is not a valid nonnegative integer.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise IndexError(f"{axis} position must be an integer")
    if not 0 <= value < length:
        raise IndexError(f"{axis} position is out of range")
    return int(value)


def _selection(
    recording: SigMFRecording,
    item: int | None,
    start: int,
    count: int,
    coordinates: Mapping[str, int],
) -> tuple[int | slice, ...]:
    """Build a slice using named axes without assuming I/Q storage order.

    Args:
        recording: Selected recording.
        item: Optional item position.
        start: Window start.
        count: Requested window length.
        coordinates: Additional axis selections.

    Returns:
        Basic indexing tuple that reads only the selected window.

    Raises:
        ValueError: If required coordinates or I/Q layout are unsupported.
        IndexError: If a selected coordinate is outside its axis.
    """
    axes = dict(
        zip(recording.runtime_axes, recording.samples.shape, strict=True)
    )
    extra = set(axes) - {"item", "time", "iq"}
    if set(coordinates) != extra:
        raise ValueError(
            f"Select exactly these additional axes: {sorted(extra)}"
        )
    if recording.batched != (item is not None):
        raise ValueError("Select an item only for batched recordings")
    if "iq" in axes and axes["iq"] != 2:
        raise ValueError("The iq axis must contain exactly I and Q")
    _position(start, axes["time"], "time")
    result: list[int | slice] = []
    for axis, length in axes.items():
        if axis == "time":
            result.append(slice(start, min(start + count, length)))
        elif axis == "iq":
            result.append(slice(None))
        else:
            value = item if axis == "item" else coordinates[axis]
            assert value is not None
            result.append(_position(value, length, axis))
    return tuple(result)
