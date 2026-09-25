"""Signal views and shared SigMF capture-coordinate interpretation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

from sigmf_zarr.json import JSONObject, json_object, json_object_list

if TYPE_CHECKING:
    from sigmf_zarr.store import SigMFRecording

AxisMode = Literal["samples", "elapsed", "timestamp"]


def sample_rate(metadata: Mapping[str, object]) -> float | None:
    """Read a finite positive sample rate.

    Args:
        metadata: SigMF metadata bundle.

    Returns:
        Samples per second, or None when unavailable.
    """
    global_info = metadata.get("global", {})
    value = (
        global_info.get("core:sample_rate")
        if isinstance(global_info, dict)
        else None
    )
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and value > 0
        else None
    )


def _offset(metadata: Mapping[str, object]) -> int:
    """Read the original sample-index offset.

    Args:
        metadata: SigMF metadata bundle.

    Returns:
        Original source offset, defaulting to zero.
    """
    global_info = metadata.get("global", {})
    value = (
        global_info.get("core:offset", 0)
        if isinstance(global_info, dict)
        else 0
    )
    return value if type(value) is int else 0


def _item_capture_index(entries: object, item: int) -> int | None:
    """Find the last shared capture applicable to an item.

    Args:
        entries: Shared capture list in storage order.
        item: Selected item position.

    Returns:
        Original list index, or None before the first valid capture.
    """
    selected = None
    latest = -1
    for index, entry in enumerate(
        entries if isinstance(entries, list) else []
    ):
        if isinstance(entry, dict):
            start = entry.get("core:sample_start")
            if type(start) is int and latest <= start <= item and start >= 0:
                selected, latest = index, start
    return selected


def item_capture_metadata(
    metadata: Mapping[str, object],
    recording_metadata: Mapping[str, object],
    item_metadata: Mapping[str, object] | None,
    item: int,
) -> dict[str, object]:
    """Adapt item-offset shared captures to one item's time coordinates.

    Shared capture fields apply until the next item-offset capture. A
    shared timestamp applies only to the item at the declared offset.
    Items are independent, so later item timestamps cannot be inferred.
    Per-item captures retain their original time coordinates and override
    shared fields. Shared acquisition fields supply defaults
    for each per-item capture. Source metadata is unchanged.

    Args:
        metadata: Resolved metadata supplying globals and annotations.
        recording_metadata: Original shared capture list.
        item_metadata: Original per-item captures, if present.
        item: Selected item position.

    Returns:
        Metadata with captures expressed on the selected time axis.
    """
    captures = []
    defaults: dict[str, object] = {}
    entries = recording_metadata.get("captures", [])
    index = _item_capture_index(entries, item)
    if isinstance(entries, list) and index is not None:
        entry = dict(entries[index])
        if entry["core:sample_start"] != item:
            entry.pop("core:datetime", None)
            entry.pop("core:global_index", None)
        entry["core:sample_start"] = _offset(metadata)
        captures.append(entry)
        defaults = {
            key: value for key, value in entry.items()
            if key not in {
                "core:sample_start", "core:datetime", "core:global_index"
            }
        }
    local = (item_metadata or {}).get("captures", [])
    if isinstance(local, list):
        captures.extend(
            {**defaults, **entry} if isinstance(entry, dict) else entry
            for entry in local
        )
    return {**metadata, "captures": captures}


def _capture_timestamp(value: object) -> Fraction | None:
    """Parse UTC seconds without discarding fractional capture precision.

    Args:
        value: ISO timestamp, with naive legacy timestamps treated as UTC.

    Returns:
        Exact POSIX seconds, or None for an invalid or missing timestamp.
    """
    if not isinstance(value, str):
        return None
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        fraction = Fraction(0)
        match = re.search(r"([Tt ]\d{2}:?\d{2}:?\d{2})[.,](\d+)", value)
        if match is not None:
            # datetime validates the original string but retains only
            # microseconds. Use its whole seconds and the original digits.
            fraction = Fraction("0." + match[2])
            date = date.replace(microsecond=0)
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        delta = date - datetime(1970, 1, 1, tzinfo=UTC)
    except (ValueError, OverflowError):
        return None
    return (
        delta.days * 86400 + delta.seconds
        + Fraction(delta.microseconds, 1000000) + fraction
    )


@dataclass(frozen=True)
class CaptureSegment:
    """Stored interval with independently declared capture timing."""

    start: int
    """First stored sample in the segment."""

    stop: int
    """Exclusive stored sample stop."""

    origin: int
    """Stored sample coordinate corresponding to the declared timestamp."""

    timestamp: Fraction | None
    """Exact UTC POSIX seconds at origin, or unknown."""

    frequency: float | None
    """Declared RF center frequency in Hz, or unknown."""

    metadata: Mapping[str, object]
    """Original capture attributes."""


def capture_segments(
    metadata: Mapping[str, object], length: int
) -> tuple[CaptureSegment, ...]:
    """Derive acquisition segments without inferring missing timing fields.

    Duplicate capture starts use the last entry, including item overrides.
    A prefix without capture metadata has unknown timing. Legacy timestamps
    without a timezone are interpreted as UTC.

    Args:
        metadata: Resolved SigMF metadata.
        length: Number of stored samples.

    Returns:
        Ordered segments covering the stored signal.
    """
    entries: dict[int, Mapping[str, object]] = {0: {}}
    captures = metadata.get("captures", [])
    offset = _offset(metadata)
    for entry in captures if isinstance(captures, list) else []:
        if (
            isinstance(entry, dict)
            and type(entry.get("core:sample_start")) is int
        ):
            start = entry["core:sample_start"] - offset
            if start < length:
                # A capture can start before the stored subset. Clip coverage
                # here, but keep its original origin below for timestamp math.
                entries[max(0, start)] = entry
    starts = sorted(entries)
    result = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else length
        if stop <= start:
            continue
        entry = entries[start]
        stamp = _capture_timestamp(entry.get("core:datetime"))
        freq = entry.get("core:frequency")
        frequency = (
            float(freq)
            if isinstance(freq, (int, float)) and np.isfinite(freq)
            else None
        )
        origin = entry.get("core:sample_start", start + offset)
        result.append(
            CaptureSegment(
                start,
                stop,
                cast(int, origin) - offset,
                stamp,
                frequency,
                entry,
            )
        )
    return tuple(result)


def project_samples(
    positions: npt.ArrayLike,
    segment: CaptureSegment,
    rate: float | None,
    mode: AxisMode,
    *,
    epoch: Fraction | float = 0.0,
) -> npt.NDArray[np.float64]:
    """Project coordinates using only the applicable capture's timestamp.

    Args:
        positions: Stored sample coordinates.
        segment: Applicable capture segment.
        rate: Sample rate, if known.
        mode: Sample index, elapsed sample time, or UTC timestamp.
        epoch: POSIX seconds subtracted before adding sample offsets. A nearby
            epoch preserves fine sample spacing when plotting timestamps.

    Returns:
        Display coordinates. Timestamp values are seconds relative to epoch.
        Unknown time coordinates are NaN.
    """
    values = np.asarray(positions, dtype=np.float64)
    if mode == "samples":
        return values
    if rate is None or mode == "timestamp" and segment.timestamp is None:
        return np.full(values.shape, np.nan)
    if mode == "elapsed":
        return values / rate
    # Subtract the nearby epoch before adding fractional sample offsets.
    # Adding them to a full POSIX timestamp first would lose fine spacing.
    return (
        float(cast(Fraction, segment.timestamp) - Fraction(epoch))
        + (values - segment.origin) / rate
    )


def _sample_start(entry: JSONObject) -> int:
    """Read an absolute sample coordinate without accepting Boolean values.

    Args:
        entry: Capture or annotation metadata.

    Returns:
        Nonnegative sample coordinate.

    Raises:
        ValueError: If the coordinate is absent or malformed.
    """
    value = entry.get("core:sample_start")
    if type(value) is not int or value < 0:
        raise ValueError("core:sample_start must be a nonnegative integer")
    return value


class SignalView:
    """One signal with detached metadata and bounded access to stored samples.

    A batch item is independent of adjacent items. Shared capture starts
    remain item offsets in storage and are translated only in this view.
    Keep the recording open and unchanged while using the view. Metadata
    is detached, but sample reads access the recording and do not create
    a frozen snapshot.
    """

    def __init__(
        self, recording: SigMFRecording, item_index: int | None = None
    ) -> None:
        """Resolve one unbatched signal or an explicitly selected batch item.

        Args:
            recording: Source recording, which remains owned by the caller.
            item_index: Batch item, or None for an unbatched signal.

        Raises:
            ValueError: If selection or capture coordinates are invalid.
        """
        if recording.batched:
            if type(item_index) is not int or not 0 <= item_index < len(
                recording
            ):
                raise ValueError("Select an existing nonnegative batch item")
        elif item_index is not None:
            raise ValueError("An unbatched signal has no item index")
        self._recording = recording
        self.item_index = item_index
        self.sample_axes = recording.sample_axes
        self.sample_shape = recording.sample_shape
        self.dtype = np.dtype(recording.samples.dtype)
        self._metadata = self._resolve_metadata()

    item_index: int | None
    """Selected batch position, or None for an unbatched recording."""
    sample_axes: tuple[str, ...]
    """Names of the per-signal axes, excluding item."""
    sample_shape: tuple[int, ...]
    """Stored per-signal shape in sample-axis order."""
    dtype: np.dtype[Any]
    """Stored component dtype, independent of interchange byte order."""

    def _resolve_metadata(self) -> JSONObject:
        """Resolve scopes without interpreting unknown extension fields.

        Returns:
            Detached metadata with sorted absolute capture coordinates.

        Raises:
            ValueError: If metadata coordinates are malformed.
        """
        recording = self._recording
        if self.item_index is None:
            metadata: JSONObject = {
                "global": recording.global_metadata,
                "captures": list(recording.captures),
                "annotations": list(recording.annotations),
            }
        else:
            metadata = recording.resolved_item_metadata(self.item_index)
            local = recording.get_item_metadata(self.item_index)
            for entry in recording.captures:
                _sample_start(entry)
            for entry in json_object_list(
                local.get("captures", []), name="item captures"
            ):
                _sample_start(entry)
            metadata = json_object(
                item_capture_metadata(
                    metadata, recording.metadata(), local, self.item_index
                ),
                name="resolved signal metadata",
            )
            global_info = json_object(metadata["global"], name="global")
            local_global = json_object(local.get("global", {}), name="global")
            if "core:sha512" not in local_global:
                global_info.pop("core:sha512", None)
            metadata["global"] = global_info
        global_info = json_object(metadata["global"], name="global")
        offset = global_info.get("core:offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("core:offset must be a nonnegative integer")
        captures = {
            _sample_start(entry): entry
            for entry in json_object_list(
                metadata["captures"], name="captures"
            )
        }
        metadata["captures"] = [captures[start] for start in sorted(captures)]
        annotations = json_object_list(
            metadata["annotations"], name="annotations"
        )
        annotations.sort(key=_sample_start)
        metadata["annotations"] = [entry for entry in annotations]
        return json_object(metadata, name="signal metadata")

    @property
    def metadata(self) -> JSONObject:
        """Return an independent metadata copy in signal sample coordinates.

        Returns:
            Global, capture, and annotation objects without index projection.
        """
        return json_object(self._metadata, name="signal metadata")

    @property
    def sample_count(self) -> int:
        """Return the time-axis length.

        Returns:
            Number of stored time samples.
        """
        return self.sample_shape[self.sample_axes.index("time")]

    @property
    def sample_rate(self) -> float | None:
        """Return the declared sample rate without inventing a physical clock.

        Returns:
            Positive samples per second, or None when unavailable.
        """
        return sample_rate(self._metadata)

    @property
    def source_offset(self) -> int:
        """Return the absolute coordinate of the first stored time sample.

        Returns:
            Source sample offset, defaulting to zero.
        """
        return _offset(self._metadata)

    @property
    def segments(self) -> tuple[CaptureSegment, ...]:
        """Resolve each capture's independent clock and frequency.

        Returns:
            Capture segments covering the stored sample interval.
        """
        return capture_segments(self._metadata, self.sample_count)

    def read_samples(
        self, start: int = 0, stop: int | None = None
    ) -> npt.NDArray[Any]:
        """Read a detached time interval, retaining the native per-signal axes.

        Args:
            start: Inclusive stored position, independent of source_offset.
            stop: Exclusive stored position, or None for the signal end.

        Returns:
            Sample tensor for the selected interval.

        Raises:
            ValueError: If interval bounds are not valid stored positions.
        """
        stop = self.sample_count if stop is None else stop
        if (
            type(start) is not int or type(stop) is not int
            or not 0 <= start <= stop <= self.sample_count
        ):
            raise ValueError("Invalid stored sample interval")
        key: list[int | slice] = [slice(None)] * len(
            self._recording.runtime_axes
        )
        key[self._recording.axis_index("time")] = slice(start, stop)
        if self.item_index is not None:
            key[0] = self.item_index
        return np.array(self._recording.samples[tuple(key)], copy=True)
