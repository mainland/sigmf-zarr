"""SigMF adapters and piecewise sample-to-time coordinate mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from sigmf_zarr.viewer.regions import Region, Span

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


@dataclass(frozen=True)
class CaptureSegment:
    """Stored interval with independently declared capture timing."""

    start: int
    """First stored sample in the segment."""

    stop: int
    """Exclusive stored sample stop."""

    origin: int
    """Stored sample coordinate corresponding to the declared timestamp."""

    timestamp: float | None
    """UTC POSIX seconds at origin, or unknown."""

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
        stamp = None
        value = entry.get("core:datetime")
        if isinstance(value, str):
            try:
                date = datetime.fromisoformat(value.replace("Z", "+00:00"))
                stamp = (
                    date.replace(tzinfo=UTC).timestamp()
                    if date.tzinfo is None
                    else date.timestamp()
                )
            except (ValueError, OverflowError):
                pass
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
    epoch: float = 0.0,
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
        cast(float, segment.timestamp)
        - epoch
        + (values - segment.origin) / rate
    )


def sigmf_regions(
    metadata: Mapping[str, object],
    length: int,
    *,
    scope: str = "recording",
    source_metadata: Mapping[str, object] | None = None,
) -> tuple[Region, ...]:
    """Adapt capture and annotation entries into named-axis regions.

    Args:
        metadata: SigMF metadata bundle in original coordinates.
        length: Stored signal length.
        scope: Source prefix used in stable region identifiers.
        source_metadata: Original scope supplying entries. Timing and offsets
            still come from the resolved metadata.

    Returns:
        Regions preserving original metadata. Entries outside the recording
        or with invalid intervals have no drawable region.
    """
    segments = capture_segments(metadata, length)
    offset = _offset(metadata)
    result = []
    for kind, field in (
        ("capture", "captures"),
        ("annotation", "annotations"),
    ):
        # Timing uses resolved metadata, but region identity and inspected JSON
        # come from the original scope to distinguish shared and item entries.
        entries = (
            metadata if source_metadata is None else source_metadata
        ).get(field, [])
        for index, entry in enumerate(
            entries if isinstance(entries, list) else []
        ):
            if (
                not isinstance(entry, dict)
                or type(entry.get("core:sample_start")) is not int
            ):
                continue
            start = entry["core:sample_start"] - offset
            end = next(
                (
                    segment.stop
                    for segment in segments
                    if segment.start <= max(0, start) < segment.stop
                ),
                length,
            )
            count = entry.get("core:sample_count")
            if kind == "annotation" and type(count) is int:
                end = start + count
            lo, hi = max(0, start), min(length, end)
            if lo >= hi:
                continue
            bounds = {"sample": Span(lo, hi)}
            lower, upper = (
                entry.get("core:freq_lower_edge"),
                entry.get("core:freq_upper_edge"),
            )
            if (
                kind == "annotation"
                and isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and np.isfinite([lower, upper]).all()
                and lower < upper
            ):
                reference = (
                    "RF"
                    if any(s.frequency is not None for s in segments)
                    else "baseband"
                )
                bounds["frequency"] = Span(lower, upper, "Hz", reference)
            label = str(
                entry.get(
                    "core:description",
                    entry.get(
                        "core:label", f"{kind.capitalize()} {index + 1}"
                    ),
                )
            )
            result.append(
                Region(
                    f"{scope}/{field}/{index}",
                    bounds,
                    label,
                    kind,
                    dict(entry),
                )
            )
    return tuple(result)
