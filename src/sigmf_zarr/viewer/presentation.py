"""Capture-aware plotting coordinates and region projection."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from math import ceil, floor

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.ticker import FuncFormatter, MaxNLocator

from sigmf_zarr.viewer.captures import (
    AxisMode,
    CaptureSegment,
    capture_segments,
    project_samples,
    sample_rate,
    sigmf_regions,
)
from sigmf_zarr.viewer.data import SampleWindow
from sigmf_zarr.viewer.overlays import draw_regions
from sigmf_zarr.viewer.regions import Region, Span


@dataclass
class PlotOptions:
    """Shared display settings independent of signal processing."""

    axis_mode: AxisMode = "samples"
    """Coordinate system used for the sample axis."""

    captures: bool = True
    """Whether capture overlays are visible."""

    annotations: bool = True
    """Whether annotation overlays are visible."""


def sample_interval(
    window: SampleWindow,
    count: int,
    limits: tuple[float, float],
    options: PlotOptions,
) -> tuple[int, int] | None:
    """Map visible plot coordinates to a containing stored sample interval.

    Timestamp overlaps include every matching capture. Disjoint matches use
    their smallest containing interval because the scrubber selects one range.
    Timestamp gaps and unknown timing do not produce sample selections.

    Args:
        window: Metadata and full source length for the displayed signal.
        count: Displayed count when the full source length is unavailable.
        limits: Visible sample-axis coordinates, in either direction.
        options: Coordinate system used by the plot.

    Returns:
        Clamped half-open sample interval, or None when no samples match.
    """
    lo, hi = sorted(limits)
    if not np.isfinite([lo, hi]).all() or lo == hi:
        return None
    length = window.total_samples or window.start + count
    rate = sample_rate(window.signal_metadata)
    if options.axis_mode != "samples" and rate is None:
        return None
    scale = rate or 1.0
    epoch = timestamp_origin(window) if options.axis_mode == "timestamp" else 0
    intervals = []
    for capture in capture_segments(window.signal_metadata, length):
        if options.axis_mode == "samples":
            start, stop = lo, hi
        elif options.axis_mode == "elapsed":
            start, stop = lo * scale, hi * scale
        else:
            if capture.timestamp is None:
                continue
            origin = float(capture.timestamp - epoch)
            start = capture.origin + (lo - origin) * scale
            stop = capture.origin + (hi - origin) * scale
        # Round outward to contain fractional plot limits, then intersect with
        # this capture. Timestamp gaps must not create synthetic sample ranges.
        start = max(capture.start, floor(start))
        stop = min(capture.stop, ceil(stop))
        if start < stop:
            intervals.append((start, stop))
    if not intervals:
        return None
    return min(s for s, _ in intervals), max(e for _, e in intervals)


def window_regions(window: SampleWindow, count: int) -> tuple[Region, ...]:
    """Adapt regions while preserving recording and item source scopes.

    Args:
        window: Original scopes and resolved metadata.
        count: Number of displayed samples when full length is unavailable.

    Returns:
        Regions referencing the original recording or item entry.
    """
    length = window.total_samples or window.start + count
    regions = sigmf_regions(
        window.signal_metadata,
        length,
        source_metadata=window.recording_metadata or window.metadata,
        capture_item=window.item if window.recording_metadata else None,
    )
    if window.item_metadata is not None:
        regions += sigmf_regions(
            window.signal_metadata,
            length,
            scope="item",
            source_metadata=window.item_metadata,
        )
    return regions


def segments(window: SampleWindow, count: int) -> Iterator[CaptureSegment]:
    """Yield captures intersecting a displayed window.

    Args:
        window: Source samples and metadata.
        count: Number of represented samples.

    Yields:
        Clipped segments retaining their timestamp origins.
    """
    end = window.start + count
    for segment in capture_segments(
        window.signal_metadata, window.total_samples or end
    ):
        lo, hi = max(window.start, segment.start), min(end, segment.stop)
        if lo < hi:
            yield replace(segment, start=lo, stop=hi)


def timestamp_origin(window: SampleWindow) -> Fraction:
    """Choose a shared timestamp origin before converting to floating point.

    Args:
        window: Source metadata.

    Returns:
        Earliest declared capture timestamp, or zero when unknown.
    """
    stamps = [
        c.timestamp
        for c in capture_segments(
            window.signal_metadata,
            window.total_samples or window.start + max(1, len(window.samples)),
        )
        if c.timestamp is not None
    ]
    return min(stamps, default=Fraction(0))


def project(
    window: SampleWindow,
    positions: npt.ArrayLike,
    segment: CaptureSegment,
    options: PlotOptions,
) -> npt.NDArray[np.float64]:
    """Project samples relative to a nearby epoch for timestamp precision.

    Args:
        window: Signal metadata.
        positions: Source sample positions.
        segment: Applicable acquisition segment.
        options: Requested axis mode.

    Returns:
        Display positions in samples or seconds.
    """
    return project_samples(
        positions,
        segment,
        sample_rate(window.signal_metadata),
        options.axis_mode,
        epoch=timestamp_origin(window)
        if options.axis_mode == "timestamp"
        else 0,
    )


def _position_limits(
    axes: Axes,
    window: SampleWindow,
    count: int,
    options: PlotOptions,
    vertical: bool,
) -> None:
    """Set the position axis to the represented sample interval.

    Args:
        axes: Target plot.
        window: Signal metadata.
        count: Number of represented samples.
        options: Coordinate mapping settings.
        vertical: Whether position is the vertical coordinate.
    """
    extents = [
        project(window, [c.start, c.stop], c, options)
        for c in segments(window, count)
    ]
    finite = [v for v in extents if np.isfinite(v).all()]
    if finite:
        lo, hi = min(v[0] for v in finite), max(v[1] for v in finite)
        if vertical:
            axes.set_ylim(hi, lo)
        else:
            axes.set_xlim(lo, hi)


def decorate(
    axes: Axes,
    window: SampleWindow,
    count: int,
    options: PlotOptions,
    *,
    vertical: bool = False,
    frequency_axis: bool = False,
) -> None:
    """Format a sample axis and add capture-aware projected region overlays.

    Args:
        axes: Signal axes.
        window: Window metadata in source coordinates.
        count: Represented sample count.
        options: Axis and overlay settings.
        vertical: Whether time increases down the vertical axis.
        frequency_axis: Whether the other axis represents frequency.
    """
    rate = sample_rate(window.signal_metadata)
    captures = tuple(segments(window, count))
    _position_limits(axes, window, count, options, vertical)
    projected = []
    regions = window_regions(window, count)
    for region in regions:
        if not (
            options.captures
            if region.kind == "capture"
            else options.annotations
        ):
            continue
        source = region.bounds["sample"]
        # A region may span captures with different clocks or RF centers.
        # Project each intersection separately while retaining its source key.
        for capture in captures:
            lo, hi = (
                max(source.start, capture.start),
                min(source.stop, capture.stop),
            )
            if lo >= hi:
                continue
            positions = project(window, [lo, hi], capture, options)
            if not np.isfinite(positions).all():
                continue
            bounds = {
                "position": Span(
                    float(positions[0]),
                    float(positions[1]),
                    unit=options.axis_mode,
                )
            }
            frequency = region.bounds.get("frequency")
            if frequency is not None and frequency_axis:
                if (
                    rate is None
                    or frequency.reference == "RF"
                    and capture.frequency is None
                ):
                    continue
                center = (
                    capture.frequency if frequency.reference == "RF" else 0
                )
                # FFT panels show offsets from the capture's center frequency.
                # Convert only the drawn bounds; source JSON stays in RF units.
                bounds["frequency"] = Span(
                    frequency.start - (center or 0),
                    frequency.stop - (center or 0),
                    "Hz",
                    "baseband",
                )
            projected.append(
                Region(
                    region.key,
                    bounds,
                    region.label,
                    region.kind,
                    region.metadata,
                )
            )
    draw_regions(
        axes, projected, orientation="vertical" if vertical else "horizontal"
    )
    axis = axes.yaxis if vertical else axes.xaxis
    label = {
        "samples": "Sample index",
        "elapsed": "Elapsed sample time (s)",
        "timestamp": "Capture timestamp (UTC)",
    }[options.axis_mode]
    if options.axis_mode == "timestamp":
        epoch = timestamp_origin(window)

        _timestamp_ticks(axis, epoch)
    (axes.set_ylabel if vertical else axes.set_xlabel)(label)
    if options.axis_mode != "samples" and (
        rate is None
        or options.axis_mode == "timestamp"
        and any(c.timestamp is None for c in captures)
    ):
        axes.text(
            0.02,
            0.98,
            "Segments with unknown timing are omitted",
            va="top",
            transform=axes.transAxes,
        )


def _timestamp_ticks(axis: Axis, epoch: Fraction) -> None:
    """Format relative timestamp coordinates as UTC labels.

    Args:
        axis: Matplotlib axis receiving the formatter.
        epoch: UTC POSIX origin for the relative coordinates.
    """

    def label_time(value: float, position: float | None = None) -> str:
        """Format timestamp ticks without rounding sample positions.

        Args:
            value: Seconds relative to the selected timestamp origin.
            position: Unused tick index.

        Returns:
            UTC date and time with fractional seconds when needed.
        """
        # Add the relative coordinate to the exact epoch before rounding the
        # label to nanoseconds. Integer division also handles pre-1970 dates
        # and carries into the next second without a floating-point epoch.
        whole, nanos = divmod(
            round((epoch + Fraction(value)) * 1000000000), 1000000000
        )
        date = datetime.fromtimestamp(whole, tz=UTC)
        suffix = f".{nanos:09d}".rstrip("0") if nanos else ""
        return date.strftime("%Y-%m-%d\n%H:%M:%S") + suffix

    axis.set_major_locator(MaxNLocator(nbins=4))
    axis.set_major_formatter(FuncFormatter(label_time))
