"""Capture-aware plotting coordinates and region projection."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from fractions import Fraction
from math import ceil, floor, isfinite
from typing import cast

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

MAX_DRAWN_REGIONS = 500
"""Maximum number of region overlays drawn by one plot.

Plots draw the regions intersecting the displayed samples in order of their
first sample. A region split at capture boundaries uses one overlay for each
capture. The plot reports how many regions it omits.
"""


@dataclass
class PlotOptions:
    """Shared display settings independent of signal processing."""

    axis_mode: AxisMode = "samples"
    """Coordinate system used for the sample axis."""

    captures: bool = True
    """Whether capture overlays are visible."""

    annotations: bool = True
    """Whether annotation overlays are visible."""


@dataclass
class _Layout:
    """Coordinates derived from one window and shared by its plots."""

    metadata: Mapping[str, object]
    """Resolved signal metadata."""

    rate: float | None
    """Sample rate, if known."""

    epoch: Fraction
    """Shared timestamp origin."""

    segments: dict[int, tuple[CaptureSegment, ...]] = field(
        default_factory=dict
    )
    """Capture segments by stored signal length."""

    regions: dict[int, tuple[Region, ...]] = field(default_factory=dict)
    """Source regions in metadata order by represented sample count."""

    ordered: dict[int, tuple[list[int | float], tuple[Region, ...]]] = field(
        default_factory=dict
    )
    """Region starts and regions sorted by start, by represented count."""


_layouts: ContextVar[dict[int, tuple[SampleWindow, _Layout]] | None] = (
    ContextVar("_layouts", default=None)
)
"""Layouts shared within the active `shared_layouts()` context."""


class PreparedLayouts:
    """Window coordinates derived ahead of drawing, usually in a worker."""

    _entries: dict[int, tuple[SampleWindow, _Layout]]
    """Layouts by window identity. Each entry holds its window."""

    def __init__(
        self, entries: Mapping[int, tuple[SampleWindow, object]]
    ) -> None:
        """Wrap derived layouts.

        Args:
            entries: Layouts by window identity, as `prepare_layouts()`
                derives them.
        """
        # The private layout type stays out of the public signature.
        self._entries = cast(
            dict[int, tuple[SampleWindow, _Layout]], dict(entries)
        )


def ordered_regions(window: SampleWindow, count: int) -> tuple[Region, ...]:
    """Return a window's regions in order of their first sample.

    Within `shared_layouts()`, later calls for the same window reuse the
    result.

    Args:
        window: Original scopes and resolved metadata.
        count: Number of displayed samples when full length is unavailable.

    Returns:
        Regions in a stable order by first sample.
    """
    return _ordered_regions(window, count)[1]


def prepare_layouts(window: SampleWindow, count: int) -> PreparedLayouts:
    """Derive capture segments and ordered regions for one window.

    Call this in a worker. Drawing within `shared_layouts(prepared)` then
    reuses the result instead of deriving regions on the GUI thread.

    Args:
        window: Window to display. It must not change afterward.
        count: Number of represented samples.

    Returns:
        Derived layouts for the window.
    """
    token = _layouts.set({})
    try:
        _ordered_regions(window, count)
        return PreparedLayouts(dict(_layouts.get() or {}))
    finally:
        _layouts.reset(token)


_pinned: ContextVar[PreparedLayouts | None] = ContextVar(
    "_pinned", default=None
)
"""Layouts that drawing in this thread reuses outside any context."""


def pin_layouts(prepared: PreparedLayouts | None) -> None:
    """Reuse prepared layouts for later drawing in the calling thread.

    Redraws after display-setting changes then reuse the regions derived in
    a worker. A layout is used only for the exact window object it was
    derived from.

    Args:
        prepared: Layouts to reuse, or None to stop reusing them.
    """
    _pinned.set(prepared)


@contextmanager
def shared_layouts(prepared: PreparedLayouts | None = None) -> Iterator[None]:
    """Derive window coordinates once until the outermost context exits.

    Timestamps, capture segments, and regions are derived from metadata once
    for each window object, instead of for every projected point or plot.
    Windows must not change while the context is active.

    Args:
        prepared: Layouts from `prepare_layouts()` to reuse.

    Yields:
        None.
    """
    active = _layouts.get()
    if active is not None:
        if prepared is not None:
            active.update(prepared._entries)
        yield
        return
    token = _layouts.set(
        dict(prepared._entries) if prepared is not None else {}
    )
    try:
        yield
    finally:
        _layouts.reset(token)


def _layout(window: SampleWindow) -> _Layout:
    """Return the window's derived coordinates.

    Args:
        window: Displayed window.

    Returns:
        Layout shared within the active context, or a new layout.
    """
    cache = _layouts.get()
    # The cache holds each window, so an identifier cannot be reused while
    # the context is active.
    cached = None if cache is None else cache.get(id(window))
    if cached is not None:
        return cached[1]
    pinned = _pinned.get()
    if pinned is not None:
        cached = pinned._entries.get(id(window))
        # A pinned entry holds its window, but check identity anyway: the
        # identifier of an unrelated window must not select its layout.
        if cached is not None and cached[0] is window:
            return cached[1]
    metadata = window.signal_metadata
    length = window.total_samples or window.start + max(1, len(window.samples))
    segments = capture_segments(metadata, length)
    layout = _Layout(
        metadata,
        sample_rate(metadata),
        min(
            (c.timestamp for c in segments if c.timestamp is not None),
            default=Fraction(0),
        ),
        {length: segments},
    )
    if cache is not None:
        cache[id(window)] = (window, layout)
    return layout


def _segments(window: SampleWindow, length: int) -> tuple[CaptureSegment, ...]:
    """Return the cached capture segments for a stored signal length.

    Args:
        window: Displayed window.
        length: Stored signal length.

    Returns:
        Ordered segments covering the stored signal.
    """
    layout = _layout(window)
    if length not in layout.segments:
        layout.segments[length] = capture_segments(layout.metadata, length)
    return layout.segments[length]


@shared_layouts()
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
    layout = _layout(window)
    rate = layout.rate
    if options.axis_mode != "samples" and rate is None:
        return None
    scale = rate or 1.0
    intervals = []
    for capture in _segments(window, length):
        if options.axis_mode == "samples":
            start, stop = lo, hi
        elif options.axis_mode == "elapsed":
            start, stop = lo * scale, hi * scale
        else:
            if capture.timestamp is None:
                continue
            origin = float(capture.timestamp - layout.epoch)
            start = capture.origin + (lo - origin) * scale
            stop = capture.origin + (hi - origin) * scale
        # Intersect with this capture, then round outward to contain
        # fractional plot limits. Timestamp gaps must not create synthetic
        # sample ranges. Clamping first also bounds positions that overflow
        # to infinity at extreme sample rates.
        start = floor(min(max(start, capture.start), capture.stop))
        stop = ceil(min(max(stop, capture.start), capture.stop))
        if start < stop:
            intervals.append((start, stop))
    if not intervals:
        return None
    return min(s for s, _ in intervals), max(e for _, e in intervals)


def window_regions(window: SampleWindow, count: int) -> tuple[Region, ...]:
    """Adapt regions while preserving recording and item source scopes.

    Within `shared_layouts()`, later calls for the same window reuse the
    result.

    Args:
        window: Original scopes and resolved metadata.
        count: Number of displayed samples when full length is unavailable.

    Returns:
        Regions referencing the original recording or item entry.
    """
    layout = _layout(window)
    if count in layout.regions:
        return layout.regions[count]
    length = window.total_samples or window.start + count
    regions = sigmf_regions(
        layout.metadata,
        length,
        source_metadata=window.recording_metadata or window.metadata,
        capture_item=window.item if window.recording_metadata else None,
    )
    if window.item_metadata is not None:
        regions += sigmf_regions(
            layout.metadata,
            length,
            scope="item",
            source_metadata=window.item_metadata,
        )
    layout.regions[count] = regions
    return regions


def _ordered_regions(
    window: SampleWindow, count: int
) -> tuple[list[int | float], tuple[Region, ...]]:
    """Sort regions by first sample once for all plots of a window.

    Args:
        window: Original scopes and resolved metadata.
        count: Number of displayed samples when full length is unavailable.

    Returns:
        First samples and regions in the same stable order.
    """
    layout = _layout(window)
    if count not in layout.ordered:
        regions = tuple(
            sorted(
                window_regions(window, count),
                key=lambda region: region.bounds["sample"].start,
            )
        )
        layout.ordered[count] = (
            [region.bounds["sample"].start for region in regions],
            regions,
        )
    return layout.ordered[count]


def segments(window: SampleWindow, count: int) -> Iterator[CaptureSegment]:
    """Yield captures intersecting a displayed window.

    Args:
        window: Source samples and metadata.
        count: Number of represented samples.

    Yields:
        Clipped segments retaining their timestamp origins.
    """
    end = window.start + count
    for segment in _segments(window, window.total_samples or end):
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
    return _layout(window).epoch


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
    layout = _layout(window)
    return project_samples(
        positions,
        segment,
        layout.rate,
        options.axis_mode,
        epoch=layout.epoch if options.axis_mode == "timestamp" else 0,
    )


def _position_limits(
    axes: Axes,
    window: SampleWindow,
    captures: tuple[CaptureSegment, ...],
    options: PlotOptions,
    vertical: bool,
) -> None:
    """Set the position axis to the represented sample interval.

    Args:
        axes: Target plot.
        window: Signal metadata.
        captures: Captures clipped to the represented samples.
        options: Coordinate mapping settings.
        vertical: Whether position is the vertical coordinate.
    """
    extents = [
        project(window, [c.start, c.stop], c, options) for c in captures
    ]
    finite = [v for v in extents if np.isfinite(v).all()]
    if finite:
        lo, hi = min(v[0] for v in finite), max(v[1] for v in finite)
        # A short interval far from the timestamp origin can round to a
        # single coordinate. Keep the automatic limits in that case.
        if not lo < hi:
            return
        if vertical:
            axes.set_ylim(hi, lo)
        else:
            axes.set_xlim(lo, hi)


def _project_piece(
    window: SampleWindow,
    region: Region,
    capture: CaptureSegment,
    options: PlotOptions,
    frequency_axis: bool,
) -> Region | None:
    """Project the part of a region within one capture onto plot axes.

    Args:
        window: Window metadata in source coordinates.
        region: Source region with a sample bound.
        capture: Capture clipped to the represented samples.
        options: Axis settings.
        frequency_axis: Whether the other axis represents frequency.

    Returns:
        Projected overlay, or None when it has no drawable extent.
    """
    source = region.bounds["sample"]
    lo, hi = max(source.start, capture.start), min(source.stop, capture.stop)
    if lo >= hi:
        return None
    positions = project(window, [lo, hi], capture, options)
    # A short interval far from the timestamp origin can round to an empty
    # floating-point span. Omit it rather than fail the whole plot.
    if not np.isfinite(positions).all() or not positions[0] < positions[1]:
        return None
    bounds = {
        "position": Span(
            float(positions[0]), float(positions[1]), unit=options.axis_mode
        )
    }
    frequency = region.bounds.get("frequency")
    if frequency is not None and frequency_axis:
        if _layout(window).rate is None or (
            frequency.reference == "RF" and capture.frequency is None
        ):
            return None
        # FFT panels show offsets from the capture's center frequency.
        # Convert only the drawn bounds. Source JSON stays in RF units.
        center = (
            (capture.frequency or 0.0) if frequency.reference == "RF" else 0
        )
        low, high = frequency.start - center, frequency.stop - center
        if not (isfinite(low) and isfinite(high) and low < high):
            return None
        bounds["frequency"] = Span(low, high, "Hz", "baseband")
    return Region(
        region.key, bounds, region.label, region.kind, region.metadata
    )


def _projected_regions(
    window: SampleWindow,
    count: int,
    captures: tuple[CaptureSegment, ...],
    options: PlotOptions,
    frequency_axis: bool,
) -> tuple[list[Region], int]:
    """Project enabled regions intersecting the represented samples.

    Regions are visited in order of their first sample. At most
    `MAX_DRAWN_REGIONS` capture intersections are projected.

    Args:
        window: Window metadata in source coordinates.
        count: Represented sample count.
        captures: Captures clipped to the represented samples.
        options: Axis and overlay settings.
        frequency_axis: Whether the other axis represents frequency.

    Returns:
        Projected overlays and the number of regions omitted by the limit.
    """
    starts = [capture.start for capture in captures]
    keys, ordered = _ordered_regions(window, count)
    projected = []
    remaining = MAX_DRAWN_REGIONS
    omitted = 0
    for region in ordered[: bisect_left(keys, window.start + count)]:
        source = region.bounds["sample"]
        if source.stop <= window.start or not (
            options.captures
            if region.kind == "capture"
            else options.annotations
        ):
            continue
        # A region may span captures with different clocks or RF centers.
        # Project each intersection separately while retaining its source key.
        first = max(0, bisect_right(starts, source.start) - 1)
        for capture in captures[first:]:
            if capture.start >= source.stop:
                break
            if not remaining:
                omitted += 1
                break
            remaining -= 1
            piece = _project_piece(
                window, region, capture, options, frequency_axis
            )
            if piece is not None:
                projected.append(piece)
    return projected, omitted


@shared_layouts()
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
    layout = _layout(window)
    captures = tuple(segments(window, count))
    _position_limits(axes, window, captures, options, vertical)
    projected, omitted = _projected_regions(
        window, count, captures, options, frequency_axis
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
        _timestamp_ticks(axis, layout.epoch)
    (axes.set_ylabel if vertical else axes.set_xlabel)(label)
    if options.axis_mode != "samples" and (
        layout.rate is None
        or (
            options.axis_mode == "timestamp"
            and any(c.timestamp is None for c in captures)
        )
    ):
        axes.text(
            0.02,
            0.98,
            "Segments with unknown timing are omitted",
            va="top",
            transform=axes.transAxes,
        )
    if omitted:
        axes.text(
            0.02,
            0.02,
            f"Regions not drawn: {omitted:,}. Zoom in to show them.",
            va="bottom",
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
            UTC date and time with fractional seconds when needed, or POSIX
            seconds when the time is outside the representable date range.
        """
        # Add the relative coordinate to the exact epoch before rounding the
        # label to nanoseconds. Integer division also handles pre-1970 dates
        # and carries into the next second without a floating-point epoch.
        if not isfinite(value):
            return ""
        seconds = epoch + Fraction(value)
        whole, nanos = divmod(round(seconds * 1000000000), 1000000000)
        try:
            date = datetime.fromtimestamp(whole, tz=UTC)
        except (OverflowError, OSError, ValueError):
            # Dates support only years 1 through 9999, and the platform
            # time_t can be narrower. Extreme sample rates can reach either.
            return f"{float(seconds):.12g} s"
        suffix = f".{nanos:09d}".rstrip("0") if nanos else ""
        return date.strftime("%Y-%m-%d\n%H:%M:%S") + suffix

    axis.set_major_locator(MaxNLocator(nbins=4))
    axis.set_major_formatter(FuncFormatter(label_time))
