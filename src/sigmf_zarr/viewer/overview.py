"""Bounded signal reduction using NumPy, independent of storage and GUI
APIs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from sigmf_zarr.viewer.windows import WindowFunction, fft_window


def summarize_peaks(
    blocks: Iterable[npt.NDArray[Any]], count: int, *, bins: int = 1024
) -> npt.NDArray[np.float64]:
    """Reduce consecutive sample blocks to a bounded magnitude envelope.

    Every finite sample contributes to its bin. Empty or nonfinite bins
    have zero magnitude. No spectral analysis or metadata is required.

    Args:
        blocks: Consecutive numeric vectors of at most 65536 samples.
        count: Exact positive source sample count.
        bins: Maximum number of bins, from 1 through 4096.

    Returns:
        Peak magnitudes in uniformly spaced integer sample bins.

    Raises:
        ValueError: If settings or blocks are invalid or counts disagree.
    """
    if type(count) is not int or not 1 <= count <= np.iinfo(np.int64).max:
        raise ValueError("Count must be a positive int64 sample count")
    if type(bins) is not int or not 1 <= bins <= 4096:
        raise ValueError("Bins must be between 1 and 4096")
    bins = min(bins, count)
    # Plan edges in source coordinates so read-block boundaries do not change
    # bin membership. Python integer arithmetic avoids overflow in i * count.
    edges = np.array([i * count // bins for i in range(bins + 1)])
    peaks = np.zeros(bins, dtype=np.float64)
    offset = 0
    for block in blocks:
        block = np.asarray(block)
        if (
            block.ndim != 1
            or not 1 <= len(block) <= 65536
            or block.dtype.kind not in "iufc"
            or offset + len(block) > count
        ):
            raise ValueError("Invalid sample block")
        values = np.abs(
            block.astype(
                np.complex128 if np.iscomplexobj(block) else np.float64
            )
        )
        values[~np.isfinite(values)] = 0
        # Each block may cover only part of its first and last bins. Merge
        # those partial maxima with earlier blocks instead of replacing them.
        first = int(np.searchsorted(edges, offset, side="right") - 1)
        last = int(
            np.searchsorted(edges, offset + len(block) - 1, side="right")
        )
        for index in range(first, last):
            lo = max(0, int(edges[index]) - offset)
            hi = min(len(block), int(edges[index + 1]) - offset)
            peaks[index] = max(peaks[index], float(values[lo:hi].max()))
        offset += len(block)
    if offset != count:
        raise ValueError("Sample blocks do not match the declared count")
    peaks.setflags(write=False)
    return peaks


@dataclass(frozen=True)
class SignalOverview:
    """Reduced representation of a complete, contiguous signal interval."""

    count: int
    """Number of source samples represented."""

    edges: npt.NDArray[np.int64]
    """Time-bin edges relative to the beginning of the interval."""

    frame_edges: npt.NDArray[np.float64]
    """Spectral time-bin edges around complete frame centers."""

    minimum: npt.NDArray[Any]
    """Per-bin component minima, with I and Q reduced independently."""

    maximum: npt.NDArray[Any]
    """Per-bin component maxima, with I and Q reduced independently."""

    peak: npt.NDArray[np.float64]
    """Per-bin peak magnitudes for timeline drawing."""

    power: npt.NDArray[np.float64]
    """Mean windowed FFT bin power per time bin, before conversion to dB."""

    spectrum: npt.NDArray[np.float64]
    """Mean bin power over all finite complete FFT frames."""

    fft_size: int
    """FFT frame length."""

    hop: int
    """Distance between consecutive FFT frame starts."""

    iq: npt.NDArray[Any]
    """Uniformly selected original samples for an explicitly sampled I/Q
    view.
    """

    mean_power: float
    """Mean squared magnitude over all source samples."""

    peak_magnitude: float
    """Maximum magnitude over all source samples."""

    window_function: WindowFunction = "hann"
    """Periodic FFT window used to calculate the spectral powers."""


def _validate_settings(
    count: int, bins: int, fft_size: int, overlap: float, iq_points: int
) -> None:
    """Validate bounded reduction settings.

    Args:
        count: Source sample count.
        bins: Number of time bins.
        fft_size: Transform length.
        overlap: Fractional overlap.
        iq_points: Retained sample count.

    Raises:
        ValueError: If any setting is invalid.
    """
    for name, value, lower, upper in (
        ("count", count, 2, np.iinfo(np.int64).max),
        ("bins", bins, 1, 1024),
        ("fft_size", fft_size, 2, 8192),
        ("iq_points", iq_points, 1, 65536),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid {name}")
    if not np.isfinite(overlap) or not 0 <= overlap <= 0.875:
        raise ValueError("Overlap must be between zero and 0.875")


def _validate_boundaries(boundaries: tuple[int, ...], count: int) -> None:
    """Validate acquisition boundaries.

    Args:
        boundaries: Interior sample coordinates.
        count: Interval length.

    Raises:
        ValueError: If boundaries are invalid, repeated, or unsorted.
    """
    if (
        any(type(b) is not int or not 0 < b < count for b in boundaries)
        or tuple(sorted(set(boundaries))) != boundaries
    ):
        raise ValueError("Boundaries must be sorted unique interior samples")


def _mask_boundary_bins(
    power: npt.NDArray[np.float64],
    frame_edges: npt.NDArray[np.float64],
    boundaries: tuple[int, ...],
) -> None:
    """Omit waterfall bins that combine different acquisition segments.

    Args:
        power: Averaged spectral bins, modified in place.
        frame_edges: Source coordinates of spectral bin edges.
        boundaries: Acquisition boundary coordinates.
    """
    for boundary in boundaries:
        power[(frame_edges[:-1] < boundary) & (boundary < frame_edges[1:])] = (
            np.nan
        )


def summarize_signal(
    blocks: Iterable[npt.NDArray[Any]],
    count: int,
    *,
    bins: int = 512,
    fft_size: int = 256,
    overlap: float = 0.5,
    iq_points: int = 4096,
    boundaries: tuple[int, ...] = (),
    window_function: WindowFunction = "hann",
) -> SignalOverview:
    """Reduce consecutive blocks without loading the entire signal at once.

    Every sample contributes to the amplitude envelope and scalar metrics.
    FFT frames cross block boundaries without gaps or duplication. Complete
    finite frames contribute to mean linear power, grouped by frame midpoint.
    The final incomplete frame is omitted. I/Q points are uniformly sampled.

    Output bins and retained I/Q positions are planned before reading blocks.
    Each block updates amplitude summaries and completes pending FFT frames.
    A short tail preserves frame continuity across block boundaries. Spectral
    sums and frame counts are converted to averages after the final block.

    Args:
        blocks: Consecutive nonempty numeric vectors, at most 65536
            samples each. All blocks must consistently be real or complex.
        count: Exact total sample count, at least two.
        bins: Maximum number of time bins, from 1 through 1024.
        fft_size: FFT length from 2 through 8192, clipped to count.
        overlap: Fraction of overlap, from zero through 0.875.
        window_function: Periodic FFT window, defaulting to Hann.
        iq_points: Maximum number of I/Q points, from 1 through 65536.
        boundaries: Sorted acquisition boundaries relative to the interval.
            FFT frames crossing a boundary are excluded. Spectral time bins
            spanning boundaries are omitted from the waterfall.

    Returns:
        Read-only reduced arrays and scalar metrics for the complete interval.

    Raises:
        ValueError: If settings, blocks, or the total sample count are invalid.
    """
    _validate_settings(count, bins, fft_size, overlap, iq_points)
    _validate_boundaries(boundaries, count)
    bins, fft_size = min(bins, count), min(fft_size, count)
    # All coordinates are relative to the supplied interval. Integer division
    # keeps sample-bin boundaries exact even for very large sample counts.
    edges = np.array(
        [i * count // bins for i in range(bins + 1)], dtype=np.int64
    )
    hop = max(1, fft_size - int(fft_size * overlap))
    frame_count = (count - fft_size) // hop + 1
    spectral_bins = min(bins, frame_count)
    # Group complete frame ordinals, independently of the amplitude bins.
    # Display edges lie half a hop before the corresponding frame centers.
    frame_bins = np.array(
        [i * frame_count // spectral_bins for i in range(spectral_bins + 1)]
    )
    frame_edges = fft_size / 2 - hop / 2 + frame_bins * hop
    # Plan I/Q decimation globally so changing read-block sizes cannot change
    # the displayed points. Include both endpoints when retaining two or more.
    positions = np.array(
        [
            i * (count - 1) // max(1, min(iq_points, count) - 1)
            for i in range(min(iq_points, count))
        ]
    )
    minimum = np.full((bins, 2), np.inf)
    maximum = np.full((bins, 2), -np.inf)
    peak = np.zeros(bins)
    power = np.zeros((spectral_bins, fft_size))
    frames_per_bin = np.zeros(spectral_bins, dtype=np.int64)
    iq: npt.NDArray[Any] = np.empty(len(positions), dtype=np.complex128)
    taper = fft_window(fft_size, window_function)
    # Streaming invariant: offset is the next block's first sample, while
    # frame_start is the next unprocessed FFT start on the global hop grid.
    # tail holds exactly the samples in [frame_start, offset).
    offset, frame_start = 0, 0
    tail = np.empty(0, dtype=np.complex128)
    total_power = 0.0
    is_complex: bool | None = None
    for block in blocks:
        block = np.asarray(block)
        if (
            block.ndim != 1
            or block.dtype.kind not in "iufc"
            or not len(block)
            or len(block) > 65536
            or offset + len(block) > count
        ):
            raise ValueError("Invalid signal block")
        complex_block = np.iscomplexobj(block)
        if is_complex is not None and complex_block != is_complex:
            raise ValueError(
                "Signal blocks must use consistent real/complex data"
            )
        is_complex = complex_block
        # Promote before magnitude and squaring to avoid integer overflow.
        # Amplitude summaries retain nonfinite input. Only the spectral
        # averages below exclude frames containing nonfinite values.
        block = block.astype(np.complex128 if is_complex else np.float64)
        magnitude = np.abs(block)
        with np.errstate(over="ignore", invalid="ignore"):
            total_power += float(np.sum(magnitude**2))
        # A block can partially cover several bins. Merge every overlapping
        # slice so extrema do not depend on storage or read-block boundaries.
        first = int(np.searchsorted(edges, offset, side="right") - 1)
        last = int(
            np.searchsorted(edges, offset + len(block) - 1, side="right")
        )
        for index in range(first, last):
            lo = max(0, int(edges[index]) - offset)
            hi = min(len(block), int(edges[index + 1]) - offset)
            part = block[lo:hi]
            minimum[index] = np.minimum(
                minimum[index], [part.real.min(), part.imag.min()]
            )
            maximum[index] = np.maximum(
                maximum[index], [part.real.max(), part.imag.max()]
            )
            peak[index] = np.maximum(peak[index], magnitude[lo:hi].max())
        lo, hi = np.searchsorted(positions, [offset, offset + len(block)])
        iq[lo:hi] = block[positions[lo:hi] - offset]
        # Prepending the pending tail lets FFT frames cross read boundaries.
        # FFT work uses one block plus a tail shorter than fft_size. Temporary
        # sizes depend on block and FFT settings, not the recording length.
        joined = np.concatenate((tail, block))
        if len(joined) >= fft_size:
            frames = np.lib.stride_tricks.sliding_window_view(
                joined, fft_size
            )[::hop]
            # Normalize by the window's sum so a bin-centered tone keeps its
            # amplitude across window choices. Accumulate power before any
            # conversion to dB, which would change the meaning of the average.
            with np.errstate(over="ignore", invalid="ignore"):
                transformed = (
                    np.fft.fftshift(np.fft.fft(frames * taper, axis=1), axes=1)
                    / taper.sum()
                )
                powers = np.abs(transformed) ** 2
            valid = np.isfinite(powers).all(axis=1)
            starts = frame_start + np.arange(len(frames)) * hop
            # Reject frames that cross an acquisition boundary. Keep the
            # global hop grid instead of restarting it at each capture.
            # Frames starting at a boundary belong to the new capture.
            ends = np.asarray((*boundaries, count))[
                np.searchsorted(boundaries, starts, side="right")
            ]
            valid &= starts + fft_size <= ends
            ordinals = frame_start // hop + np.arange(len(frames))
            rows = np.searchsorted(frame_bins, ordinals, side="right") - 1
            # Several frames can map to the same row. add.at accumulates all
            # repeated indices, unlike buffered advanced-index assignment.
            np.add.at(power, rows[valid], powers[valid])
            np.add.at(frames_per_bin, rows[valid], 1)
            # Advance past every examined start, including rejected frames.
            # Keep overlap samples needed by the next frame. Copying gives
            # the retained tail its own bounded allocation.
            consumed = len(frames) * hop
            frame_start += consumed
            tail = joined[consumed:].copy()
        else:
            tail = joined
        offset += len(block)
    if offset != count:
        raise ValueError(
            "Signal blocks do not match the declared sample count"
        )
    # Average the accumulated sums by frame count before averaging rows.
    # Averaging row means here would give sparsely populated bins too much
    # weight, particularly near rejected frames and acquisition boundaries.
    spectrum = power.sum(axis=0) / max(1, int(frames_per_bin.sum()))
    if not frames_per_bin.any():
        spectrum[:] = np.nan
    power /= np.maximum(1, frames_per_bin[:, None])
    # Empty bins are missing data, not zero-power observations. Mask coarse
    # bins spanning captures only after computing the full-interval spectrum,
    # which must still include their individually valid FFT frames.
    power[frames_per_bin == 0] = np.nan
    _mask_boundary_bins(power, frame_edges, boundaries)
    lower = minimum[:, 0] + 1j * minimum[:, 1] if is_complex else minimum[:, 0]
    upper = maximum[:, 0] + 1j * maximum[:, 1] if is_complex else maximum[:, 0]
    if not is_complex:
        iq = iq.real.copy()
    # Caches and GUI consumers share these arrays. Prevent rendering code from
    # mutating a summary that another view may reuse later.
    for array in (edges, frame_edges, lower, upper, peak, power, spectrum, iq):
        array.setflags(write=False)
    return SignalOverview(
        count,
        edges,
        frame_edges,
        lower,
        upper,
        peak,
        power,
        spectrum,
        fft_size,
        hop,
        iq,
        total_power / count,
        float(peak.max()),
        window_function,
    )
