"""Matplotlib drawing functions and NumPy metrics for selected windows."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sigmf_zarr.viewer.data import OverviewWindow, SampleWindow
from sigmf_zarr.viewer.overview import summarize_signal
from sigmf_zarr.viewer.presentation import (
    PlotOptions,
    decorate,
    project,
    segments,
)
from sigmf_zarr.viewer.windows import WINDOW_LABELS, WindowFunction, fft_window

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure


def _sample_rate(window: SampleWindow) -> float | None:
    """Read a valid sample rate from the selected metadata.

    Args:
        window: Selected source window.

    Returns:
        Sample rate in Hz, or None when unavailable or invalid.
    """
    metadata = window.metadata.get("global", {})
    rate = (
        metadata.get("core:sample_rate")
        if isinstance(metadata, dict)
        else None
    )
    if (
        isinstance(rate, int | float)
        and not isinstance(rate, bool)
        and 0 < rate <= 1e12
    ):
        return float(rate)
    return None


def waveform(
    figure: Figure, window: SampleWindow, *, options: PlotOptions | None = None
) -> None:
    """Draw real and imaginary components against time or sample positions.

    Args:
        figure: Figure cleared by its owner before drawing.
        window: Selected source window.
        options: Sample-axis and region visibility settings.
    """
    options = options or PlotOptions()
    axes = figure.subplots()
    # Separate artists prevent a line from connecting discontinuous captures,
    # including captures whose timestamps overlap or run backward.
    for index, segment in enumerate(segments(window, len(window.samples))):
        positions = project(
            window, np.arange(segment.start, segment.stop), segment, options
        )
        if not np.isfinite(positions).all():
            continue
        samples = window.samples[
            segment.start - window.start : segment.stop - window.start
        ]
        axes.plot(
            positions,
            samples.real,
            color="tab:blue",
            label=("I" if np.iscomplexobj(samples) else "Amplitude")
            if index == 0
            else None,
        )
        if np.iscomplexobj(samples):
            axes.plot(
                positions,
                samples.imag,
                color="tab:orange",
                label="Q" if index == 0 else None,
            )
    axes.set_ylabel("Stored amplitude")
    if axes.lines:
        axes.legend()
    axes.grid(True, alpha=0.25)
    decorate(axes, window, len(window.samples), options)


def constellation(figure: Figure, window: SampleWindow) -> None:
    """Draw complex samples without symbol timing or amplitude correction.

    Args:
        figure: Figure cleared by its owner before drawing.
        window: Selected source window.
    """
    axes = figure.subplots()
    axes.set_box_aspect(1)
    axes.set_aspect("equal", adjustable="datalim")
    if not np.iscomplexobj(window.samples):
        axes.text(
            0.5,
            0.5,
            "I/Q view requires complex or I/Q samples",
            ha="center",
            transform=axes.transAxes,
        )
        return
    axes.scatter(window.samples.real, window.samples.imag, s=4, alpha=0.5)
    axes.set(xlabel="I", ylabel="Q")
    axes.grid(True, alpha=0.25)


def spectrum(
    figure: Figure,
    window: SampleWindow,
    *,
    window_function: WindowFunction = "hann",
) -> None:
    """Draw two-sided windowed DFT bin power in stored units.

    Args:
        figure: Figure cleared by its owner before drawing.
        window: Selected source window.
        window_function: Periodic FFT window, defaulting to Hann.
    """
    axes = figure.subplots()
    samples = window.samples
    captures = tuple(segments(window, len(samples)))
    if len(captures) > 1:
        # A single DFT would treat concatenated acquisitions as continuous.
        # Average powers of frames that stay within capture boundaries instead.
        summary = summarize_signal(
            (
                samples[start : start + 65536]
                for start in range(0, len(samples), 65536)
            ),
            len(samples),
            fft_size=min(256, len(samples)),
            boundaries=tuple(c.start - window.start for c in captures[1:]),
            window_function=window_function,
        )
        power = summary.spectrum
        frequency = np.fft.fftshift(np.fft.fftfreq(summary.fft_size))
        axes.set_title(
            f"Mean capture {WINDOW_LABELS[window_function]}-window bin power"
        )
    else:
        taper = fft_window(len(samples), window_function)
        # Coherent-gain normalization preserves a bin-centered tone's amplitude
        # across window choices. This is bin power, not power density per Hz.
        power = (
            np.abs(np.fft.fftshift(np.fft.fft(samples * taper)) / taper.sum())
            ** 2
        )
        axes.set_title(f"{WINDOW_LABELS[window_function]} window")
        frequency = np.fft.fftshift(np.fft.fftfreq(len(samples)))
    sample_rate = _sample_rate(window)
    if sample_rate is not None:
        frequency *= sample_rate
    axes.plot(frequency, 10 * np.log10(np.maximum(power, 1e-30)))
    axes.set(
        xlabel=(
            "Frequency offset (Hz)"
            if sample_rate is not None
            else "Frequency (cycles/sample)"
        ),
        ylabel="Bin power (dB re 1 stored unit squared)",
    )
    axes.grid(True, alpha=0.25)


def mean_power(window: SampleWindow) -> float:
    """Calculate mean squared magnitude in stored amplitude units squared.

    Args:
        window: Selected source window.

    Returns:
        Mean squared magnitude without physical power calibration.
    """
    return float(np.mean(np.abs(window.samples) ** 2))


def rms(window: SampleWindow) -> float:
    """Calculate root mean square magnitude in stored amplitude units.

    Args:
        window: Selected source window.

    Returns:
        Root mean square magnitude.
    """
    return float(np.sqrt(mean_power(window)))


def peak_magnitude(window: SampleWindow) -> float:
    """Calculate peak magnitude in stored amplitude units.

    Args:
        window: Selected source window.

    Returns:
        Maximum sample magnitude.
    """
    return float(np.max(np.abs(window.samples)))


def _spectrogram_rows(
    window: SampleWindow,
    length: int,
    hop: int,
    options: PlotOptions,
    window_function: WindowFunction,
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Compute independent capture frames and projected display bounds.

    Args:
        window: Selected source samples and metadata.
        length: FFT length.
        hop: Distance between frames in samples.
        options: Coordinate settings independent of frame selection.
        window_function: Periodic FFT window.

    Returns:
        Projected bounds and dB powers for each capture with complete frames.
    """
    samples = window.samples
    taper = fft_window(length, window_function)
    rows = []
    for segment in segments(window, len(samples)):
        part = samples[
            segment.start - window.start : segment.stop - window.start
        ]
        if len(part) < length:
            continue
        frames = np.lib.stride_tricks.sliding_window_view(part, length)[::hop]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = (
                np.fft.fftshift(np.fft.fft(frames * taper, axis=1), axes=1)
                / taper.sum()
            )
            power_db = 10 * np.log10(
                np.maximum(np.abs(transformed) ** 2, 1e-30)
            )
        # Image columns represent frame centers, with half a hop on either
        # side. FFT support can overlap; display cells themselves do not.
        first = segment.start + length / 2 - hop / 2
        times = project(
            window, [first, first + len(frames) * hop], segment, options
        )
        if np.isfinite(times).all():
            rows.append((times, power_db))
    return rows


def spectrogram(
    figure: Figure,
    window: SampleWindow,
    *,
    fft_size: int | None = None,
    window_function: WindowFunction = "hann",
    overlap: float = 0.5,
    dynamic_range: float = 80.0,
    maximum_db: float | None = None,
    cmap: str | Colormap = "magma",
    options: PlotOptions | None = None,
) -> None:
    """Draw a two-sided windowed spectrogram with time on the x axis.

    Args:
        figure: Figure cleared by its owner before drawing.
        window: Selected source window. No additional samples are read.
        options: Sample-axis and region visibility settings.
        fft_size: FFT length, clipped to the available samples. None chooses
            a power of two up to 256, targeting at least four complete frames.
        overlap: Fraction of overlap between frames, from zero through 0.875.
        window_function: Periodic FFT window, defaulting to Hann.
        dynamic_range: Positive color range below the maximum, in dB.
        maximum_db: Color maximum in dB, or None for the window's peak power.
        cmap: Registered Matplotlib colormap name or colormap instance.

    Raises:
        ValueError: If FFT or color-scale settings are invalid.
    """
    if fft_size is not None and (type(fft_size) is not int or fft_size < 2):
        raise ValueError("FFT size must be an integer of at least two")
    if not np.isfinite(overlap) or not 0 <= overlap <= 0.875:
        raise ValueError("Overlap must be between zero and 0.875")
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        raise ValueError("Dynamic range must be positive and finite")
    if maximum_db is not None and not np.isfinite(maximum_db):
        raise ValueError("Color maximum must be finite")
    axes = figure.subplots()
    samples = window.samples
    if len(samples) < 2:
        axes.text(
            0.5,
            0.5,
            "Spectrogram requires at least two samples",
            ha="center",
            transform=axes.transAxes,
        )
        return
    if fft_size is None:
        fft_size = 2 ** int(np.log2(min(256, max(2, len(samples) // 4))))
    length = min(fft_size, len(samples))
    hop = max(1, length - int(length * overlap))
    options = options or PlotOptions()
    rate = _sample_rate(window)
    scale = rate if rate is not None else 1.0
    frequencies = np.fft.fftshift(np.fft.fftfreq(length)) * scale
    spacing = scale / length
    rows = _spectrogram_rows(window, length, hop, options, window_function)
    finite = (
        np.concatenate([power[np.isfinite(power)] for _, power in rows])
        if rows
        else np.empty(0)
    )
    if finite.size:
        peak = float(finite.max())
        # The logarithmic floor is -300 dB. Anchor an all-zero view at 0 dB
        # so auto-level does not render numerical silence as maximum power.
        top = (
            (0.0 if peak <= -300 else peak)
            if maximum_db is None
            else maximum_db
        )
        for times, power_db in rows:
            # Frame powers are stored as (time, frequency). Transpose for image
            # rows to represent frequency and columns to represent time.
            image = axes.imshow(
                np.ma.masked_invalid(power_db.T),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                extent=(
                    times[0],
                    times[1],
                    frequencies[0] - spacing / 2,
                    frequencies[-1] + spacing / 2,
                ),
                vmin=top - dynamic_range,
                vmax=top,
            )
        axes.set_xlim(
            min(times[0] for times, _ in rows),
            max(times[1] for times, _ in rows),
        )
        figure.colorbar(
            image, ax=axes, label="Bin power (dB re 1 stored unit squared)"
        )
    else:
        axes.text(
            0.5,
            0.5,
            "No finite spectrogram frames",
            ha="center",
            transform=axes.transAxes,
        )
    axes.set(
        ylabel="Frequency offset (Hz)"
        if rate is not None
        else "Frequency (cycles/sample)",
        title=(
            f"{WINDOW_LABELS[window_function]} window, FFT {length}, "
            f"hop {hop} samples"
        ),
    )
    decorate(axes, window, len(samples), options, frequency_axis=True)


def overview_waveform(
    figure: Figure,
    window: OverviewWindow,
    *,
    options: PlotOptions | None = None,
) -> None:
    """Draw component extrema covering every sample in a reduced interval.

    Args:
        figure: Figure cleared by its owner.
        window: Reduced interval and metadata.
        options: Sample-axis and region visibility settings.
    """
    options = options or PlotOptions()
    signal, context = window.signal, window.context
    edges = context.start + signal.edges
    axes = figure.subplots()
    for segment in segments(context, signal.count):
        # A reduced bin cannot be split back into its source samples. Omit bins
        # straddling captures instead of attributing their extrema to one side.
        indexes = np.flatnonzero(
            (edges[:-1] >= segment.start) & (edges[1:] <= segment.stop)
        )
        if not len(indexes):
            continue
        positions = project(
            context, edges[np.r_[indexes, indexes[-1] + 1]], segment, options
        )
        if not np.isfinite(positions).all():
            continue
        for label, lower, upper, color in (
            (
                "I" if np.iscomplexobj(signal.minimum) else "Amplitude",
                signal.minimum.real,
                signal.maximum.real,
                "tab:blue",
            ),
            ("Q", signal.minimum.imag, signal.maximum.imag, "tab:orange"),
        ):
            if label == "Q" and not np.iscomplexobj(signal.minimum):
                continue
            axes.fill_between(
                positions,
                np.r_[lower[indexes], lower[indexes[-1]]],
                np.r_[upper[indexes], upper[indexes[-1]]],
                step="post",
                alpha=0.5,
                label=label,
                color=color,
            )
    axes.set(ylabel="Stored amplitude", title="Minimum/maximum envelope")
    if axes.collections:
        handles, labels = axes.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        axes.legend(unique.values(), unique.keys())
    decorate(axes, context, signal.count, options)


def overview_constellation(figure: Figure, window: OverviewWindow) -> None:
    """Draw uniformly sampled I/Q points without implying symbol recovery.

    Args:
        figure: Figure cleared by its owner.
        window: Reduced interval and metadata.
    """
    constellation(figure, replace(window.context, samples=window.signal.iq))
    figure.axes[0].set_title(
        f"{len(window.signal.iq):,} uniformly sampled points"
    )


def overview_spectrum(figure: Figure, window: OverviewWindow) -> None:
    """Draw mean linear bin power over all finite complete FFT frames.

    Args:
        figure: Figure cleared by its owner.
        window: Reduced interval and metadata.
    """
    rate = _sample_rate(window.context)
    frequency = np.fft.fftshift(np.fft.fftfreq(window.signal.fft_size)) * (
        rate if rate is not None else 1
    )
    axes = figure.subplots()
    axes.plot(
        frequency, 10 * np.log10(np.maximum(window.signal.spectrum, 1e-30))
    )
    axes.set(
        xlabel="Frequency offset (Hz)"
        if rate is not None
        else "Frequency (cycles/sample)",
        ylabel="Bin power (dB re 1 stored unit squared)",
        title=(
            f"Mean {WINDOW_LABELS[window.signal.window_function]}-window "
            f"bin power, FFT {window.signal.fft_size}"
        ),
    )
    axes.grid(True, alpha=0.25)


def overview_spectrogram(
    figure: Figure,
    window: OverviewWindow,
    *,
    dynamic_range: float = 80.0,
    maximum_db: float | None = None,
    cmap: str | Colormap = "magma",
    options: PlotOptions | None = None,
) -> None:
    """Draw mean linear frame power grouped into bounded time bins.

    Args:
        figure: Figure cleared by its owner.
        window: Reduced interval and metadata.
        options: Sample-axis and region visibility settings.
        dynamic_range: Positive color range below the maximum, in dB.
        maximum_db: Fixed color maximum, or None for automatic scaling.
        cmap: Registered Matplotlib colormap name or colormap instance.

    Raises:
        ValueError: If color-scale settings are invalid.
    """
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        raise ValueError("Dynamic range must be positive and finite")
    if maximum_db is not None and not np.isfinite(maximum_db):
        raise ValueError("Color maximum must be finite")
    signal, context = window.signal, window.context
    axes = figure.subplots()
    power = 10 * np.log10(np.maximum(signal.power, 1e-30))
    finite = power[np.isfinite(power)]
    if not finite.size:
        axes.text(
            0.5,
            0.5,
            "No finite spectrogram frames",
            ha="center",
            transform=axes.transAxes,
        )
        return
    peak = float(finite.max())
    top = (0.0 if peak <= -300 else peak) if maximum_db is None else maximum_db
    rate = _sample_rate(context)
    scale = rate if rate is not None else 1.0
    frequencies = np.fft.fftshift(np.fft.fftfreq(signal.fft_size)) * scale
    spacing = scale / signal.fft_size
    edges = np.r_[frequencies - spacing / 2, frequencies[-1] + spacing / 2]
    options = options or PlotOptions()
    source_edges = context.start + signal.frame_edges
    images = []
    limits: list[float] = []
    for segment in segments(context, signal.count):
        indexes = np.flatnonzero(
            (source_edges[:-1] >= segment.start)
            & (source_edges[1:] <= segment.stop)
        )
        if not len(indexes):
            continue
        times = project(
            context,
            source_edges[np.r_[indexes, indexes[-1] + 1]],
            segment,
            options,
        )
        if not np.isfinite(times).all():
            continue
        images.append(
            axes.pcolormesh(
                times,
                edges,
                np.ma.masked_invalid(power[indexes].T),
                cmap=cmap,
                vmin=top - dynamic_range,
                vmax=top,
                rasterized=True,
            )
        )
        limits.extend(times)
    if images:
        axes.set_xlim(min(limits), max(limits))
        figure.colorbar(
            images[0], ax=axes, label="Bin power (dB re 1 stored unit squared)"
        )
    axes.set(
        ylabel="Frequency offset (Hz)"
        if rate is not None
        else "Frequency (cycles/sample)",
        title=(
            f"Mean {WINDOW_LABELS[signal.window_function]}-window "
            f"power per time bin, FFT {signal.fft_size}"
        ),
    )
    decorate(axes, context, signal.count, options, frequency_axis=True)
