"""Matplotlib drawing functions and NumPy metrics for selected windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sigmf_zarr.viewer.data import SampleWindow
from sigmf_zarr.viewer.presentation import (
    PlotOptions,
    decorate,
    project,
    segments,
)

if TYPE_CHECKING:
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
