"""Plot scaling, coordinates, and short-window behavior without Qt."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sigmf_zarr.viewer.data import SampleWindow
from sigmf_zarr.viewer.plots import (
    constellation,
    spectrogram,
    spectrum,
    waveform,
)
from sigmf_zarr.viewer.windows import WindowFunction


@pytest.mark.parametrize(
    "sample_rate", [None, 8000, 0, -1, True, float("nan")]
)
def test_time_and_frequency_units(sample_rate: Any) -> None:
    """Time and frequency views should use the same explicit sample rate.

    Args:
        sample_rate: Valid, absent, or invalid metadata sample rate.
    """
    samples = np.exp(2j * np.pi * np.arange(32) / 8)
    window = SampleWindow(
        "rec",
        None,
        1024,
        {},
        samples,
        {"global": {"core:sample_rate": sample_rate}},
        {},
    )
    physical = sample_rate == 8000
    rate = 8000 if physical else 1
    figure = Figure()
    waveform(figure, window)
    axes = figure.axes[0]
    np.testing.assert_allclose(axes.lines[0].get_xdata(), 1024 + np.arange(32))
    assert axes.get_xlabel() == "Sample index"
    figure.clear()
    spectrum(figure, window)
    axes = figure.axes[0]
    line = axes.lines[0]
    assert line.get_xdata()[np.argmax(line.get_ydata())] == rate / 8
    assert axes.get_xlabel() == (
        "Frequency offset (Hz)" if physical else "Frequency (cycles/sample)"
    )


def test_iq_square_and_equal_units() -> None:
    """I/Q must keep a square box and equal units for unequal sample spans."""
    window = SampleWindow(
        "rec", None, 0, {}, np.array([-10 - 1j, 10 + 1j]), {}, {}
    )
    figure = Figure(figsize=(10, 4), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    constellation(figure, window)
    axes = figure.axes[0]
    for width, height in [(10, 4), (4, 10)]:
        figure.set_size_inches(width, height)
        axes.set_xlim(-5, 5)
        axes.set_ylim(-1, 1)
        canvas.draw()
        bounds = axes.get_window_extent()
        assert bounds.width == pytest.approx(bounds.height)
        points = axes.transData.transform([(0, 0), (1, 0), (0, 1)])
        assert points[1, 0] - points[0, 0] == pytest.approx(
            points[2, 1] - points[0, 1]
        )


@pytest.mark.parametrize("sample_rate", [None, 8000])
@pytest.mark.parametrize(
    "window_function",
    ["hann", "hamming", "blackman", "blackmanharris", "rectangular"],
)
def test_spectrogram_tone_changes(
    sample_rate: int | None, window_function: WindowFunction
) -> None:
    """Two signed tones should occupy the correct bins, times, and powers.

    Args:
        sample_rate: Optional explicit sample rate.
        window_function: Window whose coherent tone gain must be preserved.
    """
    time = np.arange(128)
    samples = np.concatenate(
        (
            2 * np.exp(2j * np.pi * time / 8),
            np.exp(-2j * np.pi * time / 4),
        )
    )
    metadata = (
        {}
        if sample_rate is None
        else {"global": {"core:sample_rate": sample_rate}}
    )
    window = SampleWindow("rec", 0, 1024, {}, samples, metadata, {})
    figure = Figure()
    spectrogram(
        figure, window, fft_size=64, overlap=0, window_function=window_function
    )
    axes = figure.axes[0]
    image = axes.images[0]
    power = image.get_array()
    assert power.shape == (64, 4)
    left, right, bottom, top = image.get_extent()
    frequency = bottom + (np.arange(64) + 0.5) * (top - bottom) / 64
    peaks = np.argmax(power, axis=0)
    rate = sample_rate or 1
    np.testing.assert_allclose(
        frequency[peaks] / rate, [0.125, 0.125, -0.25, -0.25]
    )
    np.testing.assert_allclose(
        power[peaks, np.arange(4)], [10 * np.log10(4)] * 2 + [0, 0], atol=1e-12
    )
    assert (left, right) == pytest.approx((1024, 1280))
    assert not axes.yaxis_inverted()
    assert axes.get_xlabel() == "Sample index"
    assert image.get_clim() == pytest.approx(
        (10 * np.log10(4) - 80, 10 * np.log10(4))
    )


def test_spectrogram_short_real_and_fixed_scale() -> None:
    """Real samples retain both frequency sides and support a fixed scale."""
    samples = np.cos(2 * np.pi * np.arange(32) / 8)
    window = SampleWindow("rec", None, 0, {}, samples, {}, {})
    figure = Figure()
    spectrogram(figure, window, fft_size=256, maximum_db=20, dynamic_range=40)
    image = figure.axes[0].images[0]
    assert image.get_array().shape == (32, 1)
    assert image.get_clim() == (-20, 20)
    np.testing.assert_allclose(
        image.get_array()[[12, 20], 0], -10 * np.log10(4)
    )


@pytest.mark.parametrize("length", [0, 1, 2, 128])
def test_spectrogram_zero_and_tiny_windows(length: int) -> None:
    """Tiny windows should explain limitations and zeros should remain dark.

    Args:
        length: Selected sample count.
    """
    window = SampleWindow("rec", 0, 0, {}, np.zeros(length), {}, {})
    figure = Figure()
    spectrogram(figure, window)
    axes = figure.axes[0]
    if length < 2:
        assert not axes.images
        assert "at least two samples" in axes.texts[0].get_text()
    else:
        image = axes.images[0]
        assert np.all(np.isfinite(image.get_array()))
        assert np.all(image.get_array() < image.get_clim()[0])
        if length == 128:
            assert image.get_array().shape == (32, 7)


def test_spectrogram_nonfinite_samples() -> None:
    """Invalid FFT frames should be masked without poisoning valid frames."""
    samples = np.ones(128, dtype=complex)
    samples[0] = np.nan
    window = SampleWindow("rec", 0, 0, {}, samples, {}, {})
    figure = Figure()
    spectrogram(figure, window, fft_size=64, overlap=0)
    power = figure.axes[0].images[0].get_array()
    assert power.mask[:, 0].all()
    assert not power.mask[:, 1].any()
    figure.clear()
    samples[:] = np.nan
    spectrogram(figure, window)
    assert not figure.axes[0].images
    assert "No finite" in figure.axes[0].texts[0].get_text()


@pytest.mark.parametrize(
    "settings",
    [
        {"fft_size": 0},
        {"fft_size": True},
        {"fft_size": 1.5},
        {"overlap": -0.1},
        {"overlap": 1},
        {"overlap": float("nan")},
        {"dynamic_range": 0},
        {"dynamic_range": float("inf")},
        {"maximum_db": float("nan")},
        {"window_function": "unknown"},
    ],
)
def test_spectrogram_bad_settings(settings: dict[str, Any]) -> None:
    """Invalid settings should fail before computing an FFT.

    Args:
        settings: Invalid display settings.
    """
    window = SampleWindow("rec", 0, 0, {}, np.ones(32), {}, {})
    with pytest.raises(ValueError):
        spectrogram(Figure(), window, **settings)
