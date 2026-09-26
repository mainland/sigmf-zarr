"""Streaming reductions must preserve extrema, statistics, and FFT frames."""

from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr.store import SigMFZarrStore
from sigmf_zarr.viewer import DatasetSource
from sigmf_zarr.viewer.overview import summarize_peaks, summarize_signal
from sigmf_zarr.viewer.windows import WindowFunction


def test_peak_envelope_preserves_bursts_across_blocks() -> None:
    """Peak scans must retain short bursts without integer overflow."""
    samples = np.zeros(131075, dtype=np.int16)
    samples[65535:65538] = [-32768, 42, 31000]
    edges = np.array([i * len(samples) // 1024 for i in range(1025)])
    expected = np.array(
        [
            np.abs(samples[lo:hi].astype(float)).max()
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    actual = summarize_peaks(np.split(samples, [65536, 131072]), len(samples))
    np.testing.assert_array_equal(actual, expected)
    assert actual.max() == 32768
    np.testing.assert_array_equal(
        summarize_peaks([np.array([3 + 4j])], 1), [5]
    )
    np.testing.assert_array_equal(
        summarize_peaks([np.array([np.nan, np.inf])], 2), [0, 0]
    )
    for blocks, count, bins in (
        ([], 1, 1),
        ([np.ones(3)], 2, 1),
        ([], 0, 1),
        ([], 1, 4097),
    ):
        with pytest.raises(ValueError):
            summarize_peaks(blocks, count, bins=bins)


@pytest.mark.parametrize("complex_signal", [False, True])
@pytest.mark.parametrize(
    "window_function",
    ["hann", "hamming", "blackman", "blackmanharris", "rectangular"],
)
def test_streaming_reduction_matches_full_signal(
    complex_signal: bool, window_function: WindowFunction
) -> None:
    """Arbitrary block boundaries must preserve all frames and source extrema.

    Args:
        complex_signal: Whether to include independent quadrature samples.
        window_function: Periodic window used by every complete FFT frame.
    """
    rng = np.random.default_rng(12)
    samples = rng.normal(size=2003)
    if complex_signal:
        samples = samples + 1j * rng.normal(size=2003)
    samples[1777] = 123
    summary = summarize_signal(
        np.split(samples, [17, 266, 1266]),
        len(samples),
        bins=13,
        fft_size=64,
        overlap=0.75,
        iq_points=57,
        window_function=window_function,
    )
    for index, (start, stop) in enumerate(
        zip(summary.edges[:-1], summary.edges[1:], strict=True)
    ):
        part = samples[start:stop]
        assert summary.minimum[index].real == part.real.min()
        assert summary.maximum[index].real == part.real.max()
        assert summary.minimum[index].imag == part.imag.min()
        assert summary.maximum[index].imag == part.imag.max()
        assert summary.peak[index] == np.abs(part).max()
    assert summary.mean_power == pytest.approx(np.mean(np.abs(samples) ** 2))
    assert summary.peak_magnitude == 123
    np.testing.assert_array_equal(
        summary.iq, samples[np.array([i * 2002 // 56 for i in range(57)])]
    )
    taper = {
        "hann": np.hanning(65)[:-1],
        "hamming": np.hamming(65)[:-1],
        "blackman": np.blackman(65)[:-1],
        # Expand cos(2*x) and cos(3*x) into a polynomial in cos(x), giving an
        # equivalent reference without calling the production window helper.
        "blackmanharris": np.polynomial.polynomial.polyval(
            np.cos(2 * np.pi * np.arange(64) / 64),
            [0.21747, -0.45325, 0.28256, -0.04672],
        ),
        "rectangular": np.ones(64),
    }[window_function]
    assert summary.window_function == window_function
    frames = np.lib.stride_tricks.sliding_window_view(samples, 64)[::16]
    power = (
        np.abs(
            np.fft.fftshift(np.fft.fft(frames * taper, axis=1), axes=1)
            / taper.sum()
        )
        ** 2
    )
    np.testing.assert_allclose(summary.spectrum, power.mean(axis=0))
    rows = (
        np.searchsorted(
            summary.frame_edges, np.arange(len(frames)) * 16 + 32, side="right"
        )
        - 1
    )
    for row in range(13):
        np.testing.assert_allclose(
            summary.power[row], power[rows == row].mean(axis=0)
        )
    for array in (
        summary.minimum,
        summary.maximum,
        summary.peak,
        summary.edges,
        summary.power,
        summary.spectrum,
        summary.iq,
    ):
        assert not array.flags.writeable


def test_reduction_handles_invalid_frames_and_input() -> None:
    """Invalid frames must not poison valid spectral averages."""
    samples = np.ones(128)
    samples[1] = np.nan
    result = summarize_signal([samples], 128, bins=2, fft_size=64, overlap=0)
    assert np.isnan(result.power[0]).all()
    assert np.isfinite(result.power[1]).all()
    np.testing.assert_allclose(result.spectrum, result.power[1])
    assert np.isnan(result.mean_power)
    for blocks, count in (
        ([samples], 129),
        ([samples], 127),
        ([np.ones(65537)], 65537),
        ([np.ones((2, 3))], 6),
        ([np.ones(4), np.ones(4, dtype=complex)], 8),
    ):
        with pytest.raises(ValueError):
            summarize_signal(blocks, count)
    for settings in (
        {"bins": 0},
        {"fft_size": 1},
        {"fft_size": 8193},
        {"overlap": 1},
        {"iq_points": 0},
    ):
        with pytest.raises(ValueError):
            summarize_signal([samples], 128, **settings)


def test_overview_reads_bounded_blocks_and_cancels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long source reads must cover all samples in bounded, cancellable blocks.

    Args:
        tmp_path: Temporary store directory.
        monkeypatch: Array-read observer.
    """
    path = tmp_path / "long.zarr"
    store = SigMFZarrStore.create(path)
    recording = store.recordings.open(
        "long",
        batched=False,
        sample_dtype="complex64",
        sample_shape=(150000,),
        sample_axes=("time",),
    )
    recording.set_samples(np.full(150000, 3 + 4j, dtype=np.complex64))
    source = DatasetSource(path)
    original = Array.__getitem__
    reads: list[slice] = []

    def observe(
        array: Array, selection: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Record sample-array slices before delegating to Zarr."""
        if array.path.endswith("samples"):
            reads.append(selection[0])
        return original(array, selection, *args, **kwargs)

    monkeypatch.setattr(Array, "__getitem__", observe)
    result = source.read_overview("long", start=7, count=140000)
    assert reads == [
        slice(7, 8),
        slice(7, 65543),
        slice(65543, 131079),
        slice(131079, 140007),
    ]
    assert result.context.start == 7
    assert result.context.samples.size == 0
    assert result.signal.count == 140000
    assert result.signal.mean_power == 25
    assert result.signal.peak_magnitude == 5
    assert result.signal.power.shape == (512, 256)
    reads.clear()
    with pytest.raises(CancelledError):
        source.read_overview(
            "long", count=140000, cancelled=lambda: len(reads) >= 2
        )
    assert reads == [slice(0, 1), slice(0, 65536)]
    with pytest.raises(IndexError):
        source.read_overview("long", start=100000, count=100000)
    reads.clear()
    peaks = source.read_peaks("long", start=7, count=140000)
    assert reads == [
        slice(7, 65543),
        slice(65543, 131079),
        slice(131079, 140007),
    ]
    np.testing.assert_array_equal(peaks, np.full(1024, 5))
    reads.clear()
    with pytest.raises(CancelledError):
        source.read_peaks("long", count=140000, cancelled=lambda: bool(reads))
    assert reads == [slice(0, 65536)]
    with pytest.raises(IndexError):
        source.read_peaks("long", start=100000, count=100000)
    with pytest.raises(ValueError):
        source.read_peaks("long", count=0)
