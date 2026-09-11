"""Region projections and acquisition boundaries preserve source semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sigmf_zarr.viewer.captures import (
    capture_segments,
    project_samples,
    sigmf_regions,
)
from sigmf_zarr.viewer.data import OverviewWindow, SampleWindow
from sigmf_zarr.viewer.overlays import RegionLabel, RegionPatch, draw_regions
from sigmf_zarr.viewer.overview import summarize_signal
from sigmf_zarr.viewer.plots import overview_spectrogram, spectrogram, waveform
from sigmf_zarr.viewer.presentation import (
    PlotOptions,
    sample_interval,
    window_regions,
)
from sigmf_zarr.viewer.regions import Region, Span


@pytest.fixture
def window() -> SampleWindow:
    """Return two discontinuous captures with a crossing RF annotation."""
    metadata: dict[str, Any] = {
        "global": {"core:sample_rate": 1000, "core:offset": 200},
        "captures": [
            {
                "core:sample_start": 200,
                "core:datetime": "2025-01-01T00:00:00Z",
                "core:frequency": 100,
            },
            {
                "core:sample_start": 300,
                "core:datetime": "2025-01-01T00:00:10Z",
                "core:frequency": 105,
            },
        ],
        "annotations": [
            {
                "core:sample_start": 280,
                "core:sample_count": 40,
                "core:freq_lower_edge": 110,
                "core:freq_upper_edge": 120,
                "core:label": "Burst",
            }
        ],
    }
    return SampleWindow(
        "test",
        None,
        0,
        {},
        np.r_[np.ones(100), -np.ones(100)],
        metadata,
        {},
        recording_metadata=metadata,
        total_samples=200,
    )


def test_generic_regions_and_capture_mapping(window: SampleWindow) -> None:
    """Adapters must preserve source metadata and piecewise timestamp gaps.

    Args:
        window: Discontinuous capture fixture.
    """
    captures = capture_segments(window.metadata, 200)
    assert [(c.start, c.stop) for c in captures] == [(0, 100), (100, 200)]
    epoch = captures[0].timestamp
    np.testing.assert_allclose(
        project_samples(
            [100, 120], captures[1], 1000, "timestamp", epoch=epoch
        ),
        [10, 10.02],
    )
    np.testing.assert_allclose(
        project_samples([100, 120], captures[1], 1000, "elapsed"), [0.1, 0.12]
    )
    regions = sigmf_regions(window.metadata, 200)
    annotation = regions[-1]
    assert annotation.bounds["sample"] == Span(80, 120)
    assert annotation.metadata["core:sample_start"] == 280
    assert annotation.key == "recording/annotations/0"
    assert annotation.intersects("sample", 90, 100)
    assert not annotation.intersects("sample", 120, 130)
    assert Region("custom", {"channel": Span(1, 3)}).intersects(
        "sample", 0, 100
    )
    with pytest.raises(ValueError):
        Span(4, 4)


def test_inverse_sample_coordinates(window: SampleWindow) -> None:
    """Plot navigation must resolve offsets, timestamp gaps, and overlaps.

    Args:
        window: Recording with discontinuous captures and a sample offset.
    """
    assert sample_interval(window, 200, (90.2, 20.5), PlotOptions()) == (
        20,
        91,
    )
    assert sample_interval(window, 200, (-100, 300), PlotOptions()) == (0, 200)
    assert sample_interval(window, 200, (300, 400), PlotOptions()) is None
    elapsed = PlotOptions(axis_mode="elapsed")
    assert sample_interval(window, 200, (0.025, 0.075), elapsed) == (25, 75)
    timestamp = PlotOptions(axis_mode="timestamp")
    assert sample_interval(window, 200, (1, 2), timestamp) is None
    assert sample_interval(window, 200, (10, 11), timestamp) == (100, 200)
    window.metadata["captures"][1]["core:datetime"] = window.metadata[
        "captures"
    ][0]["core:datetime"]
    assert sample_interval(window, 200, (0.025, 0.075), timestamp) == (25, 175)
    del window.metadata["captures"][1]["core:datetime"]
    assert sample_interval(window, 200, (0.025, 0.075), timestamp) == (25, 75)
    del window.metadata["global"]["core:sample_rate"]
    assert sample_interval(window, 200, (0, 1), timestamp) is None


def test_annotation_label_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Annotation labels appear only when the visible box has enough space.

    Args:
        monkeypatch: Fixture recording text rendering calls.
    """
    from matplotlib.text import Text

    figure = Figure(figsize=(6, 4))
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_xlim(0, 10000)
    axes.set_ylim(-100, 100)
    metadata = {"core:description": "Packet", "custom:value": "<test>"}
    region = Region(
        "packet",
        {"position": Span(100, 200), "frequency": Span(-10, 10)},
        "Packet",
        metadata=metadata,
    )
    draw_regions(axes, [region])
    patch = axes.patches[0]
    assert isinstance(patch, RegionPatch)
    assert patch.region.metadata is metadata
    assert patch.get_edgecolor()[3] == 1
    assert patch.get_facecolor()[3] == 0.1
    assert not patch.get_path_effects()
    assert not axes.lines
    drawn = []
    original = Text.draw

    def record_draw(text: Text, renderer: Any) -> None:
        """Record visible annotation labels.

        Args:
            text: Text artist being rendered.
            renderer: Matplotlib renderer.
        """
        if isinstance(text, RegionLabel):
            drawn.append(text.get_text())
        original(text, renderer)

    monkeypatch.setattr(Text, "draw", record_draw)
    canvas.draw()
    assert patch.get_linewidth() * canvas.get_renderer().points_to_pixels(
        1
    ) == pytest.approx(1)
    assert not drawn
    axes.set_xlim(90, 210)
    axes.set_ylim(-15, 15)
    canvas.draw()
    assert drawn == ["Packet"]
    drawn.clear()
    axes.set_xlim(300, 400)
    canvas.draw()
    assert not drawn


def test_overlays_split_at_retunes_and_keep_source_identity(
    window: SampleWindow,
) -> None:
    """One annotation must project to separate rectangles at a retune.

    Args:
        window: Discontinuous capture fixture.
    """
    fig = Figure()
    spectrogram(
        fig, window, fft_size=64, options=PlotOptions(axis_mode="timestamp")
    )
    axes = fig.axes[0]
    assert len(axes.images) == 2
    assert axes.images[0].get_extent()[1] < axes.images[1].get_extent()[0]
    patches = [
        p for p in axes.patches if p.get_gid() == "recording/annotations/0"
    ]
    assert len(patches) == 2
    assert [(p.get_y(), p.get_height()) for p in patches] == [
        (10, 10),
        (5, 10),
    ]
    assert patches[0].get_x() == pytest.approx(0.08)
    assert patches[1].get_x() == pytest.approx(10)
    assert len(axes.lines) == 2  # Capture boundaries only.
    assert window.metadata["annotations"][0]["core:sample_start"] == 280
    fig.clear()
    spectrogram(fig, window, fft_size=64, options=PlotOptions(captures=False))
    assert len(fig.axes[0].patches) == 2
    assert not fig.axes[0].lines
    fig.clear()
    spectrogram(
        fig,
        window,
        fft_size=64,
        options=PlotOptions(captures=False, annotations=False),
    )
    assert not fig.axes[0].patches
    assert len(fig.axes[0].images) == 2


def test_capture_boundaries_exclude_fft_transients(
    window: SampleWindow,
) -> None:
    """A discontinuous sign change must not create a spurious FFT transient.

    Args:
        window: Discontinuous constant signals.
    """
    signal = summarize_signal(
        np.split(window.samples, [73, 129]),
        200,
        fft_size=64,
        boundaries=(100,),
    )
    taper = np.hanning(65)[:-1]
    expected = np.abs(np.fft.fftshift(np.fft.fft(taper)) / taper.sum()) ** 2
    np.testing.assert_allclose(signal.spectrum, expected, atol=1e-20)
    assert signal.mean_power == 1
    fig = Figure()
    overview_spectrogram(
        fig,
        OverviewWindow(replace(window, samples=window.samples[:0]), signal),
        options=PlotOptions(axis_mode="timestamp"),
    )
    assert len(fig.axes[0].collections) == 2
    for boundaries in ((100, 50), (100, 100), (-1,), (200,)):
        with pytest.raises(ValueError):
            summarize_signal([window.samples], 200, boundaries=boundaries)


def test_waveform_modes_overlap_unknown_and_fine_timestamps(
    window: SampleWindow,
) -> None:
    """Waveforms must remain separate across gaps and preserve fine spacing.

    Args:
        window: Capture fixture.
    """
    fig = Figure()
    options = PlotOptions(
        axis_mode="elapsed", captures=False, annotations=False
    )
    waveform(fig, window, options=options)
    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].lines[1].get_xdata()[0] == pytest.approx(0.1)
    captures = window.metadata["captures"]
    captures[1]["core:datetime"] = captures[0]["core:datetime"]
    window.metadata["global"]["core:sample_rate"] = 40000000
    fig.clear()
    waveform(fig, window, options=replace(options, axis_mode="timestamp"))
    lines = fig.axes[0].lines
    assert lines[0].get_xdata()[0] == lines[1].get_xdata()[0] == 0
    assert lines[0].get_xdata()[1] == pytest.approx(1 / 40000000)
    del captures[1]["core:datetime"]
    fig.clear()
    waveform(fig, window, options=replace(options, axis_mode="timestamp"))
    assert len(fig.axes[0].lines) == 1
    assert "unknown timing" in fig.axes[0].texts[-1].get_text()


def test_region_sources_preserve_recording_and_item_scopes(
    window: SampleWindow,
) -> None:
    """Resolved regions must still identify the original JSON scope.

    Args:
        window: Recording metadata fixture.
    """
    item = {
        "annotations": [
            {
                "core:sample_start": 210,
                "core:sample_count": 3,
                "core:label": "Item label",
            }
        ]
    }
    resolved = {
        **window.metadata,
        "annotations": window.metadata["annotations"] + item["annotations"],
    }
    regions = window_regions(
        replace(window, metadata=resolved, item_metadata=item), 200
    )
    assert [r.key for r in regions] == [
        "recording/captures/0",
        "recording/captures/1",
        "recording/annotations/0",
        "item/annotations/0",
    ]
    assert regions[-1].bounds["sample"] == Span(10, 13)


def test_implicit_annotation_extent_and_fractional_timestamps(
    window: SampleWindow,
) -> None:
    """Implicit extents end at capture boundaries and UTC retains fractions.

    Args:
        window: Offset recording with two captures.
    """
    window.metadata["annotations"] = [
        {"core:sample_start": 290},
        {"core:sample_start": 190},
    ]
    annotations = [
        r
        for r in sigmf_regions(window.metadata, 200)
        if r.kind == "annotation"
    ]
    assert [r.bounds["sample"] for r in annotations] == [
        Span(90, 100),
        Span(0, 100),
    ]
    window.metadata["captures"][0]["core:datetime"] = (
        "2025-01-01T00:00:00.142626Z"
    )
    figure = Figure()
    waveform(figure, window, options=PlotOptions(axis_mode="timestamp"))
    formatter = figure.axes[0].xaxis.get_major_formatter()
    assert formatter(0) == "2025-01-01\n00:00:00.142626"
    assert formatter(0.00000001).endswith("00:00:00.14262601")
