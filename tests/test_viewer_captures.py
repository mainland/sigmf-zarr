"""Batched shared captures index items, not time samples within every item."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sigmf_zarr.json import JSONObject
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.viewer import DatasetSource, SampleWindow
from sigmf_zarr.viewer.captures import capture_segments
from sigmf_zarr.viewer.plots import spectrogram, waveform
from sigmf_zarr.viewer.presentation import segments, window_regions
from sigmf_zarr.viewer.regions import Span


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_appended_captures_select_items(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Appended batches must not introduce false time-axis boundaries.

    Args:
        tmp_path: Store directory.
        zarr_format: Physical Zarr format.
    """
    path = tmp_path / "batched.zarr"
    with SigMFZarrStore.create(path, zarr_format=zarr_format) as store:
        recording = store.recordings.create(
            "rec",
            batched=True,
            sample_shape=(8,),
            sample_axes=("time",),
            sample_dtype="complex64",
            global_metadata={"core:sample_rate": 8},
        )
        for frequency in (100, 200):
            recording.append_samples(
                np.ones((2, 8), dtype=np.complex64),
                capture={
                    "core:frequency": frequency,
                    "core:datetime": "2025-01-01T00:00:00Z",
                },
            )
        stored = deepcopy(recording.metadata())
        assert [c["core:sample_start"] for c in stored["captures"]] == [0, 2]
    source = DatasetSource(path)
    for item in range(4):
        window = source.read_window("rec", item=item, count=8)
        captures = tuple(segments(window, 8))
        assert [(c.start, c.stop) for c in captures] == [(0, 8)]
        assert captures[0].frequency == (100 if item < 2 else 200)
        assert (captures[0].timestamp is not None) == (item % 2 == 0)
        regions = window_regions(window, 8)
        assert len(regions) == 1
        assert regions[0].key == f"recording/captures/{item // 2}"
        assert regions[0].bounds["sample"] == Span(0, 8)
        assert regions[0].metadata == stored["captures"][item // 2]
        assert window.metadata == stored
        assert window.recording_metadata == stored

        figure = Figure()
        waveform(figure, window)
        # One connected trace for each of I and Q.
        assert [
            len(np.asarray(line.get_xdata()))
            for line in figure.axes[0].lines[:2]
        ] == [8, 8]
        figure.clear()
        spectrogram(figure, window, fft_size=8)
        assert len(figure.axes[0].images) == 1
        overview = source.read_overview("rec", item=item, count=8, fft_size=8)
        assert np.isfinite(overview.signal.spectrum).all()

    with SigMFZarrStore(path, mode="r") as store:
        assert store.recordings.open("rec").metadata() == stored


def test_shared_and_item_capture_coordinates() -> None:
    """Item overrides use time offsets while shared region IDs stay stable."""
    from sigmf_zarr.viewer.data import SampleWindow

    shared: JSONObject = {
        "global": {"core:offset": 100, "core:sample_rate": 8},
        "captures": [
            {"core:sample_start": 2, "core:frequency": 20},
            {"core:sample_start": 0, "core:frequency": 10},
            {"core:sample_start": 2, "core:frequency": 30},
            {"core:sample_start": -1, "core:frequency": 99},
        ],
        "annotations": [{"core:sample_start": 102, "core:sample_count": 2}],
    }
    local: JSONObject = {
        "captures": [{"core:sample_start": 104, "core:frequency": 40}],
    }
    shared_captures = shared["captures"]
    local_captures = local["captures"]
    assert isinstance(shared_captures, list)
    assert isinstance(local_captures, list)
    # JSON resolution appends lists without converting their coordinates.
    resolved = {**shared, "captures": shared_captures + local_captures}
    window = SampleWindow(
        "rec",
        3,
        0,
        {},
        np.ones(8),
        resolved,
        {},
        recording_metadata=shared,
        item_metadata=local,
        total_samples=8,
    )
    original = deepcopy(resolved)
    captures = tuple(segments(window, 8))
    assert [(c.start, c.stop, c.frequency) for c in captures] == [
        (0, 4, 30),
        (4, 8, 40),
    ]
    regions = window_regions(window, 8)
    assert [r.key for r in regions] == [
        "recording/captures/2",
        "recording/annotations/0",
        "item/captures/0",
    ]
    assert [r.bounds["sample"] for r in regions] == [
        Span(0, 4),
        Span(2, 4),
        Span(4, 8),
    ]
    assert regions[0].metadata["core:sample_start"] == 2
    assert resolved == original
    local_capture = local_captures[0]
    assert isinstance(local_capture, dict)
    local_capture["core:sample_start"] = 100
    assert [c.frequency for c in segments(window, 8)] == [40]

    # Before the first shared capture, timing and center frequency are unknown.
    future = {**shared, "captures": [shared_captures[0]]}
    early = replace(
        window, item=0, recording_metadata=future, item_metadata=None
    )
    capture = capture_segments(early.signal_metadata, 8)[0]
    assert capture.timestamp is None
    assert capture.frequency is None


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_appended_item_timestamps_keep_shared_capture_settings(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Independent item clocks retain shared settings without shared times.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    from fractions import Fraction

    path = tmp_path / "captured.zarr"
    with SigMFZarrStore.create(path, zarr_format=zarr_format) as store:
        recording = store.recordings.create(
            "rec",
            batched=True,
            sample_shape=(8,),
            sample_axes=("time",),
            global_metadata={"core:sample_rate": 8, "core:offset": 100},
        )
        recording.append_samples(
            np.ones((4, 8)),
            capture={
                "core:frequency": 100,
                "core:datetime": "2025-01-01T00:00:00Z",
            },
            item_captures=[
                [{}],
                [{"core:datetime": "2025-01-01T00:00:10.000000001Z"}],
                [
                    {"core:datetime": "2025-01-01T00:00:02Z"},
                    {"core:sample_start": 204, "core:frequency": 200},
                ],
                None,
            ],
            item_metadata=[
                None,
                None,
                {"global": {"core:offset": 200}},
                None,
            ],
        )
        stored = deepcopy(recording.metadata())
    source = DatasetSource(path)
    windows = [source.read_window("rec", item=i, count=8) for i in range(4)]
    captures = [tuple(segments(window, 8)) for window in windows]
    assert captures[0][0].timestamp is None
    assert captures[1][0].timestamp == (
        Fraction(1735689610) + Fraction(1, 1000000000)
    )
    assert captures[2][0].timestamp == Fraction(1735689602)
    assert [(c.start, c.stop, c.frequency) for c in captures[2]] == [
        (0, 4, 100),
        (4, 8, 200),
    ]
    assert captures[2][1].timestamp is None
    assert captures[3][0].timestamp is None
    assert all(item[0].frequency == 100 for item in captures)
    assert all(window.recording_metadata == stored for window in windows)
    assert windows[1].item_metadata == {
        "captures": [
            {
                "core:sample_start": 100,
                "core:datetime": "2025-01-01T00:00:10.000000001Z",
            }
        ],
    }


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_numbers_outside_float_range_are_not_drawn(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Unrepresentable JSON numbers must not prevent drawing a recording.

    Args:
        tmp_path: Store directory.
        zarr_format: Physical Zarr format.
    """
    path = tmp_path / "large.zarr"
    with SigMFZarrStore.create(path, zarr_format=zarr_format) as store:
        store.recordings.create(
            "rec",
            sample_dtype="complex64",
            sample_shape=(64,),
            sample_axes=("time",),
            global_metadata={"core:sample_rate": 10**400},
            captures=[{"core:sample_start": 0, "core:frequency": 10**400}],
            annotations=[
                {
                    "core:sample_start": 8,
                    "core:sample_count": 16,
                    "core:freq_lower_edge": 10**400,
                    "core:freq_upper_edge": 10**401,
                },
                {
                    "core:sample_start": 8,
                    "core:sample_count": 16,
                    "core:freq_lower_edge": -(10**400),
                    "core:freq_upper_edge": 0.25,
                },
            ],
        )
    window = DatasetSource(path).read_window("rec", count=64)
    assert [c.frequency for c in segments(window, 64)] == [None]
    regions = window_regions(window, 64)
    assert [r.key for r in regions] == [
        "recording/captures/0",
        "recording/annotations/0",
        "recording/annotations/1",
    ]
    assert all("frequency" not in r.bounds for r in regions)
    assert regions[1].bounds["sample"] == Span(8, 24)
    figure = Figure()
    FigureCanvasAgg(figure)
    for draw in (waveform, spectrogram):
        figure.clear()
        draw(figure, window)
        figure.canvas.draw()
        assert len(figure.axes[0].patches) == 3


@pytest.mark.parametrize("value", [2**63, 2**64, 2**100])
def test_integers_beyond_int64_are_drawn(value: int) -> None:
    """JSON integers outside int64 but inside the float range must plot.

    Args:
        value: Frequency and sample-rate magnitude in Hz.
    """
    metadata: JSONObject = {
        "global": {"core:sample_rate": value},
        "captures": [{"core:sample_start": 0, "core:frequency": value}],
        "annotations": [
            {
                "core:sample_start": 8,
                "core:sample_count": 16,
                "core:freq_lower_edge": value,
                "core:freq_upper_edge": 2 * value,
            }
        ],
    }
    window = SampleWindow(
        "rec", None, 0, {}, np.ones(64, dtype=complex), metadata, {}
    )
    assert [c.frequency for c in segments(window, 64)] == [float(value)]
    annotation = window_regions(window, 64)[1]
    assert annotation.bounds["frequency"] == Span(
        float(value), float(2 * value), "Hz", "RF"
    )
    figure = Figure()
    FigureCanvasAgg(figure)
    spectrogram(figure, window)
    figure.canvas.draw()
    assert len(figure.axes[0].patches) == 2
