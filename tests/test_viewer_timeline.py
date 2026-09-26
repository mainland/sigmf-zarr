"""Reusable range gestures and full-signal viewer navigation."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
pytest.importorskip("jmespath")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from sigmf_zarr.store import SigMFZarrStore
from sigmf_zarr.viewer import DatasetSource, OverviewWindow, SampleWindow
from sigmf_zarr.viewer.qt import DatasetViewer
from sigmf_zarr.viewer.timeline import RangeSelector


@pytest.fixture(scope="module")
def app() -> QApplication:
    """Return the process-wide offscreen application."""
    return QApplication.instance() or QApplication([])


def wait_for(app: QApplication, predicate: Callable[[], bool]) -> None:
    """Process events until a worker result arrives.

    Args:
        app: Qt application.
        predicate: Completion condition.

    Raises:
        AssertionError: If the condition does not succeed within 15 seconds.
    """
    deadline = time.monotonic() + 15
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_range_selector_gestures(app: QApplication) -> None:
    """Moving and resizing must preserve valid half-open intervals.

    Args:
        app: Qt application.
    """
    selector = RangeSelector()
    selector.resize(424, 80)
    selector.set_extent(1000)
    selector.set_overview([0, 2, 1, 4, 0])
    selector.show()
    app.processEvents()
    selector.set_range(200, 400)

    def point(position: int) -> QPoint:
        """Map a source coordinate into the test widget.

        Args:
            position: Integer source position.

        Returns:
            Widget pixel position.
        """
        return QPoint(12 + position * 400 // 1000, 40)

    def drag(start: int, stop: int) -> None:
        """Send a pointer drag across the test widget.

        Args:
            start: Initial source coordinate.
            stop: Final source coordinate.
        """
        QTest.mousePress(selector, Qt.MouseButton.LeftButton, pos=point(start))
        QTest.mouseMove(selector, point(stop))
        QTest.mouseRelease(
            selector, Qt.MouseButton.LeftButton, pos=point(stop)
        )

    try:
        drag(300, 600)
        assert selector.selection == (500, 700)
        drag(500, 400)
        assert selector.selection == (400, 700)
        drag(700, 900)
        assert selector.selection == (400, 900)
        QTest.mouseClick(selector, Qt.MouseButton.LeftButton, pos=point(100))
        assert selector.selection == (0, 500)
        QTest.keyClick(selector, Qt.Key.Key_Home)
        assert selector.selection == (0, 500)
        QTest.keyClick(selector, Qt.Key.Key_Plus)
        assert selector.visible_range == (250, 750)
        assert selector.selection == (0, 500)
        QTest.keyClick(selector, Qt.Key.Key_Right)
        assert selector.selection == (25, 525)
        QTest.keyClick(
            selector, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier
        )
        assert selector.selection == (25, 500)
        selector.set_extent(1)
        QTest.keyClick(selector, Qt.Key.Key_Plus)
        assert selector.selection == (0, 1)
    finally:
        selector.close()


def test_range_selector_large_coordinates(app: QApplication) -> None:
    """The generic range API must preserve integers beyond 32-bit limits.

    Args:
        app: Qt application.
    """
    selector = RangeSelector()
    events: list[tuple[int, int]] = []
    selector.range_changed.connect(
        lambda start, stop: events.append((start, stop))
    )
    extent = 2**45 + 17
    selector.set_extent(extent)
    selector.set_range(extent - 100, extent - 1)
    assert events[-1] == (extent - 100, extent - 1)
    selector.zoom_to_selection()
    assert selector.visible_range == selector.selection
    selector.scrollbar.setValue(0)
    assert selector.visible_range == (0, 99)
    selector.scrollbar.setValue(selector.scrollbar.maximum())
    assert selector.visible_range == (extent - 99, extent)
    assert selector.selection == (extent - 100, extent - 1)
    # A viewport almost as long as the recording must not overflow Qt ints.
    selector.set_visible_range(0, extent - 1)
    for start, stop in ((-1, 4), (1, 1), (0, extent + 1)):
        with pytest.raises(ValueError):
            selector.set_range(start, stop)
    selector.set_extent(0)
    assert selector.selection == (0, 0)
    assert selector.visible_range == (0, 0)
    assert not selector.isEnabled()


def test_overview_zoom_preserves_selection(app: QApplication) -> None:
    """Pointer zoom and scrollbar pan must not emit selection changes.

    Args:
        app: Qt application.
    """
    selector = RangeSelector()
    selector.resize(424, 100)
    selector.set_extent(1000)
    selector.set_range(400, 600)
    events: list[tuple[int, int]] = []
    selector.range_changed.connect(lambda lo, hi: events.append((lo, hi)))
    selector.show()
    app.processEvents()
    try:
        pointer = QPointF(112, 40)  # One quarter across the track.
        wheel = QWheelEvent(
            pointer,
            pointer,
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(selector, wheel)
        assert selector.visible_range == (125, 625)
        assert selector._coordinate(112) == 250
        selector.scrollbar.setValue(selector.scrollbar.maximum())
        assert selector.visible_range == (500, 1000)
        selector.zoom_to_selection()
        assert selector.visible_range == (400, 600)
        assert selector.selection == (400, 600)
        assert events == []
        # Selection handles still operate in source coordinates after zoom.
        QTest.mousePress(
            selector, Qt.MouseButton.LeftButton, pos=QPoint(12, 40)
        )
        QTest.mouseMove(selector, QPoint(112, 40))
        QTest.mouseRelease(
            selector, Qt.MouseButton.LeftButton, pos=QPoint(112, 40)
        )
        assert selector.selection == (450, 600)
        selector.fit_view()
        assert selector.visible_range == (0, 1000)
        assert selector.selection == (450, 600)
        selector.set_visible_range(0, 1)
        selector.set_overview(np.ones(512))
        selector.grab()  # Offscreen peaks must not overflow painter integers.
        for start, stop in ((-1, 10), (0, 0), (0, 1001)):
            with pytest.raises(ValueError):
                selector.set_visible_range(start, stop)
    finally:
        selector.close()


def test_overview_cache_retains_recent_regions_and_clears_on_reset(
    app: QApplication,
) -> None:
    """Recent envelopes must survive navigation with bounded retention.

    Args:
        app: Qt application.
    """
    selector = RangeSelector(cache_size=2)
    selector.set_extent(1000)
    selector.set_overview([1, 2, 1, 2])
    for start in (100, 300):
        selector.set_visible_range(start, start + 100)
        selector.set_overview([1, 4, 2], start=start, stop=start + 100)
    selector.set_visible_range(100, 200)
    np.testing.assert_array_equal(selector._overview, [0.25, 1, 0.5])
    selector.set_visible_range(500, 600)
    selector.set_overview([4, 2, 1], start=500, stop=600)
    assert selector.has_overview(100, 200)
    assert not selector.has_overview(300, 400)
    assert selector.has_overview(0, 1000)
    assert len(selector._overview_cache) == 2
    selector.set_visible_range(110, 190)
    assert selector._overview_range == (100, 200)
    assert not selector.has_overview(110, 190)
    selector.set_visible_range(800, 900)
    assert selector._overview_range == (0, 1000)
    np.testing.assert_array_equal(selector._overview, [0.5, 1, 0.5, 1])
    selector.set_extent(1000)  # Another signal with the same length.
    assert not selector.has_overview(0, 1000)
    assert not selector.has_overview(100, 200)
    assert not selector._overview_cache
    assert selector._overview.size == 0


def test_overview_detail_cache_can_be_disabled(app: QApplication) -> None:
    """A consumer may disable retention while keeping the full preview.

    Args:
        app: Qt application.
    """
    selector = RangeSelector(cache_size=0)
    selector.set_extent(1000)
    selector.set_overview([1])
    selector.set_visible_range(100, 200)
    selector.set_overview([1, 2], start=100, stop=200)
    assert selector.has_overview(100, 200)
    selector.fit_view()
    selector.set_visible_range(100, 200)
    assert not selector.has_overview(100, 200)
    assert selector._overview_range == (0, 1000)
    with pytest.raises(ValueError):
        RangeSelector(cache_size=-1)


@pytest.fixture
def source(tmp_path: Path) -> DatasetSource:
    """Create long and short continuous recordings.

    Args:
        tmp_path: Temporary dataset directory.

    Returns:
        Read-only viewer source.
    """
    path = tmp_path / "signals.zarr"
    store = SigMFZarrStore.create(path)
    for name, count in (("long", 131073), ("short", 4096)):
        rec = store.recordings.open(
            name,
            batched=False,
            sample_dtype="complex64",
            sample_shape=(count,),
            sample_axes=("time",),
            global_metadata={"core:sample_rate": 8000},
        )
        rec.set_samples(
            np.exp(2j * np.pi * np.arange(count) / 8).astype(np.complex64)
        )
    store.collections.open("Signals", recording_ids=["long", "short"])
    return DatasetSource(path)


def test_full_signal_and_selected_range(
    app: QApplication, source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full overviews and raw intervals must be distinct consumer contracts.

    Args:
        app: Qt application.
        source: Long and short test signals.
        monkeypatch: Read observer.
    """
    widget = DatasetViewer(source)
    raw: list[SampleWindow | None] = []
    reduced: list[OverviewWindow | None] = []
    custom = QLabel("Custom raw panel")
    widget.add_panel("Custom", custom, raw.append)
    widget.overview_changed.connect(reduced.append)
    widget.resize(1200, 850)
    widget.show()
    try:
        wait_for(app, lambda: "Reduced overview" in widget.status.text())
        assert widget.timeline.selection == (0, 131073)
        for branch in (
            widget.tree.topLevelItem(1),
            widget.tree.topLevelItem(0).child(0),
        ):
            for index in range(branch.childCount()):
                node = branch.child(index)
                assert node.childCount() == 0
        assert widget.timeline_controls.isVisible()
        widget.tabs.setCurrentWidget(widget.recording_metadata)
        assert not widget.timeline_controls.isVisible()
        widget.tabs.setCurrentIndex(0)
        assert widget.timeline_controls.isVisible()
        assert not widget.tabs.isTabVisible(widget.tabs.indexOf(custom))
        assert raw[-1] is None
        assert reduced[-1].signal.count == 131073
        assert all(widget.tabs.isTabVisible(i) for i in range(4))
        assert widget.tabs.widget(0).figure.axes[0].collections
        original_peak = widget.timeline._overview.copy()
        widget.timeline.set_range(100000, 104096)
        wait_for(
            app, lambda: "Samples [100000, 104096)" in widget.status.text()
        )
        assert widget.tabs.isTabVisible(widget.tabs.indexOf(custom))
        assert raw[-1].start == 100000
        assert len(raw[-1].samples) == 4096
        np.testing.assert_array_equal(widget.timeline._overview, original_peak)

        def no_rescan(*args: Any, **kwargs: Any) -> Any:
            """Fail if Fit signal redundantly scans the full recording."""
            raise AssertionError("Unexpected full scan")

        with monkeypatch.context() as patch:
            patch.setattr(source, "read_overview", no_rescan)
            widget.timeline.fit()
            wait_for(app, lambda: "Reduced overview" in widget.status.text())
        panel = widget.tabs.widget(0)
        artist = panel.figure.axes[0].collections[0]
        with monkeypatch.context() as patch:
            patch.setattr(source, "read_overview", no_rescan)
            patch.setattr(source, "read_window", no_rescan)
            panel.colormap.setCurrentText("inferno")
            assert panel.figure.axes[0].collections[0] is artist
            assert artist.get_cmap().name == "inferno"
        panel.fft_size.setCurrentText("512")
        wait_for(
            app,
            lambda: (
                "Reduced overview" in widget.status.text()
                and panel._overview is not None
                and panel._overview.signal.fft_size == 512
            ),
        )
        assert panel._overview.signal.hop == 256
        assert panel.figure.axes[0].collections[0].get_cmap().name == "inferno"
        # A shorter recording also starts at its entire extent, beyond 1024.
        widget.tree.setCurrentItem(widget.tree.topLevelItem(1).child(1))
        wait_for(
            app, lambda: "short | Samples [0, 4096)" in widget.status.text()
        )
        assert widget.timeline.selection == (0, 4096)
        widget.tree.setCurrentItem(widget.tree.topLevelItem(1))
        assert not widget.timeline_controls.isVisible()
        assert widget.timeline.extent == 0
    finally:
        widget.close()
        app.processEvents()


def test_narrow_selection_while_overview_loads(
    app: QApplication, source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow overview must neither move the selection nor replace raw data.

    Args:
        app: Qt application.
        source: Long test signal.
        monkeypatch: Controlled slow read.
    """
    started, release = Event(), Event()
    read = source.read_overview

    def delayed(*args: Any, **kwargs: Any) -> OverviewWindow:
        """Wait for the test to release the full-signal scan."""
        started.set()
        assert release.wait(10)
        return read(*args, **kwargs)

    monkeypatch.setattr(source, "read_overview", delayed)
    widget = DatasetViewer(source)
    widget.resize(1200, 850)
    widget.show()
    try:
        wait_for(app, started.is_set)
        widget.timeline.set_range(2000, 3000)
        wait_for(app, lambda: "Samples [2000, 3000)" in widget.status.text())
        widget.timeline.set_visible_range(1800, 3200)
        wait_for(app, lambda: widget.timeline._overview_range == (1800, 3200))
        release.set()
        wait_for(app, lambda: widget._full_view is not None)
        assert widget.timeline.selection == (2000, 3000)
        assert widget.timeline._overview_range == (1800, 3200)
        assert "Samples [2000, 3000)" in widget.status.text()
        assert widget.tabs.widget(0)._window.start == 2000
    finally:
        release.set()
        widget.close()
        app.processEvents()


def test_overview_detail_is_independent_and_discards_stale_results(
    app: QApplication, source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Viewport reads must preserve panels and ignore superseded results.

    Args:
        app: Qt application.
        source: Long and short test signals.
        monkeypatch: Controlled detail reader.
    """
    started, release, finished = Event(), Event(), Event()
    calls: list[tuple[int, int]] = []
    read = source.read_peaks

    def delayed(name: str, **kwargs: Any) -> Any:
        """Delay one envelope while allowing a newer request to complete.

        Args:
            name: Recording name.
            **kwargs: Envelope selection and cancellation arguments.

        Returns:
            Requested peak magnitudes.
        """
        calls.append((kwargs["start"], kwargs["count"]))
        if kwargs["start"] == 1000:
            started.set()
            assert release.wait(10)
            # Simulate an I/O backend that completes despite cancellation.
            kwargs.pop("cancelled")
        result = read(name, **kwargs)
        if kwargs["start"] == 1000:
            finished.set()
        return result

    widget = DatasetViewer(source)
    widget.resize(1200, 850)
    widget.show()
    monkeypatch.setattr(source, "read_peaks", delayed)
    try:
        wait_for(app, lambda: "Reduced overview" in widget.status.text())
        selection = widget.timeline.selection
        status = widget.status.text()
        panel_view = widget.tabs.widget(0)._overview
        full_peaks = widget.timeline._overview.copy()
        widget.timeline.set_visible_range(1000, 2000)
        wait_for(app, started.is_set)
        widget.timeline.set_visible_range(70000, 71000)
        wait_for(
            app, lambda: widget.timeline._overview_range == (70000, 71000)
        )
        release.set()
        wait_for(app, finished.is_set)
        widget._tasks.poll()
        assert widget.timeline._overview_range == (70000, 71000)
        assert len(widget.timeline._overview) == 1000
        assert calls == [(1000, 1000), (70000, 1000)]
        assert widget.timeline.selection == selection
        assert widget.status.text() == status
        assert widget.tabs.widget(0)._overview is panel_view
        widget.timeline.fit_view()
        np.testing.assert_array_equal(widget.timeline._overview, full_peaks)
        assert calls == [(1000, 1000), (70000, 1000)]
        started.clear()
        release.clear()
        finished.clear()
        widget.timeline.set_visible_range(1000, 2000)
        wait_for(app, started.is_set)
        widget.tree.setCurrentItem(widget.tree.topLevelItem(1).child(1))
        wait_for(app, lambda: "short | Samples" in widget.status.text())
        release.set()
        wait_for(app, finished.is_set)
        widget._tasks.poll()
        assert widget.timeline.visible_range == (0, 4096)
        assert widget.timeline._overview_range == (0, 4096)
    finally:
        release.set()
        widget.close()
        app.processEvents()


def test_revisited_overview_is_immediate_without_sample_reads(
    app: QApplication, source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed detail summary must reappear before timers or I/O run.

    Args:
        app: Qt application.
        source: Long and short test signals.
        monkeypatch: Read guard for the cached navigation path.
    """
    widget = DatasetViewer(source)
    widget.resize(1200, 850)
    widget.show()
    try:
        wait_for(app, lambda: "Reduced overview" in widget.status.text())
        selected = widget.timeline.selection
        panel = widget.tabs.widget(0)._overview
        for start in (1000, 70000):
            widget.timeline.set_visible_range(start, start + 1000)
            wait_for(
                app,
                lambda start=start: (
                    widget.timeline._overview_range == (start, start + 1000)
                ),
            )
        cached = widget.timeline._overview_cache[(1000, 2000)]

        def unexpected_read(*args: Any, **kwargs: Any) -> Any:
            """Fail if returning to a cached interval touches sample storage.

            Args:
                *args: Positional read arguments.
                **kwargs: Keyword read arguments.

            Raises:
                AssertionError: Always, because cached navigation needs no I/O.
            """
            raise AssertionError("Cached overview performed a sample read")

        monkeypatch.setattr(source, "read_peaks", unexpected_read)
        monkeypatch.setattr(source, "read_window", unexpected_read)
        monkeypatch.setattr(source, "read_overview", unexpected_read)
        widget.timeline.set_visible_range(1000, 2000)
        assert widget.timeline._overview is cached
        assert widget.timeline._overview_range == (1000, 2000)
        assert not widget._timeline_timer.isActive()
        assert "timeline" not in widget._tasks._pending
        QTest.qWait(150)
        assert "Error:" not in widget.status.text()
        assert widget.timeline.selection == selected
        assert widget.tabs.widget(0)._overview is panel
    finally:
        widget.close()
        app.processEvents()


def test_region_inspection_and_axis_controls(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Picking regions and changing axes must preserve metadata and selection.

    Args:
        app: Qt application.
        tmp_path: Temporary store directory.
        monkeypatch: Guard against display-only sample reads.
    """
    path = tmp_path / "regions.zarr"
    store = SigMFZarrStore.create(path)
    recording = store.recordings.open(
        "signal",
        batched=False,
        sample_dtype="float32",
        sample_shape=(4000,),
        sample_axes=("time",),
        global_metadata={"core:sample_rate": 1000},
        captures=[
            {"core:sample_start": 0, "core:datetime": "2025-01-01T00:00:00Z"},
            {
                "core:sample_start": 2000,
                "core:datetime": "2025-01-01T00:01:00Z",
            },
        ],
        annotations=[
            {
                "core:sample_start": 500,
                "core:sample_count": 150,
                "core:label": "Burst",
            }
        ],
    )
    recording.set_samples(np.ones(4000, dtype=np.float32))
    source = DatasetSource(path)
    widget = DatasetViewer(source)
    widget.resize(1200, 950)
    widget.show()
    try:
        wait_for(
            app, lambda: "signal | Samples [0, 4000)" in widget.status.text()
        )
        assert widget.region_browser.listing.count() == 3
        panel = widget.tabs.widget(0)
        assert len(panel.figure.axes[0].images) == 2
        panel.canvas.draw()
        x, y = panel.figure.axes[0].transData.transform((575, 0))
        point = QPoint(
            round(x / panel.canvas.device_pixel_ratio),
            round(panel.canvas.height() - y / panel.canvas.device_pixel_ratio),
        )
        QTest.mouseClick(panel.canvas, Qt.MouseButton.LeftButton, pos=point)
        assert widget.tabs.currentWidget() is panel
        QTest.mouseDClick(panel.canvas, Qt.MouseButton.LeftButton, pos=point)
        assert widget.tabs.currentWidget() is widget.region_browser
        assert "Burst" in widget.region_browser.metadata.toPlainText()
        assert widget.timeline.selection == (0, 4000)
        widget.tabs.setCurrentWidget(panel)

        def no_reads(*args: Any, **kwargs: Any) -> Any:
            """Fail if display settings trigger sample reads."""
            raise AssertionError("Display settings read samples")

        with monkeypatch.context() as patch:
            patch.setattr(source, "read_window", no_reads)
            patch.setattr(source, "read_overview", no_reads)
            patch.setattr(source, "read_peaks", no_reads)
            widget.axis_mode.setCurrentIndex(2)
            widget.show_captures.setChecked(False)
            assert widget.timeline.selection == (0, 4000)
            assert len(panel.figure.axes[0].images) == 2
            assert panel.figure.axes[0].images[1].get_extent()[0] > 50
            assert all(
                region.kind == "annotation"
                for region in widget.timeline._regions
            )
            widget.show_annotations.setChecked(False)
            assert not panel.figure.axes[0].patches
            widget.show_annotations.setChecked(True)
            # The annotation band remains in stored sample coordinates.
            point = QPoint(
                round(widget.timeline._pixel(575)),
                int(widget.timeline._track().top()) + 17,
            )
            QTest.mouseClick(
                widget.timeline, Qt.MouseButton.LeftButton, pos=point
            )
            assert widget.tabs.currentWidget() is panel
            QTest.mouseDClick(
                widget.timeline, Qt.MouseButton.LeftButton, pos=point
            )
            assert widget.tabs.currentWidget() is widget.region_browser
            assert (
                "recording/annotations/0"
                in widget.region_browser.metadata.toPlainText()
            )
            assert "Burst" in widget.region_browser.metadata.toPlainText()
        widget.region_browser.select_interval.click()
        wait_for(app, lambda: "Samples [500, 650)" in widget.status.text())
        assert widget.timeline.selection == (500, 650)
        assert widget.tabs.currentWidget() is panel
        widget.tree.setCurrentItem(widget.tree.topLevelItem(1))
        assert widget.region_browser.listing.count() == 0
        assert not widget.timeline._regions
    finally:
        widget.close()
        app.processEvents()
