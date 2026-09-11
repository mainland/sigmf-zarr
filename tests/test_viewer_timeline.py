"""Reusable range gestures and full-signal viewer navigation."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
pytest.importorskip("jmespath")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sigmf_zarr.store import SigMFZarrStore
from sigmf_zarr.viewer import DatasetSource
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
