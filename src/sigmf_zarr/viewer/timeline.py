"""Reusable integer range selector with an optional amplitude overview.

This Qt widget has no dependency on recordings, storage, or Matplotlib.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QScrollBar, QWidget

from sigmf_zarr.viewer.regions import Region


class RangeSelector(QWidget):
    """Select a half-open integer interval within a fixed extent.

    Drag an edge to resize or the interior to move. Click outside to
    center the interval there. Arrow keys pan, Shift+arrows resize the
    right edge. The mouse wheel and +/- zoom the visible interval
    without changing the selection. Home fits the recording in the
    overview. The scrollbar pans the overview. Integer coordinates are
    not limited to Qt's 32-bit sliders.
    """

    # Python integer payloads retain sample positions beyond Qt's 32-bit int.
    range_changed = Signal(object, object)
    """Emit Python integer start and exclusive stop after a range change."""

    visible_range_changed = Signal(object, object)
    """Emit the visible integer interval when overview navigation changes."""

    region_selected = Signal(object)
    """Emit a source region double-clicked in one of the overview bands."""

    _regions: tuple[Region, ...]
    """Visible source regions with sample-axis bounds."""

    scrollbar: QScrollBar
    """Normalized overview navigation, independent of the selected range."""

    _visible: tuple[int, int]
    """Source interval mapped to the overview track."""

    _overview_range: tuple[int, int]
    """Source interval represented by the installed peak magnitudes."""

    _length: int
    """Total integer extent, or zero when empty."""

    _start: int
    """Inclusive selected coordinate."""

    _stop: int
    """Exclusive selected coordinate."""

    _overview: npt.NDArray[np.float64]
    """Detached normalized peak magnitudes across the overview interval."""

    _overview_cache: OrderedDict[tuple[int, int], npt.NDArray[np.float64]]
    """Recently used detail envelopes, owned until eviction or extent reset."""

    _full_overview: npt.NDArray[np.float64]
    """Full-extent envelope retained independently as a coarse preview."""

    _cache_size: int
    """Maximum number of retained detail envelopes."""

    _drag: str | None
    """Active drag mode: left edge, right edge, or interior."""

    _anchor: int
    """Pointer coordinate at drag start."""

    _original: tuple[int, int]
    """Range at drag start, used to prevent accumulated rounding error."""

    def __init__(
        self, parent: QWidget | None = None, *, cache_size: int = 128
    ) -> None:
        """Create an empty selector.

        Args:
            parent: Optional Qt parent.
            cache_size: Number of recent detail envelopes to retain. Each
                envelope uses at most 32 KiB of array storage. Zero disables
                detail retention. The full-extent envelope is always retained.

        Raises:
            ValueError: If cache_size is not a nonnegative integer.
        """
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("Cache size must be a nonnegative integer")
        super().__init__(parent)
        self._length = self._start = self._stop = self._anchor = 0
        self._overview = np.empty(0)
        self._regions = ()
        self._full_overview = np.empty(0)
        self._overview_cache = OrderedDict()
        self._cache_size = cache_size
        self._visible = self._overview_range = (0, 0)
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self.scrollbar.setAccessibleName("Overview position")
        self.scrollbar.valueChanged.connect(self._scroll)
        self._drag = None
        self._original = (0, 0)
        self.setMinimumHeight(100)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Sample range and overview")
        self.setToolTip(
            "Drag edges to resize; drag inside to move. "
            "Click outside to center. "
            "Wheel or +/-: zoom overview. Scrollbar: pan overview. "
            "Arrows: move selection. Shift+arrows: resize. "
            "Home: fit recording."
        )

    @property
    def visible_range(self) -> tuple[int, int]:
        """Return the source interval displayed by the overview.

        Returns:
            Inclusive start and exclusive stop, or (0, 0) when empty.
        """
        return self._visible

    def set_visible_range(self, start: int, stop: int) -> None:
        """Navigate the overview without moving the selected interval.

        Args:
            start: Inclusive source coordinate.
            stop: Exclusive source coordinate.

        Raises:
            ValueError: If the interval is empty or outside the extent.
        """
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= self._length
        ):
            raise ValueError("Visible range must be within the extent")
        if self._visible != (start, stop):
            self._visible = (start, stop)
            self._drag = None
            self._restore_overview()
            self._sync_scrollbar()
            self.update()
            self.visible_range_changed.emit(start, stop)

    def fit_view(self) -> None:
        """Display the full recording without changing the selection."""
        if self._length:
            self.set_visible_range(0, self._length)

    def zoom_to_selection(self) -> None:
        """Display the selected interval across the overview track."""
        if self._length:
            self.set_visible_range(*self.selection)

    def _sync_scrollbar(self) -> None:
        """Map the integer viewport onto a bounded Qt scrollbar."""
        start, stop = self._visible
        duration = stop - start
        remaining = self._length - duration
        # Qt stores scrollbar values as bounded integers. Normalize only the
        # control; keep source positions exact and round to the nearest step.
        maximum = min(1_000_000, remaining)
        self.scrollbar.blockSignals(True)
        self.scrollbar.setRange(0, maximum)
        self.scrollbar.setPageStep(
            min(1_000_000, max(1, duration * maximum // remaining))
            if remaining
            else 1
        )
        self.scrollbar.setSingleStep(max(1, self.scrollbar.pageStep() // 20))
        self.scrollbar.setValue(
            (start * maximum + remaining // 2) // remaining if remaining else 0
        )
        self.scrollbar.blockSignals(False)

    def _scroll(self, value: int) -> None:
        """Pan the visible interval using a normalized scrollbar position.

        Args:
            value: Qt scrollbar position.
        """
        duration = self._visible[1] - self._visible[0]
        maximum = self.scrollbar.maximum()
        if maximum:
            start = (
                value * (self._length - duration) + maximum // 2
            ) // maximum
            self.set_visible_range(start, start + duration)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Position the scrollbar below the painted track.

        Args:
            event: Qt resize event.
        """
        height = self.scrollbar.sizeHint().height()
        self.scrollbar.setGeometry(
            12, self.height() - height, max(1, self.width() - 24), height
        )
        super().resizeEvent(event)

    @property
    def extent(self) -> int:
        """Return the full selectable extent.

        Returns:
            Number of integer positions in the domain.
        """
        return self._length

    @property
    def selection(self) -> tuple[int, int]:
        """Return the selected half-open interval.

        Returns:
            Inclusive start and exclusive stop.
        """
        return self._start, self._stop

    def set_extent(self, length: int) -> None:
        """Reset the domain, clear its overview, and select the full extent.

        Args:
            length: Nonnegative integer extent.

        Raises:
            ValueError: If length is not a nonnegative Python integer.
        """
        if type(length) is not int or length < 0:
            raise ValueError("Extent must be a nonnegative integer")
        changed = self.selection != (0, length)
        visible_changed = self._visible != (0, length)
        self._length = self._stop = length
        self._start = 0
        self._visible = self._overview_range = (0, length)
        self._sync_scrollbar()
        self._overview = np.empty(0)
        self._full_overview = np.empty(0)
        self._overview_cache.clear()
        self._regions = ()
        self._drag = None
        self.setEnabled(length > 0)
        self.update()
        if changed:
            self.range_changed.emit(0, length)
        if visible_changed:
            self.visible_range_changed.emit(0, length)

    def set_range(self, start: int, stop: int) -> None:
        """Select an interval and notify consumers if it changes.

        Args:
            start: Inclusive integer coordinate.
            stop: Exclusive integer coordinate, greater than start.

        Raises:
            ValueError: If the interval is empty or outside the extent.
        """
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= self._length
        ):
            raise ValueError("Range must be nonempty and within the extent")
        if self.selection != (start, stop):
            self._start, self._stop = start, stop
            self.update()
            self.range_changed.emit(start, stop)

    def reveal_selection(self) -> None:
        """Pan or expand the overview only when the selection is outside it."""
        lo, hi = self.visible_range
        start, stop = self.selection
        if start < lo or stop > hi:
            duration = max(hi - lo, stop - start)
            left = min(start, max(lo, stop - duration))
            left = min(left, self.extent - duration)
            self.set_visible_range(left, left + duration)

    def fit(self) -> None:
        """Select the full extent when nonempty."""
        if self._length:
            self.set_range(0, self._length)

    def set_overview(
        self, peaks: npt.ArrayLike, *, start: int = 0, stop: int | None = None
    ) -> None:
        """Cache uniformly spaced magnitudes and refresh the visible overview.

        The full-extent summary is retained separately. Recent detail summaries
        use bounded least-recently-used retention. Resetting the extent clears
        both caches, even if the new extent has the same length.

        Args:
            peaks: Up to 4096 nonnegative values. Nonfinite values are blank.
            start: Inclusive source coordinate represented by the peaks.
            stop: Exclusive source coordinate, defaulting to the full extent.

        Raises:
            ValueError: If the input is not a bounded nonnegative vector.
        """
        values = np.array(peaks, dtype=float, copy=True)
        stop = self._length if stop is None else stop
        if not 0 <= start <= stop <= self._length or (
            values.size and start == stop
        ):
            raise ValueError("Overview interval must be within the extent")
        if values.ndim != 1 or len(values) > 4096 or np.any(values < 0):
            raise ValueError(
                "Overview must be a nonnegative vector of at most 4096 values"
            )
        values[~np.isfinite(values)] = 0
        maximum = values.max() if values.size else 0
        values = values / maximum if maximum else values
        values.setflags(write=False)
        interval = (start, stop)
        if interval == (0, self._length):
            # Pin a coarse fallback outside the detail LRU so eviction cannot
            # blank the overview while a new visible interval is being read.
            self._full_overview = values
        elif self._cache_size and values.size:
            self._overview_cache[interval] = values
            self._overview_cache.move_to_end(interval)
            while len(self._overview_cache) > self._cache_size:
                self._overview_cache.popitem(last=False)
        else:
            self._overview_cache.pop(interval, None)
        self._overview = values
        self._overview_range = interval
        self._restore_overview()
        self.update()

    def has_overview(self, start: int, stop: int) -> bool:
        """Report whether an exact interval summary is available.

        Args:
            start: Inclusive source coordinate.
            stop: Exclusive source coordinate.

        Returns:
            Whether the active envelope or retained cache covers this exact
            interval. A coarser preview does not count as a cache hit.
        """
        interval = (start, stop)
        return (
            (interval == self._overview_range and bool(self._overview.size))
            or (
                interval == (0, self._length)
                and bool(self._full_overview.size)
            )
            or interval in self._overview_cache
        )

    def _restore_overview(self) -> None:
        """Display cached detail or the finest available covering preview."""
        start, stop = self._visible
        if self._visible in self._overview_cache:
            self._overview_cache.move_to_end(self._visible)
            self._overview = self._overview_cache[self._visible]
            self._overview_range = self._visible
            return
        candidates = [
            (interval, peaks)
            for interval, peaks in self._overview_cache.items()
            if interval[0] <= start < stop <= interval[1] and peaks.size
        ]
        if self._full_overview.size:
            candidates.append(((0, self._length), self._full_overview))
        lo, hi = self._overview_range
        if lo <= start < stop <= hi and self._overview.size:
            candidates.append((self._overview_range, self._overview))
        if candidates:
            # Compare source samples per bin, not just interval lengths: a
            # wider envelope may provide finer detail if it has more bins.
            interval, peaks = min(
                candidates,
                key=lambda entry: (entry[0][1] - entry[0][0]) / entry[1].size,
            )
            self._overview_range, self._overview = interval, peaks
            if interval in self._overview_cache:
                self._overview_cache.move_to_end(interval)
        else:
            self._overview = np.empty(0)
            self._overview_range = (0, 0)

    def _track(self) -> QRectF:
        """Return the drawing and interaction rectangle.

        Returns:
            Timeline bounds in widget coordinates.
        """
        return QRectF(
            12,
            8,
            max(1, self.width() - 24),
            max(1, self.height() - self.scrollbar.sizeHint().height() - 16),
        )

    def _coordinate(self, x: float) -> int:
        """Convert a widget position to a clamped integer coordinate.

        Args:
            x: Horizontal widget position.

        Returns:
            Coordinate in the visible interval.
        """
        track = self._track()
        start, stop = self._visible
        return start + min(
            stop - start,
            max(0, round((x - track.left()) / track.width() * (stop - start))),
        )

    def _pixel(self, coordinate: float) -> float:
        """Map a source coordinate to the track, including offscreen values.

        Args:
            coordinate: Absolute source position.

        Returns:
            Horizontal widget position.
        """
        start, stop = self._visible
        track = self._track()
        return (
            track.left()
            + (coordinate - start) / max(1, stop - start) * track.width()
        )

    def _move(self, start: int) -> None:
        """Move the selection while preserving its duration.

        Args:
            start: Requested start, clamped at both ends.
        """
        duration = self._stop - self._start
        start = min(max(0, start), self._length - duration)
        self.set_range(start, start + duration)

    def _zoom(self, inward: bool, anchor: int | None = None) -> None:
        """Zoom the overview around a coordinate or its midpoint.

        Args:
            inward: Whether to halve rather than double the interval.
            anchor: Optional source coordinate under the pointer.
        """
        if not self._length:
            return
        lo, hi = self._visible
        old = hi - lo
        new = max(1, old // 2) if inward else min(self._length, old * 2)
        center = (lo + hi) // 2 if anchor is None else anchor
        center = min(hi, max(lo, center))
        # Keep the anchor at the same fractional position in the viewport.
        # Clamping at the recording ends may override that position.
        start = center - (center - lo) * new // old
        start = min(max(0, start), self._length - new)
        self.set_visible_range(start, start + new)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the overview and range handles.

        Args:
            event: Qt paint event.
        """
        painter = QPainter(self)
        track = self._track()
        painter.fillRect(track, self.palette().base())
        painter.setPen(self.palette().mid().color())
        painter.drawRect(track)
        if not self._length:
            return
        painter.save()
        painter.setClipRect(track)
        painter.setPen(self.palette().text().color())
        begin, end = self._overview_range
        for index, value in enumerate(self._overview):
            left = self._pixel(
                begin + index * (end - begin) // len(self._overview)
            )
            right = self._pixel(
                begin + (index + 1) * (end - begin) // len(self._overview)
            )
            if right < track.left() or left > track.right():
                continue
            left, right = max(track.left(), left), min(track.right(), right)
            height = float(value) * (track.height() / 2 - 3)
            painter.fillRect(
                QRectF(
                    left,
                    track.center().y() - height,
                    max(1, right - left),
                    2 * height,
                ),
                self.palette().text(),
            )
        left = self._pixel(self._start)
        right = self._pixel(self._stop)
        color = self.palette().highlight().color()
        tint = QColor(color)
        tint.setAlpha(45)
        selected = QRectF(left, track.top(), right - left, track.height())
        painter.fillRect(selected, tint)
        painter.setPen(QPen(color, 2))
        painter.drawRect(selected)
        for x in (left, right):
            painter.fillRect(
                QRectF(x - 4, track.top(), 8, track.height()), color
            )
        painter.restore()
        self._paint_regions(painter)
        if self.hasFocus():
            painter.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
            painter.drawRect(track.adjusted(-4, -4, 4, 4))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Begin resizing or moving, or center on an outside click.

        Args:
            event: Qt mouse event.
        """
        if event.button() != Qt.MouseButton.LeftButton or not self._length:
            return
        self.setFocus()
        x = event.position().x()
        if not self._track().contains(event.position()):
            return
        if self._region_at(event.position()) is not None:
            # Qt delivers a press before a double-click. Consume the first
            # press so inspecting a region cannot also move the selection.
            return
        left = self._pixel(self._start)
        right = self._pixel(self._stop)
        self._anchor = self._coordinate(x)
        self._original = self.selection
        if min(abs(x - left), abs(x - right)) <= 8:
            self._drag = "left" if abs(x - left) < abs(x - right) else "right"
        elif left < x < right:
            self._drag = "move"
        else:
            self._move(self._anchor - (self._stop - self._start) // 2)
            self._original = self.selection
            self._drag = "move"

    def _region_at(self, point: QPointF) -> Region | None:
        """Find the topmost overview region under a pointer position.

        Args:
            point: Widget-relative pointer coordinates.

        Returns:
            Matching source region, or None outside the region bands.
        """
        if not self._length or not self._track().contains(point):
            return None
        position = self._coordinate(point.x())
        row = int((point.y() - self._track().top()) // 12)
        for region in reversed(self._regions):
            if row == (
                0 if region.kind == "capture" else 1
            ) and region.intersects("sample", position, position + 1):
                return region
        return None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Inspect an overview region without changing the selected interval.

        Args:
            event: Qt double-click event.
        """
        region = self._region_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and region is not None:
            self._drag = None
            self.region_selected.emit(region)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Apply a drag using the original interval as its reference.

        Args:
            event: Qt mouse event.
        """
        position = self._coordinate(event.position().x())
        if self._drag == "left":
            self.set_range(min(position, self._stop - 1), self._stop)
        elif self._drag == "right":
            self.set_range(self._start, max(position, self._start + 1))
        elif self._drag == "move":
            self._move(self._original[0] + position - self._anchor)
        else:
            track = self._track()
            near = min(abs(position - self._start), abs(position - self._stop))
            self.setCursor(
                Qt.CursorShape.SizeHorCursor
                if self._length
                and near
                / (self._visible[1] - self._visible[0])
                * track.width()
                <= 8
                else Qt.CursorShape.OpenHandCursor
            )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finish a pointer drag.

        Args:
            event: Qt mouse event.
        """
        self._drag = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Support keyboard range navigation.

        Args:
            event: Qt key event.
        """
        if not self._length:
            return
        key = event.key()
        if key == Qt.Key.Key_Home:
            self.fit_view()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
            self._zoom(key != Qt.Key.Key_Minus)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = max(1, (self._stop - self._start) // 20)
            step *= -1 if key == Qt.Key.Key_Left else 1
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.set_range(
                    self._start,
                    min(self._length, max(self._start + 1, self._stop + step)),
                )
            else:
                self._move(self._start + step)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom around the pointer.

        Args:
            event: Qt wheel event.
        """
        if event.angleDelta().y():
            self._zoom(
                event.angleDelta().y() > 0,
                self._coordinate(event.position().x()),
            )
            event.accept()

    def set_regions(self, regions: tuple[Region, ...]) -> None:
        """Display source regions in two pickable overview bands.

        Args:
            regions: Regions with integer sample bounds. Capture categories
                use the first band and other categories use the second.
        """
        self._regions = tuple(r for r in regions if "sample" in r.bounds)
        self.update()

    def _paint_regions(self, painter: QPainter) -> None:
        """Render regions clipped to the visible sample interval.

        Args:
            painter: Active widget painter.
        """
        track = self._track()
        start, stop = self.visible_range
        for region in self._regions:
            span = region.bounds["sample"]
            lo, hi = max(start, span.start), min(stop, span.stop)
            if lo >= hi:
                continue
            capture = region.kind == "capture"
            color = QColor("#1976d2" if capture else "#ef6c00")
            color.setAlpha(160)
            x = self._pixel(lo)
            painter.fillRect(
                QRectF(
                    x,
                    track.top() + (0 if capture else 12),
                    max(1, self._pixel(hi) - x),
                    10,
                ),
                color,
            )
            if start <= span.start < stop:
                painter.setPen(QColor("white"))
                top = int(track.top() + (0 if capture else 12))
                painter.drawLine(int(x), top, int(x), top + 10)
            if capture and self._pixel(hi) - x > 60:
                painter.save()
                font = painter.font()
                font.setPointSize(7)
                painter.setFont(font)
                painter.setPen(QColor("white"))
                painter.drawText(
                    QRectF(x + 3, track.top(), self._pixel(hi) - x - 3, 10),
                    Qt.AlignmentFlag.AlignVCenter,
                    region.label,
                )
                painter.restore()
