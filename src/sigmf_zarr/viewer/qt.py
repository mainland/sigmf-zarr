"""Embeddable Qt record browser and Matplotlib panels.

This module requires the ``viewer`` extra. Data access, filtering, and
numerical metrics remain usable without importing this module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from html import escape
from threading import Event
from typing import cast

import numpy as np
from PySide6.QtCore import QPoint, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

# Import PySide6 before the Qt Matplotlib backend to select the same binding.
from matplotlib.backends.backend_qtagg import (  # isort: skip
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.backend_bases import Event as MatplotlibEvent
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.figure import Figure

from sigmf_zarr.viewer.data import (
    OverviewWindow,
    SampleWindow,
)
from sigmf_zarr.viewer.overlays import RegionPatch
from sigmf_zarr.viewer.plots import (
    overview_spectrogram,
    spectrogram,
)
from sigmf_zarr.viewer.presentation import (
    PlotOptions,
)
from sigmf_zarr.viewer.windows import WINDOW_LABELS, WindowFunction

Metric = Callable[[SampleWindow], float | str]
"""Worker-thread function calculating a metric for the selected window."""

Draw = Callable[[Figure, SampleWindow], None]
"""GUI-thread function drawing a selected window into a cleared figure."""


def _metric_values(
    view: SampleWindow | OverviewWindow,
    metrics: Mapping[str, Metric],
    custom: bool,
    stop: Event,
) -> dict[str, float | str]:
    """Evaluate raw callbacks or use metrics reduced over every source sample.

    Args:
        view: Raw samples or explicit reduced data.
        metrics: Registered raw-sample callbacks.
        custom: Whether callbacks replace the built-in metrics.
        stop: Cooperative cancellation flag.

    Returns:
        Metric values or per-metric explanations.
    """
    if isinstance(view, OverviewWindow):
        if custom:
            return {
                title: "Zoom in for raw-sample metrics" for title in metrics
            }
        return {
            "Mean power": view.signal.mean_power,
            "RMS": float(np.sqrt(view.signal.mean_power)),
            "Peak magnitude": view.signal.peak_magnitude,
        }
    results: dict[str, float | str] = {}
    for title, metric in metrics.items():
        if stop.is_set():
            break
        try:
            results[title] = metric(view)
        except Exception as exc:
            results[title] = f"Error: {exc}"
    return results


class PlotToolbar(NavigationToolbar2QT):
    """Navigation toolbar announcing completed user gestures."""

    navigated = Signal()
    """Emit after pan, zoom, or navigation history changes the view."""

    def release_pan(self, event: MatplotlibEvent) -> None:
        """Complete a pan and notify listeners.

        Args:
            event: Mouse release event.
        """
        super().release_pan(event)
        self.navigated.emit()

    def release_zoom(self, event: MatplotlibEvent) -> None:
        """Complete a rectangle zoom and notify listeners.

        Args:
            event: Mouse release event.
        """
        super().release_zoom(event)
        self.navigated.emit()

    def home(self, *args: object) -> None:
        """Restore the initial view and notify listeners.

        Args:
            *args: Unused Qt action arguments.
        """
        super().home(*args)
        self.navigated.emit()

    def back(self, *args: object) -> None:
        """Restore the previous view and notify listeners.

        Args:
            *args: Unused Qt action arguments.
        """
        super().back(*args)
        self.navigated.emit()

    def forward(self, *args: object) -> None:
        """Restore the next view and notify listeners.

        Args:
            *args: Unused Qt action arguments.
        """
        super().forward(*args)
        self.navigated.emit()


class MatplotlibPanel(QWidget):
    """Reusable figure, canvas, and toolbar for a drawing function."""

    figure: Figure
    """Owned figure, available to embedding applications."""

    canvas: FigureCanvasQTAgg
    """Qt canvas displaying the figure."""

    toolbar: PlotToolbar
    """Plot navigation controls and view history."""

    _draw: Draw
    """Drawing callback supplied by the consumer."""

    _draw_overview: Callable[[Figure, OverviewWindow], None] | None
    """Optional drawing function for explicit reduced data."""

    region_picked = Signal(str)
    """Emit a region source identifier after an overlay is double-clicked."""

    time_navigated = Signal(object)
    """Emit horizontal limits after a completed user navigation gesture."""

    _display_xlim: tuple[float, float] | None
    """Horizontal limits before the next user gesture."""

    _home_view: tuple[tuple[float, float], tuple[float, float]] | None
    """Unzoomed plot limits retained during linked sample navigation."""

    _last_view: SampleWindow | OverviewWindow | None
    """Cached data for display-only redraws."""

    def __init__(
        self,
        draw: Draw,
        parent: QWidget | None = None,
        *,
        draw_overview: Callable[[Figure, OverviewWindow], None] | None = None,
    ) -> None:
        """Create a panel without reading any dataset content.

        Args:
            draw: Function drawing into a cleared figure.
            parent: Optional Qt parent.
            draw_overview: Optional drawing function for reduced intervals.
        """
        super().__init__(parent)
        self._draw = draw
        self._last_view = None
        self._display_xlim = None
        self._home_view = None
        self._draw_overview = draw_overview
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("pick_event", self._pick_region)
        self.canvas.mpl_connect("motion_notify_event", self._hover_region)
        self.canvas.mpl_connect("figure_leave_event", self._hover_region)
        layout = QVBoxLayout(self)
        self.toolbar = PlotToolbar(self.canvas, self)
        self.toolbar.navigated.connect(self._navigated)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def set_window(self, window: SampleWindow | None) -> None:
        """Display a window or clear the panel during selection changes.

        Args:
            window: Selected data, or None to clear the panel.
        """
        self._last_view = window
        self.figure.clear()
        if window is not None:
            self._draw(self.figure, window)
        self._remember_view()
        self.canvas.draw_idle()

    def set_overview(self, window: OverviewWindow) -> None:
        """Draw explicit reduced data or explain that raw samples are needed.

        Args:
            window: Reduced interval and metadata.
        """
        self._last_view = window
        self.figure.clear()
        if self._draw_overview is not None:
            self._draw_overview(self.figure, window)
        else:
            self.figure.subplots().text(
                0.5, 0.5, "Zoom in to display raw samples", ha="center"
            )
        self._remember_view()
        self.canvas.draw_idle()

    def redraw(self) -> None:
        """Redraw cached data after coordinate or overlay settings change."""
        if isinstance(self._last_view, OverviewWindow):
            self.set_overview(self._last_view)
        else:
            self.set_window(self._last_view)

    @contextmanager
    def preserve_view(
        self,
        *,
        enabled: bool = True,
        keep_home: bool = True,
        horizontal: tuple[float, float] | None = None,
    ) -> Iterator[None]:
        """Keep plot limits across a redraw in the same coordinate system.

        Navigation history is rebuilt from the Home view and restored zoom.
        Colorbar axes retain the limits calculated by the drawing function.

        Args:
            enabled: Whether to restore the first plot's limits.
            keep_home: Retain the original Home view during linked navigation.
            horizontal: Optional limits shared by linked time plots.

        Yields:
            Control to the drawing operation.
        """
        limits = None
        home = self._home_view if keep_home else None
        if enabled and self.figure.axes:
            axes = self.figure.axes[0]
            limits = horizontal or axes.get_xlim(), axes.get_ylim()
        yield
        if limits is not None and self.figure.axes:
            # Redrawing replaces the Axes objects referenced by toolbar
            # history. Rebuild Home and the current zoom against the new axes.
            self.toolbar.update()
            axes = self.figure.axes[0]
            if home is not None:
                axes.set_xlim(home[0])
                axes.set_ylim(home[1])
                self._home_view = home
            self.toolbar.push_current()
            axes.set_xlim(limits[0])
            axes.set_ylim(limits[1])
            self.toolbar.push_current()
            self._display_xlim = axes.get_xlim()
            self.canvas.draw_idle()

    def _remember_view(self) -> None:
        """Record fresh plot limits without emitting a navigation event."""
        self.toolbar.update()
        if self.figure.axes:
            axes = self.figure.axes[0]
            self._display_xlim = axes.get_xlim()
            self._home_view = axes.get_xlim(), axes.get_ylim()
            self.toolbar.push_current()
        else:
            self._display_xlim = None
            self._home_view = None
        QToolTip.hideText()

    def _navigated(self) -> None:
        """Report time navigation without reacting to drawing callbacks."""
        if not self.figure.axes:
            return
        limits = self.figure.axes[0].get_xlim()
        # Only a completed horizontal gesture changes the sample selection.
        # Frequency-only zoom must not trigger a new sample read.
        if limits != self._display_xlim:
            self._display_xlim = limits
            self.time_navigated.emit(limits)

    def _hover_region(self, event: MouseEvent) -> None:
        """Show original region JSON when the pointer is inside an overlay.

        Args:
            event: Canvas pointer event.
        """
        if (
            event.name != "figure_leave_event"
            and event.inaxes
            and not self.toolbar.mode
        ):
            for patch in reversed(event.inaxes.patches):
                if isinstance(patch, RegionPatch) and patch.contains(event)[0]:
                    metadata = json.dumps(
                        patch.region.metadata, indent=2, ensure_ascii=True
                    )
                    # Matplotlib events use physical pixels from the bottom
                    # left; Qt uses logical pixels from the top left.
                    point = self.canvas.mapToGlobal(
                        QPoint(
                            int(event.x / self.canvas.device_pixel_ratio),
                            self.canvas.height()
                            - int(event.y / self.canvas.device_pixel_ratio),
                        )
                    )
                    QToolTip.showText(
                        point,
                        "<pre>" + escape(metadata) + "</pre>",
                        self.canvas,
                    )
                    return
        QToolTip.hideText()

    def _pick_region(self, event: PickEvent) -> None:
        """Inspect regions only after an intentional left-button double-click.

        Args:
            event: Matplotlib artist pick event.
        """
        key = event.artist.get_gid()
        if (
            isinstance(key, str)
            and event.mouseevent.dblclick
            and event.mouseevent.button == MouseButton.LEFT
            and not self.toolbar.mode
        ):
            self.region_picked.emit(key)


class SpectrogramPanel(MatplotlibPanel):
    """Reusable spectrogram with local FFT and color-scale controls."""

    analysis_changed = Signal()
    """Request a new overview after changing FFT length or overlap."""

    _overview: OverviewWindow | None
    """Last reduced interval, if the selection exceeds the raw sample limit."""

    _analysis_pending: bool
    """Preserve the view when replacement FFT powers arrive."""

    _frequency_source: tuple[object, ...] | None
    """Source identity associated with retained frequency limits."""

    _frequency_limits: tuple[float, float] | None
    """Frequency zoom retained when the scrubber changes sample intervals."""

    fft_size: QComboBox
    """Requested FFT size, or automatic sizing for the selected window."""

    overlap: QComboBox
    """Fraction of overlap between consecutive FFT frames."""

    window_function: QComboBox
    """Periodic FFT window selected for spectral analysis."""

    dynamic_range: QDoubleSpinBox
    """Color range below the maximum, in dB."""

    colormap: QComboBox
    """Selected standard colormap applied to raw and reduced views."""

    maximum_db: QDoubleSpinBox
    """Fixed color maximum when automatic scaling is disabled."""

    auto_level: QCheckBox
    """Whether the color maximum follows the selected window's peak."""

    _window: SampleWindow | None
    """Last selected window, reused when display settings change."""

    options: PlotOptions
    """Coordinate mapping and overlay visibility shared with other panels."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        options: PlotOptions | None = None,
    ) -> None:
        """Create a spectrogram without accessing storage.

        Args:
            parent: Optional Qt parent.
            options: Shared sample-axis and region visibility settings.
        """
        self._window = None
        self.options = options or PlotOptions()
        self._overview = None
        self._analysis_pending = False
        self._frequency_source = None
        self._frequency_limits = None
        super().__init__(self._draw_spectrogram, parent)
        controls = QHBoxLayout()
        self.fft_size = QComboBox()
        self.fft_size.addItems(
            [
                "Auto",
                "32",
                "64",
                "128",
                "256",
                "512",
                "1024",
                "2048",
                "4096",
                "8192",
            ]
        )
        self.overlap = QComboBox()
        self.overlap.addItems(["0%", "50%", "75%", "87.5%"])
        self.overlap.setCurrentText("50%")
        self.window_function = QComboBox()
        for value, name in WINDOW_LABELS.items():
            self.window_function.addItem(name, value)
        self.colormap = QComboBox()
        self.colormap.addItems(
            ["magma", "viridis", "inferno", "cividis", "gray"]
        )
        self.colormap.setCurrentText("magma")
        self.colormap.setMinimumContentsLength(10)
        self.colormap.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.dynamic_range = QDoubleSpinBox()
        self.dynamic_range.setRange(1, 300)
        self.dynamic_range.setValue(80)
        self.dynamic_range.setSuffix(" dB")
        self.maximum_db = QDoubleSpinBox()
        self.maximum_db.setRange(-300, 300)
        self.maximum_db.setSuffix(" dB")
        self.auto_level = QCheckBox("Auto level")
        self.auto_level.setChecked(True)
        for label, widget in (
            ("FFT", self.fft_size),
            ("Overlap", self.overlap),
            ("Window", self.window_function),
        ):
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addStretch()
        colors = QHBoxLayout()
        for label, color_widget in (
            ("Colormap", self.colormap),
            ("Range", self.dynamic_range),
            ("Maximum", self.maximum_db),
        ):
            colors.addWidget(QLabel(label))
            colors.addWidget(color_widget)
        colors.addWidget(self.auto_level)
        colors.addStretch()
        cast(QVBoxLayout, self.layout()).insertLayout(0, controls)
        cast(QVBoxLayout, self.layout()).insertLayout(1, colors)
        self.fft_size.currentTextChanged.connect(self._analysis_refresh)
        self.overlap.currentTextChanged.connect(self._analysis_refresh)
        self.window_function.currentTextChanged.connect(self._analysis_refresh)
        self.colormap.currentTextChanged.connect(self._colormap_changed)
        self.dynamic_range.valueChanged.connect(self._refresh)
        self.maximum_db.valueChanged.connect(self._refresh)
        self.auto_level.toggled.connect(self._refresh)
        self.maximum_db.setEnabled(False)

    def set_window(self, window: SampleWindow | None) -> None:
        """Cache the selection and redraw or clear the spectrogram.

        Args:
            window: Selected samples, or None to discard the old selection.
        """
        self._window = window
        self._overview = None
        self._analysis_pending = False
        with self._frequency_view(window):
            super().set_window(window)

    @contextmanager
    def _frequency_view(self, window: SampleWindow | None) -> Iterator[None]:
        """Retain frequency limits across sample ranges of the same signal.

        Args:
            window: Replacement window, or None while a selection loads.

        Yields:
            Control to the drawing operation.
        """
        if self.figure.axes and self._frequency_source is not None:
            self._frequency_limits = self.figure.axes[0].get_ylim()
        if window is not None:
            # Frequency zoom belongs to the signal, not its selected time
            # range. Omit start/count while distinguishing items and channels.
            key = (
                window.recording,
                window.item,
                tuple(sorted(window.coordinates.items())),
            )
            if key != self._frequency_source:
                self._frequency_limits = None
            self._frequency_source = key
        yield
        if (
            window is not None
            and self._frequency_limits is not None
            and self.figure.axes
        ):
            self.figure.axes[0].set_ylim(self._frequency_limits)

    def analysis_settings(self) -> tuple[int | None, float, WindowFunction]:
        """Return the requested analysis settings.

        Returns:
            FFT length, or None for automatic sizing, fractional overlap,
            and the periodic window name.
        """
        size = self.fft_size.currentText()
        return (
            None if size == "Auto" else int(size),
            float(self.overlap.currentText().rstrip("%")) / 100,
            cast(WindowFunction, self.window_function.currentData()),
        )

    def set_overview(self, window: OverviewWindow) -> None:
        """Display a reduced interval using its precomputed FFT powers.

        Args:
            window: Reduced interval matching the analysis settings.
        """
        self._window = None
        self._overview = window
        self._last_view = window
        with (
            self._frequency_view(window.context),
            self.preserve_view(enabled=self._analysis_pending),
        ):
            self.figure.clear()
            overview_spectrogram(
                self.figure,
                window,
                dynamic_range=self.dynamic_range.value(),
                maximum_db=None
                if self.auto_level.isChecked()
                else self.maximum_db.value(),
                cmap=self.colormap.currentText(),
                options=self.options,
            )
            self._remember_view()
        self._analysis_pending = False
        self._update_level()
        self.canvas.draw_idle()

    def _refresh(self) -> None:
        """Redraw cached samples after changing a display setting."""
        self.maximum_db.setEnabled(not self.auto_level.isChecked())
        if self._overview is not None:
            size, overlap, window_function = self.analysis_settings()
            size = size or 256
            hop = max(1, size - int(size * overlap))
            if (size, hop, window_function) != (
                self._overview.signal.fft_size,
                self._overview.signal.hop,
                self._overview.signal.window_function,
            ):
                return
            else:
                with self.preserve_view():
                    self.set_overview(self._overview)
            return
        with self.preserve_view():
            super().set_window(self._window)

    def _analysis_refresh(self) -> None:
        """Redraw raw data and notify consumers of new analysis settings."""
        if self._overview is not None:
            self._analysis_pending = True
        else:
            self._refresh()
        self.analysis_changed.emit()

    def _colormap_changed(self, name: str) -> None:
        """Recolor existing artists without recomputing data or resetting zoom.

        Args:
            name: Registered Matplotlib colormap name.
        """
        if self.figure.axes:
            axes = self.figure.axes[0]
            for artist in (*axes.images, *axes.collections):
                artist.set_cmap(name)
            self.canvas.draw_idle()

    def _draw_spectrogram(self, figure: Figure, window: SampleWindow) -> None:
        """Apply panel settings to the standalone spectrogram drawing function.

        Args:
            figure: Figure cleared by the panel.
            window: Cached sample window.
        """
        size = self.fft_size.currentText()
        spectrogram(
            figure,
            window,
            fft_size=None if size == "Auto" else int(size),
            window_function=cast(
                WindowFunction, self.window_function.currentData()
            ),
            overlap=float(self.overlap.currentText().rstrip("%")) / 100,
            dynamic_range=self.dynamic_range.value(),
            maximum_db=None
            if self.auto_level.isChecked()
            else self.maximum_db.value(),
            cmap=self.colormap.currentText(),
            options=self.options,
        )
        self._update_level()

    def _update_level(self) -> None:
        """Remember the automatic level for a later switch to fixed scaling."""
        axes = self.figure.axes[0]
        artists = list(axes.images) + list(axes.collections)
        if self.auto_level.isChecked() and artists:
            maximum = artists[0].get_clim()[1]
            self.maximum_db.blockSignals(True)
            self.maximum_db.setValue(maximum)
            self.maximum_db.blockSignals(False)


__all__ = [
    "Draw",
    "MatplotlibPanel",
    "Metric",
    "SpectrogramPanel",
]
