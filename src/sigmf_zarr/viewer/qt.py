"""Embeddable Qt record browser and Matplotlib panels.

This module requires the ``viewer`` extra. Data access, filtering, and
numerical metrics remain usable without importing this module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from html import escape
from threading import Event
from typing import cast

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QKeySequence,
    QShortcut,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
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

from sigmf_zarr.viewer._tasks import Progress, Tasks
from sigmf_zarr.viewer.captures import AxisMode, capture_segments, sample_rate
from sigmf_zarr.viewer.data import (
    DatasetCatalog,
    DatasetSource,
    OverviewWindow,
    RecordingInfo,
    SampleWindow,
)
from sigmf_zarr.viewer.overlays import RegionPatch
from sigmf_zarr.viewer.plots import (
    constellation,
    mean_power,
    overview_constellation,
    overview_spectrogram,
    overview_spectrum,
    overview_waveform,
    peak_magnitude,
    rms,
    spectrogram,
    spectrum,
    waveform,
)
from sigmf_zarr.viewer.presentation import (
    PlotOptions,
    sample_interval,
    window_regions,
)
from sigmf_zarr.viewer.region_qt import RegionBrowser
from sigmf_zarr.viewer.regions import Region
from sigmf_zarr.viewer.timeline import RangeSelector
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


class DatasetViewer(QWidget):
    """Record browser with filtering, item navigation, and replaceable panels.

    Reads and metrics run in workers. Panels update on the GUI thread. A
    consumer can register any QWidget with an update callback using
    add_panel.
    """

    window_changed: Signal = Signal(object)
    """Emit SampleWindow or None when the displayed selection changes."""

    overview_changed: Signal = Signal(object)
    """Emit OverviewWindow or None for explicitly reduced selections."""

    metadata: QPlainTextEdit
    """Optional resolved item JSON view."""

    recording_metadata: QPlainTextEdit
    """Stored recording attributes without item overrides."""

    item_metadata: QPlainTextEdit
    """Stored metadata for the selected item."""

    channel_metadata: QPlainTextEdit
    """Stored metadata for the selected explicit channel."""

    collection_metadata: QPlainTextEdit
    """Selected collection metadata and recording references."""

    indexes: QPlainTextEdit
    """Selected index values, decoded labels, and array attributes."""

    show_resolved: QCheckBox
    """Whether to expose the derived item metadata view."""

    status: QLabel
    """Current operation or error message."""

    tree: QTreeWidget
    """Expandable collection and recording navigation."""

    recording_controls: QWidget
    """Recording-specific controls, disabled for collection selections."""

    index_selection: QListWidget
    """Explicitly selected indexes for the displayed item."""

    axis_form: QFormLayout
    """Additional named-axis selectors."""

    timeline: RangeSelector
    """Reusable range selector spanning the full selected signal."""

    region_browser: RegionBrowser
    """Read-only capture and annotation inspector."""

    axis_mode: QComboBox
    """Sample index, elapsed sample time, or capture timestamp selection."""

    show_captures: QCheckBox
    """Capture overlay visibility."""

    show_annotations: QCheckBox
    """Annotation overlay visibility."""

    _plot_options: PlotOptions
    """Shared options for built-in plots."""

    _default_panels: list[MatplotlibPanel]
    """Built-in panels supporting display-only redraws."""

    _time_panels: list[MatplotlibPanel]
    """Panels whose horizontal coordinate represents sample position or
    time.
    """

    _regions: tuple[Region, ...]
    """Source regions for the displayed recording or item."""

    _navigating_panel: MatplotlibPanel | None
    """Plot whose viewport is retained while a linked sample read completes."""

    _plot_selecting: bool
    """Whether the current scrubber change originates in plot navigation."""

    timeline_controls: QWidget
    """Timeline, interval labels, and selection and overview actions."""

    range_text: QLabel
    """Exact selected and visible sample intervals."""

    _full_view: SampleWindow | OverviewWindow | None
    """Cached full-signal data for the current item and coordinates."""

    _overview_loading: bool
    """Whether a full-signal overview is being computed."""

    _timeline_timer: QTimer
    """Debounce overview navigation independently of sample selection."""

    _spectrogram_panel: SpectrogramPanel | None
    """Default spectrogram analysis settings, if installed."""

    _frequency_panel: MatplotlibPanel | None
    """Default spectrum panel sharing the selected FFT window."""

    _custom_metrics: bool
    """Whether metrics require consumer-defined raw sample analysis."""

    metric_text: QLabel
    """Selected-window metric results."""

    item_slider: QSlider
    """Zero-based position within the filtered item selection."""

    item_position: QSpinBox
    """One-based position for precise navigation within matching items."""

    previous_item: QPushButton
    """Step to the previous matching item."""

    next_item: QPushButton
    """Step to the next matching item."""

    index_text: QLabel
    """Raw index values and decoded labels for the displayed item."""

    _window_timer: QTimer
    """Coalesce rapid selection changes before reading samples."""

    selection_text: QLabel
    """Number of items in the completed selection."""

    index_query: QPlainTextEdit
    """Core index-only query expression, independent of displayed indexes."""

    applied: QLabel
    """Definition of the last completed filter selection."""

    source: DatasetSource
    """Read-only data source shared by the browser and worker operations."""

    tabs: QTabWidget
    """Visualization and metadata panels."""

    _tasks: Tasks
    """Background operations and stale-result suppression."""

    _info: RecordingInfo | None
    """Selected recording descriptors."""

    _positions: range | npt.NDArray[np.int64]
    """Original positions in the filtered selection."""

    _metrics: dict[str, Metric]
    """Selected-window metrics evaluated in worker threads."""

    _panels: list[
        tuple[
            QWidget,
            Callable[[SampleWindow | None], None],
            Callable[[OverviewWindow], None] | None,
        ]
    ]
    """Widgets and GUI-thread callbacks for waveform-dependent panels."""

    item_navigation: QWidget
    """Item slider and stepping controls, hidden outside batched selections."""

    controls_scroll: QScrollArea
    """Recording controls, hidden when no recording is selected."""

    _has_waveform: bool
    """Whether the displayed sample window contains samples."""

    _showing_overview: bool
    """Whether visible plots use explicit reduced data."""

    _preferred_panel: QWidget | None
    """Visualization to restore after a pending sample read completes."""

    _changing_tabs: bool
    """Suppress tab preference changes during automatic visibility updates."""

    _coordinates: dict[str, QSpinBox]
    """Explicit selectors for additional sample axes."""

    _updating: bool
    """Suppress dependent requests while rebuilding navigation controls."""

    _item: int | None
    """Selected original item, or None when no item is selected."""

    def __init__(
        self,
        source: DatasetSource,
        parent: QWidget | None = None,
        *,
        default_panels: bool = True,
        metrics: Mapping[str, Metric] | None = None,
    ) -> None:
        """Create an embeddable viewer and load recording names asynchronously.

        Args:
            source: Dataset location and read-only access methods.
            parent: Optional Qt parent.
            default_panels: Whether to install the standard plot panels.
            metrics: Metric functions, replacing defaults when supplied.
                Functions run in workers and must not access Qt widgets.
        """
        super().__init__(parent)
        self.source = source
        self._info = None
        self._positions = range(0)
        self._panels = []
        self._has_waveform = False
        self._showing_overview = False
        self._preferred_panel = None
        self._changing_tabs = False
        self._coordinates = {}
        self._updating = False
        self._item = None
        self._full_view = None
        self._plot_options = PlotOptions()
        self._default_panels = []
        self._time_panels = []
        self._navigating_panel = None
        self._plot_selecting = False
        self._regions = ()
        self._overview_loading = False
        self._spectrogram_panel = None
        self._frequency_panel = None
        self._custom_metrics = metrics is not None
        self._metrics = (
            dict(metrics)
            if metrics is not None
            else {
                "Mean power": mean_power,
                "RMS": rms,
                "Peak magnitude": peak_magnitude,
            }
        )
        self._build_ui()
        self._tasks = Tasks(self, self._error, self._progress)
        self._window_timer = QTimer(self)
        self._window_timer.setSingleShot(True)
        self._window_timer.setInterval(100)
        self._window_timer.timeout.connect(self._load_window)
        self._timeline_timer = QTimer(self)
        self._timeline_timer.setSingleShot(True)
        self._timeline_timer.setInterval(100)
        self._timeline_timer.timeout.connect(self._load_timeline_detail)
        if default_panels:
            waterfall = SpectrogramPanel(options=self._plot_options)
            self._default_panels.append(waterfall)
            waterfall.region_picked.connect(self._inspect_region)
            self._spectrogram_panel = waterfall
            waterfall.analysis_changed.connect(self._analysis_requested)
            waterfall.time_navigated.connect(
                partial(self._plot_navigated, waterfall)
            )
            self._time_panels.append(waterfall)
            self.add_panel(
                "Spectrogram",
                waterfall,
                waterfall.set_window,
                overview_update=waterfall.set_overview,
            )
            for title, draw, reduced in (
                ("Time", waveform, overview_waveform),
                ("Frequency", self._draw_frequency, overview_spectrum),
                ("I/Q", constellation, overview_constellation),
            ):
                panel = MatplotlibPanel(
                    partial(waveform, options=self._plot_options)
                    if title == "Time"
                    else draw,
                    draw_overview=partial(
                        overview_waveform, options=self._plot_options
                    )
                    if title == "Time"
                    else reduced,
                )
                self._default_panels.append(panel)
                if title == "Frequency":
                    self._frequency_panel = panel
                panel.region_picked.connect(self._inspect_region)
                if title == "Time":
                    self._time_panels.append(panel)
                    panel.time_navigated.connect(
                        partial(self._plot_navigated, panel)
                    )
                self.add_panel(
                    title,
                    panel,
                    panel.set_window,
                    overview_update=panel.set_overview,
                )
        for attribute, title in (
            ("recording_metadata", "Recording metadata"),
            ("item_metadata", "Item metadata"),
            ("indexes", "Indexes"),
            ("channel_metadata", "Channel metadata"),
            ("metadata", "Resolved item metadata"),
            ("collection_metadata", "Collection metadata"),
        ):
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            setattr(self, attribute, editor)
            self.tabs.addTab(editor, title)
        self.region_browser = RegionBrowser()
        self.region_browser.interval_selected.connect(
            self._select_region_interval
        )
        self.tabs.addTab(self.region_browser, "Captures and annotations")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.tabs.tabBar().tabBarClicked.connect(self._tab_changed)
        self.show_resolved.toggled.connect(self._metadata_visibility)
        self._set_waveform_visible(False)
        self._metadata_visibility()
        self._tasks.submit(
            "catalog",
            lambda stop, report: source.catalog(),
            self._catalog_ready,
        )

    def _build_ui(self) -> None:
        """Build navigation, filter controls, item slider, and plot area."""
        layout = QVBoxLayout(self)
        self.status = QLabel("Opening dataset...")
        layout.addWidget(self.status)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        sidebar = QSplitter(Qt.Orientation.Vertical)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Dataset")
        self.tree.currentItemChanged.connect(self._tree_changed)
        sidebar.addWidget(self.tree)
        controls = self.recording_controls = QWidget()
        controls.setEnabled(False)
        form = QFormLayout(controls)
        self.index_selection = QListWidget()
        self.index_selection.setMaximumHeight(100)
        self.index_selection.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.index_selection.itemSelectionChanged.connect(self._request_window)
        form.addRow("Displayed indexes", self.index_selection)
        self.show_resolved = QCheckBox("Show resolved item metadata")
        form.addRow(self.show_resolved)
        self._build_filters(form)
        self.axis_form = QFormLayout()
        form.addRow(self.axis_form)
        scroll = self.controls_scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        sidebar.addWidget(scroll)
        sidebar.setSizes([260, 550])
        splitter.addWidget(sidebar)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        self._build_navigation(body_layout)
        self.index_text = QLabel()
        self.index_text.setWordWrap(True)
        self.index_text.setTextFormat(Qt.TextFormat.PlainText)
        body_layout.addWidget(self.index_text)
        self.metric_text = QLabel()
        body_layout.addWidget(self.metric_text)
        self.tabs = QTabWidget()
        body_layout.addWidget(self.tabs, 1)
        self.timeline_controls = QWidget()
        timeline_layout = QVBoxLayout(self.timeline_controls)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        self.timeline = RangeSelector()
        self.timeline.range_changed.connect(self._range_changed)
        self.timeline.region_selected.connect(
            lambda region: self._inspect_region(region.key)
        )
        self.timeline.visible_range_changed.connect(
            self._timeline_view_changed
        )
        timeline_layout.addWidget(self.timeline)
        overlay_row = QHBoxLayout()
        self.axis_mode = QComboBox()
        for label, mode in (
            ("Sample index", "samples"),
            ("Elapsed sample time", "elapsed"),
            ("Capture timestamp (UTC)", "timestamp"),
        ):
            self.axis_mode.addItem(label, mode)
        self.show_captures = QCheckBox("Captures")
        self.show_annotations = QCheckBox("Annotations")
        self.show_captures.setChecked(True)
        self.show_annotations.setChecked(True)
        overlay_row.addWidget(QLabel("Plot axis"))
        for widget in (
            self.axis_mode,
            self.show_captures,
            self.show_annotations,
        ):
            overlay_row.addWidget(widget)
        overlay_row.addStretch()
        timeline_layout.addLayout(overlay_row)
        self.axis_mode.currentIndexChanged.connect(self._presentation_changed)
        self.show_captures.toggled.connect(self._presentation_changed)
        self.show_annotations.toggled.connect(self._presentation_changed)
        range_row = QHBoxLayout()
        self.range_text = QLabel()
        range_row.addWidget(self.range_text, 1)
        fit = QPushButton("Select full recording")
        fit.clicked.connect(self.timeline.fit)
        range_row.addWidget(fit)
        fit_view = QPushButton("Fit recording")
        fit_view.clicked.connect(self.timeline.fit_view)
        range_row.addWidget(fit_view)
        zoom = QPushButton("Zoom to selection")
        zoom.clicked.connect(self.timeline.zoom_to_selection)
        range_row.addWidget(zoom)
        timeline_layout.addLayout(range_row)
        body_layout.addWidget(self.timeline_controls)
        self.timeline_controls.hide()
        splitter.addWidget(body)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

    def _build_navigation(self, layout: QVBoxLayout) -> None:
        """Add a slider and precise stepping through matching items.

        Args:
            layout: Layout receiving the navigation controls.
        """
        self.item_navigation = QWidget()
        navigation_layout = QVBoxLayout(self.item_navigation)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.item_navigation)
        navigation = QHBoxLayout()
        self.previous_item = QPushButton("Previous item")
        self.previous_item.clicked.connect(lambda: self.step_item(-1))
        self.next_item = QPushButton("Next item")
        self.next_item.clicked.connect(lambda: self.step_item(1))
        navigation.addWidget(self.previous_item)
        navigation.addWidget(self.next_item)
        for key, step in (("Alt+Left", -1), ("Alt+Right", 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda step=step: self.step_item(step))
        navigation.addWidget(QLabel("Match"))
        self.item_position = QSpinBox()
        self.item_position.setRange(0, 0)
        self.item_position.setEnabled(False)
        self.item_position.valueChanged.connect(
            lambda value: self.item_slider.setValue(value - 1)
        )
        navigation.addWidget(self.item_position)
        self.selection_text = QLabel()
        navigation.addWidget(self.selection_text, 1)
        navigation_layout.addLayout(navigation)
        self.item_slider = QSlider(Qt.Orientation.Horizontal)
        self.item_slider.setRange(0, 0)
        self.item_slider.setSingleStep(1)
        self.item_slider.setEnabled(False)
        self.item_slider.valueChanged.connect(self._item_changed)
        self.item_slider.setAccessibleName("Matching item")
        navigation_layout.addWidget(self.item_slider)
        self.previous_item.setEnabled(False)
        self.next_item.setEnabled(False)

    def _build_filters(self, form: QFormLayout) -> None:
        """Add a multiline index query editor and execution controls.

        Args:
            form: Control layout receiving the filter editor.
        """
        self.index_query = QPlainTextEdit()
        self.index_query.setAccessibleName("Index query")
        self.index_query.setMinimumHeight(120)
        self.index_query.setTabChangesFocus(True)
        self.index_query.setPlaceholderText(
            'label(mod_class_id) == "QPSK"\nand snr_db >= 10'
        )
        self.index_query.setToolTip(
            "Compare index values with ==, !=, <, <=, >, >=, or in. "
            "Combine with and, or, not, and parentheses. "
            'Use label(name) for labels and index("path/name") for paths. '
            "Press Ctrl+Enter to apply the query."
        )
        for key in ("Ctrl+Return", "Ctrl+Enter"):
            shortcut = QShortcut(QKeySequence(key), self.index_query)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(self.apply_filters)
        form.addRow(QLabel("Index query"))
        form.addRow(self.index_query)
        buttons = QHBoxLayout()
        apply = QPushButton("Apply query")
        apply.clicked.connect(self.apply_filters)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel_filter)
        buttons.addWidget(apply)
        buttons.addWidget(cancel)
        form.addRow(buttons)
        self.applied = QLabel("No query applied")
        self.applied.setTextFormat(Qt.TextFormat.PlainText)
        self.applied.setWordWrap(True)
        form.addRow(self.applied)

    def add_panel(
        self,
        title: str,
        widget: QWidget,
        update: Callable[[SampleWindow | None], None],
        *,
        overview_update: Callable[[OverviewWindow], None] | None = None,
    ) -> None:
        """Register an ordinary Qt widget as a visualization panel.

        Args:
            title: Tab label.
            widget: Widget adopted by the tab container.
            update: GUI-thread callback receiving selected data or None when
                the selection is cleared. Keep expensive analysis in metrics
                or consumer-managed workers.
            overview_update: Optional callback for explicitly reduced data.
                Without it, this panel is hidden for large selections.
        """
        index = self.tabs.addTab(widget, title)
        self._panels.append((widget, update, overview_update))
        self.tabs.setTabVisible(
            index,
            self._has_waveform
            and (not self._showing_overview or overview_update is not None),
        )

    def _catalog_ready(self, result: object) -> None:
        """Populate collection membership and the complete recording branch.

        Args:
            result: Dataset catalog returned by the worker.
        """
        catalog = cast(DatasetCatalog, result)
        self.tree.clear()
        collections = QTreeWidgetItem(self.tree, ["Collections"])
        recordings = QTreeWidgetItem(self.tree, ["Recordings"])
        existing = set(catalog.recordings)
        for name, references in catalog.collections.items():
            node = QTreeWidgetItem(collections, [name])
            node.setData(0, Qt.ItemDataRole.UserRole, ("collection", name))
            for recording in references:
                self._recording_node(node, recording, recording in existing)
        for name in catalog.recordings:
            self._recording_node(recordings, name, True)
        collections.setExpanded(True)
        recordings.setExpanded(True)
        first = recordings.child(0)
        if first is not None:
            self.tree.setCurrentItem(first)
        else:
            self.status.setText("This store has no recordings")

    @staticmethod
    def _recording_node(
        parent: QTreeWidgetItem,
        name: str,
        exists: bool,
    ) -> None:
        """Add a recording reference as a leaf in the dataset tree.

        Args:
            parent: Collection or recording branch.
            name: Exact recording name, independent of the displayed path.
            exists: Whether this collection reference resolves to a recording.
        """
        node = QTreeWidgetItem(parent, [name])
        if not exists:
            node.setText(0, name + " (missing)")
            node.setToolTip(0, "Collection references a missing recording")
            node.setDisabled(True)
            return
        node.setData(0, Qt.ItemDataRole.UserRole, ("recording", name))

    def _tree_changed(self) -> None:
        """Open the selected collection or recording without losing
        identity.
        """
        node = self.tree.currentItem()
        if node is None or node.isDisabled():
            return
        target = node.data(0, Qt.ItemDataRole.UserRole)
        if target is None:
            self._reset_recording()
            self.status.setText(node.text(0))
            return
        kind, name = target
        if (
            kind != "collection"
            and self._info is not None
            and self._info.name == name
        ):
            self._show_recording()
            return
        self._reset_recording()
        self.status.setText(f"Opening {name}...")
        source = self.source
        if kind == "collection":
            self._tasks.submit(
                "collection",
                lambda stop, report: source.collection_metadata(name),
                lambda result: self._collection_ready(name, result),
            )
        else:
            self._show_recording()
            self._tasks.submit(
                "describe",
                lambda stop, report: source.describe(name),
                self._description_ready,
            )

    def _show_recording(self) -> None:
        """Select the first visualization while retaining the item and
        filters.
        """
        self._preferred_panel = self._panels[0][0] if self._panels else None
        if self._has_waveform and self._preferred_panel is not None:
            self.tabs.setCurrentWidget(self._preferred_panel)
        self._navigation_visibility()

    def _reset_recording(self) -> None:
        """Cancel old work and clear recording-specific navigation controls."""
        for key in (
            "describe",
            "collection",
            "filter",
            "window",
            "overview",
            "timeline",
        ):
            self._tasks.cancel(key)
        self._timeline_timer.stop()
        self._info = None
        self._full_view = None
        self._preferred_panel = None
        self.recording_metadata.clear()
        self.collection_metadata.clear()
        self.recording_controls.setEnabled(False)
        self._positions = range(0)
        self._updating = True
        self.timeline.set_extent(0)
        self._window_timer.stop()
        self.item_slider.setRange(0, 0)
        self.item_slider.setEnabled(False)
        self.item_position.setRange(0, 0)
        self.item_position.setEnabled(False)
        self.previous_item.setEnabled(False)
        self.next_item.setEnabled(False)
        self.index_selection.clear()
        self.index_query.clear()
        self.applied.setText("No query applied")
        self.selection_text.clear()
        self._coordinates.clear()
        while self.axis_form.rowCount():
            self.axis_form.removeRow(0)
        self._updating = False
        self._clear_window()
        self._metadata_visibility()

    def _collection_ready(self, name: str, result: object) -> None:
        """Show collection metadata without opening its referenced recordings.

        Args:
            name: Selected collection name.
            result: Collection metadata and recording references.
        """
        self.collection_metadata.setPlainText(
            json.dumps(result, indent=2, ensure_ascii=True)
        )
        self._metadata_visibility(collection=True)
        self.tabs.setCurrentWidget(self.collection_metadata)
        self.status.setText(f"Collection: {name}")

    def _description_ready(self, result: object) -> None:
        """Rebuild selectors from the newly selected recording's descriptors.

        Args:
            result: Recording descriptors returned by the worker.
        """
        self._info = info = cast(RecordingInfo, result)
        self.recording_metadata.setPlainText(
            json.dumps(info.metadata, indent=2, ensure_ascii=True)
        )
        self._metadata_visibility()
        self.recording_controls.setEnabled(True)
        self._updating = True
        self.index_selection.clear()
        self.index_selection.addItems(sorted(info.indexes))
        self.index_query.clear()
        self.applied.setText("No query applied")
        self._coordinates.clear()
        while self.axis_form.rowCount():
            self.axis_form.removeRow(0)
        for axis, length in zip(info.axes, info.shape, strict=True):
            if axis in {"item", "time", "iq"}:
                continue
            selector = QSpinBox()
            selector.setRange(0, max(0, min(length - 1, 2**31 - 1)))
            selector.valueChanged.connect(self._reset_timeline)
            self.axis_form.addRow(axis, selector)
            self._coordinates[axis] = selector
        self._positions = range(info.item_count)
        self._updating = False
        self._reset_items()

    def _selected_indexes(self) -> list[str]:
        """Return explicitly selected index names.

        Returns:
            Selected names in display order.
        """
        return [item.text() for item in self.index_selection.selectedItems()]

    def _reset_items(self) -> None:
        """Reset the slider to the first item in the completed selection."""
        self._tasks.cancel("window")
        self._window_timer.stop()
        self._clear_window()
        self._updating = True
        count = len(self._positions)
        maximum = min(max(0, count - 1), 2**31 - 2)
        self.item_slider.setRange(0, maximum)
        self.item_slider.setValue(0)
        self.item_position.setRange(
            1 if count else 0, maximum + 1 if count else 0
        )
        self.item_position.setValue(1 if count else 0)
        self.item_slider.setEnabled(count > 1)
        self.item_position.setEnabled(count > 1)
        self._updating = False
        self._item_changed()

    def _item_changed(self) -> None:
        """Map the slider position to an original item and update all
        controls.
        """
        if self._updating:
            return
        position = self.item_slider.value()
        count = len(self._positions)
        self._navigation_visibility()
        self.previous_item.setEnabled(count > 0 and position > 0)
        self.next_item.setEnabled(
            count > 0 and position < self.item_slider.maximum()
        )
        if count == 0:
            self._item = None
            self._tasks.cancel("overview")
            self._tasks.cancel("timeline")
            self._timeline_timer.stop()
            self._full_view = None
            self.timeline.set_extent(0)
            self.selection_text.setText("0 matching items")
            self.status.setText("No matching items")
            return
        changed = self._item != int(self._positions[position])
        self._item = int(self._positions[position])
        self.item_position.setValue(position + 1)
        if self._info is not None and not self._info.batched:
            self.selection_text.setText("Continuous recording")
        else:
            self.selection_text.setText(
                f"of {count:,} matching items | Original item {self._item:,}"
            )
        if changed:
            self._reset_timeline()
        else:
            self._request_window()

    def _range_changed(self, start: int, stop: int) -> None:
        """Publish an exact interval and debounce its sample read.

        Args:
            start: Inclusive source sample position.
            stop: Exclusive source sample position.
        """
        self._update_range_text()
        self._request_window(clear_display=not self._plot_selecting)

    def _plot_navigated(
        self, panel: MatplotlibPanel, limits: tuple[float, float]
    ) -> None:
        """Map a completed time gesture to the shared sample selection.

        Args:
            panel: Time or spectrogram panel initiating navigation.
            limits: Visible horizontal sample or time coordinates.
        """
        view = panel._last_view
        if view is None or self._info is None:
            return
        context = view.context if isinstance(view, OverviewWindow) else view
        count = (
            view.signal.count
            if isinstance(view, OverviewWindow)
            else len(view.samples)
        )
        interval = sample_interval(context, count, limits, self._plot_options)
        if interval is None:
            panel.redraw()
            self.status.setText(
                "The selected time interval contains no samples"
            )
            return
        if interval == self.timeline.selection:
            return
        self._navigating_panel = panel
        # set_range emits synchronously. Preserve the initiating plot while its
        # finer data loads, and retain its limits for the linked redraw.
        self._plot_selecting = True
        try:
            self.timeline.set_range(*interval)
            self.timeline.reveal_selection()
        finally:
            self._plot_selecting = False

    def _update_range_text(self) -> None:
        """Display both source intervals using exclusive stop coordinates."""
        start, stop = self.timeline.selection
        lo, hi = self.timeline.visible_range
        self.range_text.setText(
            f"Selected [{start:,}, {stop:,}) | {stop - start:,} samples\n"
            f"Overview [{lo:,}, {hi:,}) | {hi - lo:,} samples"
        )

    def _analysis_requested(self) -> None:
        """Recompute reduced selections after an FFT setting changes."""
        start, stop = self.timeline.selection
        if stop - start > 65536:
            self._request_window(clear_display=False)
        elif self._frequency_panel is not None:
            with self._frequency_panel.preserve_view():
                self._frequency_panel.redraw()

    def _draw_frequency(self, figure: Figure, window: SampleWindow) -> None:
        """Draw raw bin power using the shared window selection.

        Args:
            figure: Target figure.
            window: Cached sample interval.
        """
        window_function = (
            self._spectrogram_panel.analysis_settings()[2]
            if self._spectrogram_panel is not None
            else "hann"
        )
        spectrum(figure, window, window_function=window_function)

    def _presentation_changed(self) -> None:
        """Redraw overlays and coordinate axes without reading samples."""
        self._plot_options.axis_mode = cast(
            AxisMode, self.axis_mode.currentData()
        )
        self._plot_options.captures = self.show_captures.isChecked()
        self._plot_options.annotations = self.show_annotations.isChecked()
        self._update_timeline_regions()
        for panel in self._default_panels:
            panel.redraw()

    def _axis_availability(self, window: SampleWindow, count: int) -> None:
        """Offer time modes only when the needed metadata is available.

        Args:
            window: Resolved sample and acquisition metadata.
            count: Selected sample count.
        """
        rate = sample_rate(window.metadata)
        segments = capture_segments(
            window.metadata, window.total_samples or window.start + count
        )
        available = (
            True,
            rate is not None,
            rate is not None
            and any(s.timestamp is not None for s in segments),
        )
        model = cast(QStandardItemModel, self.axis_mode.model())
        for index, enabled in enumerate(available):
            model.item(index).setEnabled(enabled)
        if not available[self.axis_mode.currentIndex()]:
            self.axis_mode.blockSignals(True)
            self.axis_mode.setCurrentIndex(0)
            self.axis_mode.blockSignals(False)
            self._plot_options.axis_mode = "samples"

    def _update_timeline_regions(self) -> None:
        """Filter overview bands independently of acquisition processing."""
        self.timeline.set_regions(
            tuple(
                region
                for region in self._regions
                if (
                    self._plot_options.captures
                    if region.kind == "capture"
                    else self._plot_options.annotations
                )
            )
        )

    def _inspect_region(self, key: str) -> None:
        """Open the original metadata for a picked source region.

        Args:
            key: Source region identifier.
        """
        self.region_browser.select_key(key)
        self.tabs.setCurrentWidget(self.region_browser)

    def _select_region_interval(self, start: int, stop: int) -> None:
        """Select the inspected source interval and return to a plot.

        Args:
            start: Inclusive stored sample coordinate.
            stop: Exclusive stored sample coordinate.
        """
        self._show_recording()
        self.timeline.set_range(start, stop)

    def _reset_timeline(self) -> None:
        """Select the whole signal and build its overview in a worker."""
        if self._updating or self._info is None or self._item is None:
            return
        self._tasks.cancel("overview")
        self._tasks.cancel("timeline")
        self._timeline_timer.stop()
        self._overview_loading = False
        self._tasks.cancel("window")
        self._full_view = None
        self._updating = True
        length = self._info.shape[self._info.axes.index("time")]
        self.timeline.set_extent(length)
        self._update_range_text()
        self._updating = False
        self._navigation_visibility()
        if not length:
            self._clear_display()
            self.status.setText("No samples")
            return
        self._request_window()

    def _start_overview(self) -> None:
        """Start the full timeline scan after the navigation debounce."""
        if self._overview_loading or self._info is None:
            return
        self._overview_loading = True
        length = self.timeline.extent
        fft_size, overlap, window_function = (
            self._spectrogram_panel.analysis_settings()
            if self._spectrogram_panel is not None
            else (None, 0.5, "hann")
        )
        name = self._info.name
        item = self._item if self._info.batched else None
        coordinates = {
            key: control.value() for key, control in self._coordinates.items()
        }

        def work(
            stop: Event, report: Progress
        ) -> SampleWindow | OverviewWindow:
            """Build a complete timeline using bounded reads.

            Args:
                stop: Cancellation flag.
                report: Examined-sample progress callback.

            Returns:
                Full raw signal or its reduced representation.
            """
            if length <= 65536:
                return self.source.read_window(
                    name, item=item, count=length, coordinates=coordinates
                )
            return self.source.read_overview(
                name,
                item=item,
                count=length,
                coordinates=coordinates,
                fft_size=fft_size or 256,
                overlap=overlap,
                window_function=window_function,
                cancelled=stop.is_set,
                progress=report,
            )

        self._tasks.submit("overview", work, self._overview_ready)

    def _overview_ready(self, result: object) -> None:
        """Install the full-signal timeline without moving its selection.

        Args:
            result: Complete signal or reduced data from the current worker.
        """
        self._overview_loading = False
        self._install_overview(cast(SampleWindow | OverviewWindow, result))
        if self.timeline.selection == (0, self.timeline.extent):
            self._request_window()

    def _install_overview(self, view: SampleWindow | OverviewWindow) -> None:
        """Cache a full-signal view and update the timeline envelope.

        Args:
            view: Data spanning the full current signal.
        """
        self._full_view = view
        if isinstance(view, OverviewWindow):
            peaks = view.signal.peak
        else:
            peaks = np.array(
                [
                    np.max(np.abs(part))
                    for part in np.array_split(
                        view.samples, min(512, len(view.samples))
                    )
                ]
            )
        self.timeline.set_overview(peaks.tolist())

    def _timeline_view_changed(self, start: int, stop: int) -> None:
        """Request finer peaks without changing waveform panels.

        Args:
            start: Inclusive overview coordinate.
            stop: Exclusive overview coordinate.
        """
        self._update_range_text()
        if self._updating:
            return
        self._tasks.cancel("timeline")
        self._timeline_timer.stop()
        if start == stop or self._info is None or self._item is None:
            return
        if self.timeline.has_overview(start, stop):
            return
        if (start, stop) == (
            0,
            self.timeline.extent,
        ) and self._full_view is not None:
            self._install_overview(self._full_view)
        else:
            self._timeline_timer.start()

    def _load_timeline_detail(self) -> None:
        """Read only the visible overview region in a replaceable worker."""
        if (
            self._info is None
            or self._item is None
            or not self.timeline.extent
        ):
            return
        name = self._info.name
        item = self._item if self._info.batched else None
        start, end = self.timeline.visible_range
        coordinates = {
            key: control.value() for key, control in self._coordinates.items()
        }

        def work(stop: Event, report: Progress) -> npt.NDArray[np.float64]:
            """Read a bounded envelope for the requested overview interval.

            Args:
                stop: Cooperative cancellation flag.
                report: Unused progress callback.

            Returns:
                Peak magnitudes for the visible interval.
            """
            return self.source.read_peaks(
                name,
                item=item,
                start=start,
                count=end - start,
                coordinates=coordinates,
                cancelled=stop.is_set,
            )

        def ready(result: object) -> None:
            """Install peaks for the current viewport.

            Args:
                result: Peak magnitudes from the current worker.
            """
            self.timeline.set_overview(
                cast(npt.NDArray[np.float64], result), start=start, stop=end
            )

        self._tasks.submit("timeline", work, ready)

    def step_item(self, step: int) -> None:
        """Step through individual items in the filtered selection.

        Args:
            step: Direction, -1 for previous or 1 for next.
        """
        if len(self._positions):
            self.item_slider.setValue(self.item_slider.value() + step)

    def _request_window(self, *, clear_display: bool = True) -> None:
        """Debounce reads while discarding results for obsolete selections.

        Args:
            clear_display: Clear old data when the selection changes. Disable
                for analysis changes to keep the same selection visible.
        """
        if self._updating or self._info is None or self._item is None:
            return
        if not self.timeline.extent:
            return
        # Suppress the old result immediately, before the debounce expires.
        # Otherwise it could be displayed under the newly selected interval.
        self._tasks.cancel("window")
        if clear_display:
            self._clear_display()
        self._window_timer.start()
        self.status.setText(f"Loading {self._info.name}, item {self._item}...")

    def _load_window(self) -> None:
        """Read selected samples and calculate metrics in a worker."""
        if self._updating or self._info is None or self._item is None:
            return
        name = self._info.name
        item = self._item if self._info.batched else None
        start, end = self.timeline.selection
        count = end - start
        if not count:
            return
        full = (start, end) == (0, self.timeline.extent)
        if self._full_view is None and (not full or count > 65536):
            self._start_overview()
            if full:
                # The full-signal worker supplies this same data. Let its
                # completion reissue the request instead of scanning twice.
                return
        cached = (
            self._full_view
            if (start, end) == (0, self.timeline.extent)
            else None
        )
        fft_size, overlap, window_function = (
            self._spectrogram_panel.analysis_settings()
            if self._spectrogram_panel is not None
            else (None, 0.5, "hann")
        )
        # Snapshot widget state on the GUI thread. The worker must not read
        # controls that may change while its backend operation is in flight.
        coordinates = {
            key: control.value() for key, control in self._coordinates.items()
        }
        columns = self._selected_indexes()
        metrics = dict(self._metrics)

        def work(
            stop: Event,
            report: Progress,
        ) -> tuple[SampleWindow | OverviewWindow, dict[str, float | str]]:
            """Read a window and calculate registered metrics.

            Args:
                stop: Cooperative cancellation flag.
                report: Progress callback supplied by the worker runner.

            Returns:
                Selected window and metric results.
            """
            if count > 65536:
                size = fft_size or 256
                hop = max(1, size - int(size * overlap))
                if isinstance(cached, OverviewWindow) and (
                    cached.signal.fft_size,
                    cached.signal.hop,
                    cached.signal.window_function,
                ) == (size, hop, window_function):
                    # Reuse spectra only for matching analysis settings.
                    # Refresh the small context separately so displayed index
                    # choices do not force another full sample scan.
                    context = self.source.read_window(
                        name,
                        item=item,
                        start=start,
                        count=1,
                        coordinates=coordinates,
                        indexes=columns,
                    )
                    reduced = OverviewWindow(
                        replace(context, samples=context.samples[:0]),
                        cached.signal,
                    )
                else:
                    reduced = self.source.read_overview(
                        name,
                        item=item,
                        start=start,
                        count=count,
                        coordinates=coordinates,
                        indexes=columns,
                        fft_size=size,
                        window_function=window_function,
                        overlap=overlap,
                        cancelled=stop.is_set,
                        progress=report,
                    )
                return reduced, _metric_values(
                    reduced, metrics, self._custom_metrics, stop
                )
            window = self.source.read_window(
                name,
                item=item,
                start=start,
                count=count,
                coordinates=coordinates,
                indexes=columns,
            )
            return window, _metric_values(
                window, metrics, self._custom_metrics, stop
            )

        self._tasks.submit("window", work, self._window_ready)

    def _window_ready(self, result: object) -> None:
        """Deliver current data to all registered panels on the GUI thread.

        Args:
            result: Window and calculated metric values.
        """
        view, metrics = cast(
            tuple[SampleWindow | OverviewWindow, dict[str, float | str]],
            result,
        )
        reduced = isinstance(view, OverviewWindow)
        window = view.context if isinstance(view, OverviewWindow) else view
        count = (
            view.signal.count
            if isinstance(view, OverviewWindow)
            else len(window.samples)
        )
        self._regions = window_regions(window, count)
        self._axis_availability(window, count)
        self.region_browser.set_regions(self._regions)
        self._metadata_visibility()
        self._update_timeline_regions()
        if self.timeline.selection == (0, self.timeline.extent):
            self._install_overview(view)
        self.status.setText(
            f"{window.recording} | "
            + (f"Item {window.item} | " if window.item is not None else "")
            + f"Samples [{window.start}, {window.start + count})"
            + (" | Reduced overview" if reduced else "")
        )
        self.index_text.setText(
            "    ".join(
                f"{name}: {json.dumps(entry['value'], ensure_ascii=True)}"
                + (
                    f" : {json.dumps(entry['label'], ensure_ascii=True)}"
                    if "label" in entry
                    else ""
                )
                for name, entry in window.indexes.items()
            )
        )
        self.metric_text.setText(
            "    ".join(
                f"{key}: {value:.6g}"
                if isinstance(value, float)
                else f"{key}: {value}"
                for key, value in metrics.items()
            )
        )
        self.recording_metadata.setPlainText(
            json.dumps(window.recording_metadata, indent=2, ensure_ascii=True)
        )
        self.item_metadata.setPlainText(
            "No stored item metadata."
            if window.item_metadata is None
            else json.dumps(window.item_metadata, indent=2, ensure_ascii=True)
        )
        self.channel_metadata.setPlainText(
            json.dumps(window.channel_metadata, indent=2, ensure_ascii=True)
            if window.channel_metadata is not None
            else ""
        )
        self.indexes.setPlainText(
            json.dumps(window.indexes, indent=2, ensure_ascii=True)
        )
        self.metadata.setPlainText(
            json.dumps(window.metadata, indent=2, ensure_ascii=True)
        )
        # Use the latest initiating view for both time panels. Each panel keeps
        # its own vertical limits and Home view while sharing the time range.
        horizontal = (
            self._navigating_panel.figure.axes[0].get_xlim()
            if self._navigating_panel is not None
            and self._navigating_panel.figure.axes
            else None
        )
        for widget, update, overview_update in self._panels:
            try:
                with (
                    widget.preserve_view(keep_home=True, horizontal=horizontal)
                    if horizontal is not None and widget in self._time_panels
                    else nullcontext()
                ):
                    if isinstance(view, OverviewWindow):
                        if overview_update is not None:
                            overview_update(view)
                    else:
                        update(window)
            except Exception as exc:
                self._error(f"Visualization error: {exc}")
        self._navigating_panel = None
        self._set_waveform_visible(count > 0, overview=reduced)
        self.window_changed.emit(None if reduced else window)
        self.overview_changed.emit(view if reduced else None)

    def _metadata_visibility(
        self, checked: bool = False, *, collection: bool = False
    ) -> None:
        """Expose metadata scopes applicable to the selected recording.

        Args:
            checked: Qt checkbox signal value. The checkbox stores the state.
            collection: Whether a collection is selected.
        """
        info = self._info
        self._changing_tabs = True
        self.tabs.setVisible(info is not None or collection)
        for editor, visible in (
            (self.recording_metadata, info is not None),
            (self.item_metadata, info is not None and info.batched),
            (self.indexes, info is not None and bool(info.indexes)),
            (
                self.channel_metadata,
                info is not None and "channel" in info.axes,
            ),
            (
                self.metadata,
                info is not None
                and info.batched
                and self.show_resolved.isChecked(),
            ),
            (self.collection_metadata, collection),
            (self.region_browser, info is not None and bool(self._regions)),
        ):
            self.tabs.setTabVisible(self.tabs.indexOf(editor), visible)
        self._changing_tabs = False
        self._navigation_visibility()

    def _navigation_visibility(self) -> None:
        """Show recording controls and item navigation in their own scope."""
        self.controls_scroll.setVisible(self._info is not None)
        self.timeline_controls.setVisible(
            self._info is not None
            and bool(len(self._positions))
            and self.timeline.extent > 0
            and (
                any(
                    self.tabs.currentWidget() is panel
                    for panel, _, _ in self._panels
                )
                if self._has_waveform
                else self._preferred_panel is not None
            )
        )
        self.item_navigation.setVisible(
            self._info is not None
            and self._info.batched
            and bool(len(self._positions))
        )

    def _tab_changed(self, index: int) -> None:
        """Remember explicit tab changes while a sample read is pending.

        Args:
            index: Selected tab index.
        """
        if not self._changing_tabs:
            widget = self.tabs.widget(index)
            self._preferred_panel = (
                widget
                if any(widget is panel for panel, _, _ in self._panels)
                else None
            )
            self._navigation_visibility()

    def _set_waveform_visible(
        self, visible: bool, *, overview: bool = False
    ) -> None:
        """Show visualization tabs only when a waveform is available.

        Args:
            visible: Whether the current window contains samples.
            overview: Whether the current selection uses reduced data.
        """
        self._has_waveform = visible
        self._showing_overview = overview
        self._changing_tabs = True
        for widget, _, overview_update in self._panels:
            self.tabs.setTabVisible(
                self.tabs.indexOf(widget),
                visible and (not overview or overview_update is not None),
            )
        if (
            visible
            and self._preferred_panel is not None
            and self.tabs.isTabVisible(
                self.tabs.indexOf(self._preferred_panel)
            )
        ):
            self.tabs.setCurrentWidget(self._preferred_panel)
        self._changing_tabs = False
        self.metric_text.setVisible(visible)
        self.index_text.setVisible(visible)
        self._navigation_visibility()

    def _clear_display(self) -> None:
        """Clear displays so a pending selection never shows another item."""
        self._set_waveform_visible(False)
        self._navigating_panel = None
        self._regions = ()
        self.region_browser.set_regions(())
        self._metadata_visibility()
        self.timeline.set_regions(())
        self.metric_text.clear()
        self.index_text.clear()
        self.metadata.clear()
        self.item_metadata.clear()
        self.channel_metadata.clear()
        self.indexes.clear()
        for _, update, _ in self._panels:
            try:
                update(None)
            except Exception as exc:
                self._error(f"Visualization error: {exc}")
        self.window_changed.emit(None)
        self.overview_changed.emit(None)

    def _clear_window(self) -> None:
        """Clear the selected item and all dependent displays."""
        self._item = None
        self._clear_display()

    def apply_filters(self) -> None:
        """Run the edited filter definition without replacing results early."""
        if self._info is None:
            return
        if not self._info.batched:
            self._error("Item filtering requires a batched recording")
            return
        name = self._info.name
        index_query = self.index_query.toPlainText().strip() or None
        self.status.setText("Filtering...")
        self._tasks.submit(
            "filter",
            lambda stop, report: self.source.filter(
                name,
                index_query=index_query,
                cancelled=stop.is_set,
                progress=report,
            ),
            lambda result: self._filtered(result, index_query or ""),
        )

    def _filtered(self, result: object, description: str) -> None:
        """Publish a completed filter selection and restart item navigation.

        Args:
            result: Matching original positions.
            description: Applied filter definition.
        """
        self._positions = cast(npt.NDArray[np.int64], result)
        self.applied.setText("Applied: " + (description or "All items"))
        self._reset_items()

    def cancel_filter(self) -> None:
        """Keep the last completed selection when cancelling a filter."""
        self._tasks.cancel("filter")
        self.status.setText("Filter cancelled; previous selection retained")

    def _progress(self, key: str, examined: int, total: int) -> None:
        """Display background filter progress.

        Args:
            key: Operation name.
            examined: Examined item count.
            total: Total item count.
        """
        if key == "overview" and self.timeline.selection != (
            0,
            self.timeline.extent,
        ):
            return
        self.status.setText(f"{key.capitalize()}: {examined:,} / {total:,}")

    def _error(self, message: str) -> None:
        """Display an operation error without terminating the browser.

        Args:
            message: Error description.
        """
        self._overview_loading = False
        self.status.setText("Error: " + message)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cancel background operations when the viewer closes.

        Args:
            event: Qt close event.
        """
        self._window_timer.stop()
        self._timeline_timer.stop()
        self._tasks.close()
        super().closeEvent(event)


def launch(path: str) -> int:
    """Launch the standalone viewer using the reusable DatasetViewer widget.

    Args:
        path: SigMF-Zarr store path or URL.

    Returns:
        Qt application exit code.

    Raises:
        RuntimeError: If a Qt application already exists. Embed DatasetViewer
            directly when another application owns the event loop.
    """
    if QApplication.instance() is not None:
        raise RuntimeError(
            "Embed DatasetViewer in the existing Qt application"
        )
    app = QApplication([])
    app.setApplicationName("SigMF-Zarr viewer")
    window = QMainWindow()
    window.setWindowTitle(f"SigMF-Zarr viewer - {path}")
    viewer = DatasetViewer(DatasetSource(path))
    app.aboutToQuit.connect(viewer._tasks.close)
    window.setCentralWidget(viewer)
    window.resize(1280, 900)
    window.show()
    return app.exec()


__all__ = [
    "DatasetViewer",
    "Draw",
    "MatplotlibPanel",
    "Metric",
    "SpectrogramPanel",
    "launch",
]
