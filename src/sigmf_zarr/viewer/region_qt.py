"""Reusable read-only region inspector for Qt applications."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sigmf_zarr._text import escape_text
from sigmf_zarr.viewer._display import MAX_METADATA_TEXT, bounded_json
from sigmf_zarr.viewer.regions import MAX_LISTED_REGIONS, Region


class RegionBrowser(QWidget):
    """Browse region labels and original metadata without editing sources.

    The list shows at most `MAX_LISTED_REGIONS` regions, followed by a row
    that counts the rest. A region outside the list can still be inspected
    through `select_key`.
    """

    interval_selected = Signal(object, object)
    """Emit integer sample bounds when the selection action is invoked."""

    listing: QListWidget
    """Region labels in source order, up to the listing limit."""

    metadata: QPlainTextEdit
    """Original metadata for the highlighted region."""

    select_interval: QPushButton
    """Action selecting a region's sample interval."""

    _regions: tuple[Region, ...]
    """Source regions displayed by this inspector."""

    _inspected: Region | None
    """Region whose metadata is shown, which may be outside the list."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty inspector.

        Args:
            parent: Optional Qt parent.
        """
        super().__init__(parent)
        self._regions = ()
        self._inspected = None
        self.listing = QListWidget()
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.listing)
        splitter.addWidget(self.metadata)
        self.select_interval = QPushButton("Select interval")
        self.select_interval.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        layout.addWidget(self.select_interval)
        self.listing.currentRowChanged.connect(self._inspect)
        self.listing.itemDoubleClicked.connect(self._select_interval)
        self.select_interval.clicked.connect(self._select_interval)

    def set_regions(self, regions: Sequence[Region]) -> None:
        """Replace displayed regions and clear stale inspection metadata.

        Args:
            regions: Read-only source regions.
        """
        # Region order can change as metadata is refreshed. Preserve inspection
        # by source key rather than selecting whatever now occupies the row.
        key = self._inspected.key if self._inspected is not None else None
        self.listing.clear()
        self._regions = tuple(regions)
        self._show(None)
        for region in self._regions[:MAX_LISTED_REGIONS]:
            span = region.bounds.get("sample")
            suffix = (
                f" [{span.start}, {span.stop})" if span is not None else ""
            )
            # Labels come from metadata. Escape controls and line feeds.
            self.listing.addItem(
                escape_text(
                    f"{region.kind.capitalize()}: {region.label}{suffix}"
                )
            )
        hidden = len(self._regions) - MAX_LISTED_REGIONS
        if hidden > 0:
            more = QListWidgetItem(f"{hidden:,} more regions not listed")
            more.setFlags(Qt.ItemFlag.NoItemFlags)
            self.listing.addItem(more)
        if key is not None:
            self.select_key(key)

    def select_key(self, key: str) -> None:
        """Inspect a region identified by a plot or another consumer.

        Args:
            key: Source identifier.
        """
        for index, region in enumerate(self._regions):
            if region.key == key:
                if index < MAX_LISTED_REGIONS:
                    self.listing.setCurrentRow(index)
                else:
                    self.listing.setCurrentRow(-1)
                    self._show(region)
                break

    def _inspect(self, row: int) -> None:
        """Show the selected region's original metadata and source identifier.

        Args:
            row: Selected list row, or -1 when cleared.
        """
        if 0 <= row < min(len(self._regions), MAX_LISTED_REGIONS):
            self._show(self._regions[row])
        else:
            self._show(None)

    def _show(self, region: Region | None) -> None:
        """Show one region's metadata, truncated for very large entries.

        Args:
            region: Region to inspect, or None to clear the inspector.
        """
        self._inspected = region
        if region is None:
            self.metadata.clear()
            self.select_interval.setEnabled(False)
            return
        self.metadata.setPlainText(
            escape_text(region.key)
            + "\n\n"
            + bounded_json(dict(region.metadata), MAX_METADATA_TEXT)
        )
        span = region.bounds.get("sample")
        self.select_interval.setEnabled(
            span is not None
            and type(span.start) is int
            and type(span.stop) is int
        )

    def _select_interval(self) -> None:
        """Request the inspected sample range without mutating metadata."""
        if self._inspected is not None:
            span = self._inspected.bounds.get("sample")
            if (
                span is not None
                and type(span.start) is int
                and type(span.stop) is int
            ):
                self.interval_selected.emit(span.start, span.stop)
