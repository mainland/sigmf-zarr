"""Reusable read-only region inspector for Qt applications."""

from __future__ import annotations

import json
from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sigmf_zarr.viewer.regions import Region


class RegionBrowser(QWidget):
    """Browse region labels and original metadata without editing sources."""

    interval_selected = Signal(object, object)
    """Emit integer sample bounds when the selection action is invoked."""

    listing: QListWidget
    """Region labels in source order."""

    metadata: QPlainTextEdit
    """Original metadata for the highlighted region."""

    select_interval: QPushButton
    """Action selecting a region's sample interval."""

    _regions: tuple[Region, ...]
    """Source regions displayed by this inspector."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty inspector.

        Args:
            parent: Optional Qt parent.
        """
        super().__init__(parent)
        self._regions = ()
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
        old = self.listing.currentRow()
        # Region order can change as metadata is refreshed. Preserve inspection
        # by source key rather than selecting whatever now occupies the row.
        key = self._regions[old].key if 0 <= old < len(self._regions) else None
        self.listing.clear()
        self._regions = tuple(regions)
        self.metadata.clear()
        self.select_interval.setEnabled(False)
        for region in self._regions:
            span = region.bounds.get("sample")
            suffix = (
                f" [{span.start}, {span.stop})" if span is not None else ""
            )
            self.listing.addItem(
                f"{region.kind.capitalize()}: {region.label}{suffix}"
            )
        if key is not None:
            self.select_key(key)

    def select_key(self, key: str) -> None:
        """Inspect a region identified by a plot or another consumer.

        Args:
            key: Source identifier.
        """
        for index, region in enumerate(self._regions):
            if region.key == key:
                self.listing.setCurrentRow(index)
                break

    def _inspect(self, row: int) -> None:
        """Show the selected region's original metadata and source identifier.

        Args:
            row: Selected list row, or -1 when cleared.
        """
        if not 0 <= row < len(self._regions):
            self.metadata.clear()
            self.select_interval.setEnabled(False)
            return
        region = self._regions[row]
        self.metadata.setPlainText(
            region.key
            + "\n\n"
            + json.dumps(dict(region.metadata), indent=2, ensure_ascii=True)
        )
        span = region.bounds.get("sample")
        self.select_interval.setEnabled(
            span is not None
            and type(span.start) is int
            and type(span.stop) is int
        )

    def _select_interval(self) -> None:
        """Request the inspected sample range without mutating metadata."""
        row = self.listing.currentRow()
        if 0 <= row < len(self._regions):
            span = self._regions[row].bounds.get("sample")
            if (
                span is not None
                and type(span.start) is int
                and type(span.stop) is int
            ):
                self.interval_selected.emit(span.start, span.stop)
