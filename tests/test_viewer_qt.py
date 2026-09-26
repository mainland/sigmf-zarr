"""Offscreen desktop integration, navigation, and worker lifecycle tests."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Generator
from pathlib import Path
from threading import Event, get_ident

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from sigmf_zarr.store import SigMFZarrStore
from sigmf_zarr.viewer import DatasetSource, SampleWindow
from sigmf_zarr.viewer._tasks import Tasks
from sigmf_zarr.viewer.plots import mean_power, peak_magnitude, rms, spectrum
from sigmf_zarr.viewer.qt import DatasetViewer, MatplotlibPanel


@pytest.fixture(scope="module")
def app() -> QApplication:
    """Keep one Qt application alive throughout the GUI tests.

    Returns:
        Offscreen application.
    """
    return QApplication.instance() or QApplication([])


def wait_for(app: QApplication, predicate: Callable[[], bool]) -> None:
    """Process GUI events until a condition succeeds or a test times out.

    Args:
        app: Qt application.
        predicate: Completion condition.

    Raises:
        AssertionError: If the operation does not complete within ten seconds.
    """
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate(), "Timed out waiting for the viewer"


@pytest.fixture
def viewer(
    app: QApplication, tmp_path: Path
) -> Generator[DatasetViewer, None, None]:
    """Open a real batched store in an embeddable viewer.

    Args:
        app: Qt application.
        tmp_path: Temporary directory.

    Yields:
        Viewer with 130 browsable items and multiple indexes.
    """
    path = tmp_path / "dataset.zarr"
    store = SigMFZarrStore.create(path)
    rec = store.recordings.open(
        "rec",
        batched=True,
        sample_shape=(2, 32),
        sample_axes=("iq", "time"),
        global_metadata={"test:class": "shared"},
    )
    samples = np.ones((130, 2, 32), dtype=np.float32)
    samples[:, 0] *= np.arange(130)[:, None]
    rec.append_samples(samples)
    rec.add_index(
        "class",
        np.arange(130) % 2,
        axis="item",
        field="test:class",
        labels=["BPSK", "QPSK"],
    )
    rec.add_index("snr", np.arange(130), axis="item", field="test:snr")
    rec.set_item_metadata(
        [
            {"global": {"test:keep": i % 3 == 0, "test:class": "item"}}
            for i in range(130)
        ]
    )
    store.collections.open(
        "Group A",
        recording_ids=["rec"],
        metadata={"description": "First group"},
    )
    store.collections.open("Group B", recording_ids=["rec", "missing"])
    store.collections.open("Empty", recording_ids=[])
    widget = DatasetViewer(DatasetSource(path))
    widget.resize(1200, 850)
    widget.show()
    try:
        wait_for(app, lambda: "Samples [0, 32)" in widget.status.text())
        yield widget
    finally:
        widget.close()
        app.processEvents()


def test_item_slider_and_precise_navigation(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Slider and numeric navigation should select original individual items.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    assert viewer.item_slider.maximum() == 129
    assert viewer.item_position.value() == 1
    assert not viewer.previous_item.isEnabled()
    viewer.item_slider.setValue(127)
    viewer.step_item(1)
    wait_for(app, lambda: "Item 128 |" in viewer.status.text())
    assert viewer.item_position.value() == 129
    viewer.item_position.setValue(130)
    wait_for(app, lambda: "Item 129 |" in viewer.status.text())
    assert viewer.item_slider.value() == 129
    assert not viewer.next_item.isEnabled()
    viewer.step_item(-1)
    wait_for(app, lambda: "Item 128 |" in viewer.status.text())
    viewer.timeline.set_range(28, 32)
    wait_for(app, lambda: "Samples [28, 32)" in viewer.status.text())


def test_filters_and_selected_index_values(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Index queries should filter original items and display decoded labels.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    viewer.index_selection.item(0).setSelected(True)
    wait_for(
        app,
        lambda: "BPSK" in viewer.index_text.text(),
    )
    viewer.index_query.setPlainText('label(class) == "QPSK"\nand snr >= 3')
    viewer.apply_filters()
    wait_for(app, lambda: viewer.item_slider.maximum() == 63)
    wait_for(app, lambda: "Item 3 |" in viewer.status.text())
    assert viewer.item_position.value() == 1
    assert "Original item 3" in viewer.selection_text.text()
    assert "QPSK" in viewer.index_text.text()
    viewer.step_item(1)
    wait_for(app, lambda: "Item 5 |" in viewer.status.text())
    viewer.index_query.setPlainText("snr > 1000")
    viewer.apply_filters()
    wait_for(app, lambda: viewer.status.text() == "No matching items")
    assert not viewer.item_slider.isEnabled()
    assert viewer._item is None
    assert viewer.metadata.toPlainText() == ""
    assert "shared" in viewer.recording_metadata.toPlainText()
    assert viewer.item_metadata.toPlainText() == ""
    assert viewer.index_text.text() == ""
    assert not viewer.item_navigation.isVisible()
    assert all(not viewer.tabs.isTabVisible(i) for i in range(4))
    assert not viewer.item_position.isEnabled()


def test_query_error_retains_previous_selection(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Bad expressions must report errors without publishing partial results.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    viewer.index_query.setPlainText("[")
    viewer.apply_filters()
    wait_for(app, lambda: viewer.status.text().startswith("Error:"))
    assert len(viewer._positions) == 130
    viewer.cancel_filter()
    assert "previous selection retained" in viewer.status.text()


def test_multiline_query_keyboard_controls(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Keyboard controls support multiline queries and restoring all items.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    viewer.activateWindow()
    viewer.index_query.setFocus()
    wait_for(app, viewer.index_query.hasFocus)
    QTest.keyClicks(viewer.index_query, "snr >= 125")
    QTest.keyClick(viewer.index_query, Qt.Key.Key_Return)
    QTest.keyClicks(viewer.index_query, 'and label(class) == "QPSK"')
    assert "\n" in viewer.index_query.toPlainText()
    assert len(viewer._positions) == 130
    QTest.keyClick(
        viewer.index_query,
        Qt.Key.Key_Return,
        Qt.KeyboardModifier.ControlModifier,
    )
    wait_for(app, lambda: "Item 125 |" in viewer.status.text())
    assert list(viewer._positions) == [125, 127, 129]
    viewer.index_query.clear()
    QTest.keyClick(
        viewer.index_query,
        Qt.Key.Key_Enter,
        Qt.KeyboardModifier.ControlModifier,
    )
    wait_for(app, lambda: len(viewer._positions) == 130)
    assert viewer.applied.text() == "Applied: All items"


def test_custom_panels_and_worker_metrics(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Consumers should register ordinary widgets and independent metrics.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    main_thread = get_ident()
    threads = []
    received = []
    label = QLabel()

    def update(window: SampleWindow | None) -> None:
        """Observe access during the test.

        Args:
            window: Selected sample window, or None.
        """
        assert get_ident() == main_thread
        if window is not None:
            received.append(window.item)
            label.setText(str(window.item))

    def metric(window: SampleWindow) -> float:
        """Observe access during the test.

        Args:
            window: Selected sample window, or None.

        Returns:
            Observed operation result.
        """
        threads.append(get_ident())
        return float(len(window.samples))

    viewer.add_panel("Custom", label, update)
    viewer._metrics = {"Length": metric}
    viewer.step_item(1)
    wait_for(app, lambda: received == [1])
    assert label.text() == "1"
    assert threads and all(thread != main_thread for thread in threads)
    assert "Length: 32" in viewer.metric_text.text()
    viewer.grab().save("/tmp/sigmf-zarr-viewer.png")


def test_superseded_work_and_close(app: QApplication) -> None:
    """Late results must never replace a newer selection or reach closed UI.

    Args:
        app: Offscreen Qt application.
    """
    parent = QLabel()
    values = []
    errors = []
    gate = Event()
    entered = Event()
    tasks = Tasks(parent, errors.append, lambda *args: None)

    def old(stop: Event, progress: Callable[[int, int], None]) -> object:
        """Observe access during the test.

        Args:
            stop: Cancellation flag.
            progress: Progress callback.

        Returns:
            Observed operation result.
        """
        entered.set()
        gate.wait(5)
        return "old"

    try:
        tasks.submit("window", old, values.append)
        wait_for(app, entered.is_set)
        tasks.submit("window", lambda *args: "new", values.append)
        wait_for(app, lambda: values == ["new"])
        gate.set()
        tasks.close()
        app.processEvents()
        assert values == ["new"]
        assert not errors
    finally:
        gate.set()
        tasks.close()


def test_metrics_and_spectrum() -> None:
    """Bin-centered complex tones should preserve amplitude and frequency."""
    samples = 2 * np.exp(2j * np.pi * np.arange(64) / 8)
    window = SampleWindow("rec", 0, 0, {}, samples, {}, {})
    assert mean_power(window) == pytest.approx(4)
    assert rms(window) == pytest.approx(2)
    assert peak_magnitude(window) == pytest.approx(2)
    figure = Figure()
    spectrum(figure, window)
    line = figure.axes[0].lines[0]
    peak = np.argmax(line.get_ydata())
    assert line.get_xdata()[peak] == 0.125
    assert line.get_ydata()[peak] == pytest.approx(10 * np.log10(4))


def test_matplotlib_panel_clear(app: QApplication) -> None:
    """Clearing the selection should remove an old plot.

    Args:
        app: Offscreen Qt application.
    """
    panel = MatplotlibPanel(spectrum)
    window = SampleWindow("rec", None, 0, {}, np.ones(8), {}, {})
    panel.set_window(window)
    assert panel.figure.axes
    panel.set_window(None)
    assert not panel.figure.axes
    panel.close()


def test_collection_and_recording_tree(
    app: QApplication,
    viewer: DatasetViewer,
) -> None:
    """Collections expand to recordings, with no redundant panel entries.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    collections = viewer.tree.topLevelItem(0)
    recordings = viewer.tree.topLevelItem(1)
    assert collections.text(0) == "Collections"
    assert recordings.text(0) == "Recordings"
    assert collections.childCount() == 3
    assert recordings.childCount() == 1
    recording = recordings.child(0)
    assert recording.childCount() == 0
    group = collections.child(1)
    group.setExpanded(True)
    assert group.text(0) == "Group A"
    assert group.child(0).text(0) == "rec"
    assert group.child(0).childCount() == 0
    viewer.tree.setCurrentItem(group)
    wait_for(app, lambda: viewer.status.text() == "Collection: Group A")
    assert "First group" in viewer.collection_metadata.toPlainText()
    assert not viewer.item_navigation.isVisible()
    assert not viewer.controls_scroll.isVisible()
    assert all(not viewer.tabs.isTabVisible(i) for i in range(4))
    assert viewer.tabs.currentWidget() is viewer.collection_metadata
    assert not viewer.item_slider.isEnabled()
    assert viewer._item is None
    assert not viewer.recording_controls.isEnabled()
    viewer.tree.setCurrentItem(group.child(0))
    wait_for(app, lambda: "Item 0 |" in viewer.status.text())
    assert viewer.item_slider.maximum() == 129
    assert viewer.item_slider.isEnabled()
    assert viewer.recording_controls.isEnabled()
    viewer.step_item(1)
    wait_for(app, lambda: "Item 1 |" in viewer.status.text())
    # Another reference to the same recording retains the selected item.
    viewer.tree.setCurrentItem(recording)
    viewer.tabs.setCurrentWidget(viewer.recording_metadata)
    assert viewer.tabs.currentWidget() is viewer.recording_metadata
    assert viewer._item == 1
    viewer.tabs.setCurrentIndex(0)
    assert viewer._item == 1
    assert collections.child(2).child(1).isDisabled()
    viewer.grab().save("/tmp/sigmf-zarr-viewer-tree.png")


def test_collection_selection_discards_pending_recording(
    app: QApplication,
    viewer: DatasetViewer,
) -> None:
    """A late recording read must not replace a collection selection.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    gate = Event()
    entered = Event()
    delivered = []

    def pending(stop: Event, progress: Callable[[int, int], None]) -> object:
        """Hold a recording result until the collection has opened.

        Args:
            stop: Cancellation request.
            progress: Worker progress callback.

        Returns:
            Dummy obsolete recording result.
        """
        entered.set()
        gate.wait(5)
        return "obsolete"

    try:
        viewer._tasks.submit("window", pending, delivered.append)
        wait_for(app, entered.is_set)
        empty = viewer.tree.topLevelItem(0).child(0)
        viewer.tree.setCurrentItem(empty)
        wait_for(app, lambda: viewer.status.text() == "Collection: Empty")
        gate.set()
        app.processEvents()
        viewer._tasks.poll()
        assert not delivered
        assert viewer._info is None
    finally:
        gate.set()


def test_collection_only_catalog(app: QApplication, tmp_path: Path) -> None:
    """A store without recordings still exposes its collection metadata.

    Args:
        app: Offscreen Qt application.
        tmp_path: Temporary directory.
    """
    path = tmp_path / "collections.zarr"
    store = SigMFZarrStore.create(path)
    store.collections.open("Empty", recording_ids=[])
    widget = DatasetViewer(DatasetSource(path), default_panels=False)
    try:
        wait_for(
            app, lambda: widget.status.text() == "This store has no recordings"
        )
        assert widget.tree.topLevelItem(1).childCount() == 0
        widget.tree.setCurrentItem(widget.tree.topLevelItem(0).child(0))
        wait_for(app, lambda: widget.status.text() == "Collection: Empty")
        assert (
            '"recording_ids": []' in widget.collection_metadata.toPlainText()
        )
    finally:
        widget.close()
        app.processEvents()


def test_slider_coalesces_reads(
    app: QApplication,
    viewer: DatasetViewer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rapid slider movement reads only the final requested item.

    Args:
        app: Offscreen Qt application.
        viewer: Populated item viewer.
        monkeypatch: Patch fixture.
    """
    read = viewer.source.read_window
    items: list[int | None] = []

    def observed(*args: object, **kwargs: object) -> SampleWindow:
        """Record the item passed to the worker read.

        Args:
            *args: Positional read arguments.
            **kwargs: Selected coordinates and item.

        Returns:
            Selected samples and metadata.
        """
        items.append(kwargs.get("item"))
        return read(*args, **kwargs)

    monkeypatch.setattr(viewer.source, "read_window", observed)
    for position in range(1, 30):
        viewer.item_slider.setValue(position)
    assert viewer.metadata.toPlainText() == ""
    assert viewer.item_position.value() == 30
    wait_for(app, lambda: "Item 29 |" in viewer.status.text())
    assert items == [29]
    viewer.grab().save("/tmp/sigmf-zarr-viewer-slider.png")


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_item_slider(
    app: QApplication,
    tmp_path: Path,
    count: int,
) -> None:
    """Empty and singleton recordings must not permit invalid item movement.

    Args:
        app: Offscreen Qt application.
        tmp_path: Temporary directory.
        count: Number of source items.
    """
    path = tmp_path / "data.zarr"
    store = SigMFZarrStore.create(path)
    recording = store.recordings.open(
        "rec",
        batched=True,
        sample_shape=(2, 8),
        sample_axes=("iq", "time"),
    )
    recording.append_samples(np.zeros((count, 2, 8), dtype=np.float32))
    viewer = DatasetViewer(DatasetSource(path), default_panels=False)
    try:
        wait_for(
            app,
            lambda: (
                viewer.status.text() == "No matching items"
                if count == 0
                else "Samples [0, 8)" in viewer.status.text()
            ),
        )
        assert not viewer.item_slider.isEnabled()
        assert not viewer.item_position.isEnabled()
        assert not viewer.previous_item.isEnabled()
        assert not viewer.next_item.isEnabled()
        assert not viewer.tabs.isTabVisible(
            viewer.tabs.indexOf(viewer.indexes)
        )
        assert not viewer.tabs.isTabVisible(
            viewer.tabs.indexOf(viewer.region_browser)
        )
        assert json.loads(viewer.recording_metadata.toPlainText())["global"]
        if count:
            assert (
                viewer.item_metadata.toPlainText()
                == "No stored item metadata."
            )
        else:
            assert viewer.item_metadata.toPlainText() == ""
        assert viewer.item_position.value() == count
        viewer.step_item(1)
        assert viewer.item_slider.value() == 0
    finally:
        viewer.close()
        app.processEvents()


def test_metadata_scopes_remain_separate(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Stored scopes must preserve overridden fields and exclude indexes.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    viewer.index_selection.item(0).setSelected(True)
    assert viewer.tabs.isTabVisible(viewer.tabs.indexOf(viewer.indexes))
    wait_for(app, lambda: "BPSK" in viewer.indexes.toPlainText())
    shared = json.loads(viewer.recording_metadata.toPlainText())
    item = json.loads(viewer.item_metadata.toPlainText())
    resolved = json.loads(viewer.metadata.toPlainText())
    indexes = json.loads(viewer.indexes.toPlainText())
    assert shared["global"]["test:class"] == "shared"
    assert item == {"global": {"test:class": "item", "test:keep": True}}
    assert resolved["global"]["test:class"] == "item"
    assert "indexes" not in resolved
    assert indexes["class"]["label"] == "BPSK"
    assert not viewer.tabs.isTabVisible(viewer.tabs.indexOf(viewer.metadata))
    viewer.show_resolved.setChecked(True)
    assert viewer.tabs.isTabVisible(viewer.tabs.indexOf(viewer.metadata))
    assert not viewer.tabs.isTabVisible(
        viewer.tabs.indexOf(viewer.channel_metadata)
    )
    group = viewer.tree.topLevelItem(0).child(1)
    viewer.tree.setCurrentItem(group)
    wait_for(app, lambda: viewer.status.text() == "Collection: Group A")
    assert viewer.tabs.currentWidget() is viewer.collection_metadata
    assert viewer.recording_metadata.toPlainText() == ""
    assert viewer.item_metadata.toPlainText() == ""
    assert viewer.indexes.toPlainText() == ""
    viewer.tree.setCurrentItem(group.child(0))
    viewer.tabs.setCurrentWidget(viewer.recording_metadata)
    wait_for(app, lambda: "Item 0 |" in viewer.status.text())
    assert viewer.tabs.currentWidget() is viewer.recording_metadata


def test_region_tab_follows_item_metadata(
    app: QApplication, tmp_path: Path
) -> None:
    """Region tabs must appear only for items with captures or annotations.

    Args:
        app: Offscreen Qt application.
        tmp_path: Temporary dataset directory.
    """
    path = tmp_path / "regions.zarr"
    store = SigMFZarrStore.create(path)
    recording = store.recordings.open(
        "rec", batched=True, sample_shape=(8,), sample_axes=("time",)
    )
    recording.append_samples(np.ones((3, 8), dtype=np.float32))
    recording.set_item_metadata(
        [
            {},
            {
                "annotations": [
                    {"core:sample_start": 1, "core:sample_count": 2}
                ]
            },
            {"captures": [{"core:sample_start": 0}]},
        ]
    )
    viewer = DatasetViewer(DatasetSource(path), default_panels=False)
    try:
        for item in (0, 1, 2, 0):
            if viewer._info is not None:
                viewer.item_slider.setValue(item)
            wait_for(
                app,
                lambda item=item: (
                    f"Item {item} | Samples" in viewer.status.text()
                ),
            )
            visible = viewer.tabs.isTabVisible(
                viewer.tabs.indexOf(viewer.region_browser)
            )
            assert visible == (item != 0)
            assert not viewer.tabs.isTabVisible(
                viewer.tabs.indexOf(viewer.indexes)
            )
    finally:
        viewer.close()
        app.processEvents()


def test_channel_metadata_follows_coordinate(
    app: QApplication, tmp_path: Path
) -> None:
    """Channel attributes must follow the selector without entering item JSON.

    Args:
        app: Offscreen Qt application.
        tmp_path: Temporary directory.
    """
    path = tmp_path / "channels.zarr"
    store = SigMFZarrStore.create(path)
    rec = store.recordings.open(
        "rec",
        batched=False,
        sample_dtype="complex64",
        sample_shape=(2, 8),
        sample_axes=("channel", "time"),
        channel_metadata=[{"test:receiver": "A"}, {"test:receiver": "B"}],
    )
    rec.set_samples(np.ones((2, 8), dtype=np.complex64))
    viewer = DatasetViewer(DatasetSource(path), default_panels=False)
    try:
        wait_for(app, lambda: '"A"' in viewer.channel_metadata.toPlainText())
        assert viewer.tabs.isTabVisible(
            viewer.tabs.indexOf(viewer.channel_metadata)
        )
        assert not viewer.tabs.isTabVisible(
            viewer.tabs.indexOf(viewer.item_metadata)
        )
        assert "test:receiver" not in viewer.recording_metadata.toPlainText()
        viewer._coordinates["channel"].setValue(1)
        assert viewer.channel_metadata.toPlainText() == ""
        wait_for(app, lambda: '"B"' in viewer.channel_metadata.toPlainText())
    finally:
        viewer.close()
        app.processEvents()


@pytest.mark.parametrize("branch", [0, 1])
def test_category_selection_clears_waveform(
    app: QApplication, viewer: DatasetViewer, branch: int
) -> None:
    """Category headings must not retain a previous recording's waveform.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
        branch: Collections or Recordings heading.
    """
    assert viewer.item_navigation.isVisible()
    assert viewer.tabs.isTabVisible(0)
    root = viewer.tree.topLevelItem(branch)
    viewer.tree.setCurrentItem(root)
    assert viewer.status.text() == root.text(0)
    assert viewer._info is None
    assert viewer._item is None
    assert not viewer.item_navigation.isVisible()
    assert not viewer.controls_scroll.isVisible()
    assert not viewer.tabs.isVisible()
    assert all(not viewer.tabs.isTabVisible(i) for i in range(4))
    app.processEvents()
    assert not viewer.tabs.isVisible()
    # Reopening a recording restores controls and plots after its read.
    viewer.tree.setCurrentItem(viewer.tree.topLevelItem(1).child(0))
    wait_for(app, lambda: "Item 0 |" in viewer.status.text())
    assert viewer.item_navigation.isVisible()
    assert viewer.tabs.isTabVisible(0)
    assert viewer.tabs.currentIndex() == 0


def test_waveform_tabs_follow_read_availability(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Pending reads hide plots and preserve the user's explicit tab choice.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    viewer.tabs.setCurrentIndex(2)
    assert viewer.timeline_controls.isVisible()
    viewer.step_item(1)
    assert all(not viewer.tabs.isTabVisible(i) for i in range(4))
    assert viewer.item_navigation.isVisible()
    assert viewer.timeline_controls.isVisible()
    wait_for(app, lambda: "Item 1 |" in viewer.status.text())
    assert viewer.tabs.currentIndex() == 2
    viewer.step_item(1)
    viewer.tabs.setCurrentWidget(viewer.indexes)
    assert not viewer.timeline_controls.isVisible()
    wait_for(app, lambda: "Item 2 |" in viewer.status.text())
    assert viewer.tabs.currentWidget() is viewer.indexes
    assert all(viewer.tabs.isTabVisible(i) for i in range(4))
    assert not viewer.timeline_controls.isVisible()
    for tab in (viewer.recording_metadata, viewer.item_metadata):
        viewer.tabs.setCurrentWidget(tab)
        assert not viewer.timeline_controls.isVisible()
    for index in range(4):
        viewer.tabs.setCurrentIndex(index)
        assert viewer.timeline_controls.isVisible()


def test_overview_fft_preserves_zoom(
    app: QApplication, tmp_path: Path
) -> None:
    """Background FFT updates retain zoom, while new selections reset it.

    Args:
        app: Offscreen Qt application.
        tmp_path: Temporary dataset directory.
    """
    from sigmf_zarr.viewer.qt import SpectrogramPanel

    path = tmp_path / "continuous.zarr"
    store = SigMFZarrStore.create(path)
    recording = store.recordings.open(
        "continuous",
        batched=False,
        sample_shape=(2, 100000),
        sample_axes=("iq", "time"),
    )
    recording.set_samples(np.ones((2, 100000), dtype=np.float32))
    widget = DatasetViewer(DatasetSource(path))
    widget.resize(1200, 850)
    widget.show()
    try:
        wait_for(app, lambda: "Reduced overview" in widget.status.text())
        panel = widget.tabs.widget(0)
        assert isinstance(panel, SpectrogramPanel)
        axes = panel.figure.axes[0]
        axes.set_ylim(-0.2, 0.2)
        axes.set_xlim(25000, 75000)
        panel.fft_size.setCurrentText("512")
        panel.overlap.setCurrentText("75%")
        assert panel.figure.axes[0] is axes
        wait_for(
            app,
            lambda: (
                panel._overview is not None
                and panel._overview.signal.fft_size == 512
                and panel._overview.signal.hop == 128
            ),
        )
        assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
        assert panel.figure.axes[0].get_xlim() == (25000, 75000)
        assert panel.window_function.currentText() == "Hann"
        original_power = panel._overview.signal.power.copy()
        panel.window_function.setCurrentText("Hamming")
        wait_for(
            app,
            lambda: (
                panel._overview is not None
                and panel._overview.signal.window_function == "hamming"
            ),
        )
        assert not np.allclose(panel._overview.signal.power, original_power)
        assert panel.figure.axes[0].get_xlim() == (25000, 75000)
        assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
        assert "Hamming" in widget.tabs.widget(2).figure.axes[0].get_title()
        panel.window_function.setCurrentText("Hann")
        wait_for(
            app,
            lambda: (
                panel._overview is not None
                and panel._overview.signal.window_function == "hann"
            ),
        )
        panel.toolbar.home()
        assert panel.figure.axes[0].get_ylim() != (-0.2, 0.2)
        panel.toolbar.back()
        assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
        wait_for(app, lambda: panel._window is not None)
        assert widget.timeline.selection == (25000, 75000)
        assert panel._window.samples.size == 50000
        assert panel.figure.axes[0].get_xlim() == (25000, 75000)
        assert widget.tabs.widget(1).figure.axes[0].get_xlim() == (
            25000,
            75000,
        )

        widget.timeline.fit()
        wait_for(app, lambda: panel._overview is not None)

        # A selection change during the FFT debounce must discard old zoom.
        panel.fft_size.setCurrentText("1024")
        widget.timeline.set_range(0, 1024)
        assert not panel.figure.axes
        wait_for(app, lambda: "Samples [0, 1024)" in widget.status.text())
        assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
        assert panel.figure.axes[0].get_xlim() != (25000, 75000)
    finally:
        widget.close()
        app.processEvents()


def test_spectrogram_controls_reuse_samples(
    app: QApplication, viewer: DatasetViewer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spectrogram settings must redraw cached data without storage reads.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
        monkeypatch: Patch fixture.
    """
    from sigmf_zarr.viewer.qt import SpectrogramPanel

    panel = viewer.tabs.widget(0)
    assert isinstance(panel, SpectrogramPanel)
    assert [viewer.tabs.tabText(i) for i in range(4)] == [
        "Spectrogram",
        "Time",
        "Frequency",
        "I/Q",
    ]
    viewer.tabs.setCurrentWidget(panel)
    assert len(panel.figure.axes) == 2

    def no_read(*args: object, **kwargs: object) -> SampleWindow:
        """Reject storage access from display-only settings."""
        raise AssertionError("Spectrogram settings reread samples")

    monkeypatch.setattr(viewer.source, "read_window", no_read)
    panel.figure.axes[0].set_ylim(-0.2, 0.2)
    panel.figure.axes[0].set_xlim(8, 24)
    panel.fft_size.setCurrentText("32")
    panel.overlap.setCurrentText("75%")
    assert panel.figure.axes[0].images[0].get_array().shape == (32, 1)
    assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
    assert panel.figure.axes[0].get_xlim() == (8, 24)
    panel.window_function.setCurrentText("Rectangular")
    assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
    assert panel.figure.axes[0].get_xlim() == (8, 24)
    frequency = viewer.tabs.widget(2).figure.axes[0]
    assert "Rectangular" in frequency.get_title()
    assert frequency.lines[0].get_ydata()[15] < -250
    panel.window_function.setCurrentText("Blackman-Harris")
    assert panel.analysis_settings()[2] == "blackmanharris"
    assert "Blackman-Harris" in panel.figure.axes[0].get_title()
    frequency = viewer.tabs.widget(2).figure.axes[0]
    assert "Blackman-Harris" in frequency.get_title()
    assert panel.figure.axes[0].get_xlim() == (8, 24)
    assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
    panel.window_function.setCurrentText("Hann")
    frequency = viewer.tabs.widget(2).figure.axes[0]
    assert frequency.lines[0].get_ydata()[15] == pytest.approx(
        -6.0206, abs=0.0001
    )
    panel.auto_level.setChecked(False)
    panel.maximum_db.setValue(10)
    panel.dynamic_range.setValue(40)
    assert panel.figure.axes[0].images[0].get_clim() == (-30, 10)
    assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
    assert panel.figure.axes[0].get_xlim() == (8, 24)
    panel.toolbar.home()
    assert panel.figure.axes[0].get_ylim() != (-0.2, 0.2)
    panel.toolbar.back()
    assert panel.figure.axes[0].get_ylim() == (-0.2, 0.2)
    axes = panel.figure.axes[0]
    artist = axes.images[0]
    axes.set_xlim(-0.1, 0.1)
    panel.colormap.setCurrentText("viridis")
    assert axes.images[0] is artist
    assert artist.get_cmap().name == "viridis"
    assert artist.get_clim() == (-30, 10)
    assert axes.get_xlim() == (-0.1, 0.1)
    viewer.tree.setCurrentItem(viewer.tree.topLevelItem(0))
    assert not viewer.tabs.isTabVisible(3)
    assert not panel.figure.axes
    assert panel._window is None
    panel.fft_size.setCurrentText("64")
    panel.colormap.setCurrentText("gray")
    assert not panel.figure.axes
    app.processEvents()


def test_plot_navigation_links_scrubber(
    app: QApplication, viewer: DatasetViewer
) -> None:
    """Actual toolbar zoom and pan must update samples and preserve frequency.

    Args:
        app: Offscreen Qt application.
        viewer: Populated record browser.
    """
    from sigmf_zarr.viewer.qt import SpectrogramPanel

    panel = viewer.tabs.widget(0)
    assert isinstance(panel, SpectrogramPanel)
    panel.canvas.draw()
    axes = panel.figure.axes[0]
    initial = viewer.timeline.selection
    axes.set_ylim(-0.25, 0.25)
    panel.toolbar.navigated.emit()
    assert viewer.timeline.selection == initial
    assert not viewer._window_timer.isActive()

    panel.toolbar.zoom()
    x1, y1 = axes.transData.transform((8, -0.2))
    x2, y2 = axes.transData.transform((24, 0.2))
    panel.toolbar.press_zoom(
        MouseEvent("button_press_event", panel.canvas, x1, y1, button=1)
    )
    panel.toolbar.release_zoom(
        MouseEvent("button_release_event", panel.canvas, x2, y2, button=1)
    )
    panel.toolbar.zoom()
    start, stop = viewer.timeline.selection
    assert 7 <= start <= 8
    assert 24 <= stop <= 25
    wait_for(
        app,
        lambda: (
            panel._window is not None
            and panel._window.start == start
            and len(panel._window.samples) == stop - start
        ),
    )
    assert panel.figure.axes[0].get_ylim() == pytest.approx(
        (-0.2, 0.2), abs=0.003
    )
    assert (
        viewer.tabs.widget(1).figure.axes[0].get_xlim()
        == panel.figure.axes[0].get_xlim()
    )
    panel.toolbar.home()
    assert viewer.timeline.selection == (0, 32)
    wait_for(
        app,
        lambda: panel._window is not None and len(panel._window.samples) == 32,
    )

    # Scrubbing keeps frequency limits but resets the visible sample interval.
    panel.figure.axes[0].set_ylim(-0.1, 0.1)
    viewer.timeline.set_range(4, 20)
    wait_for(
        app, lambda: panel._window is not None and panel._window.start == 4
    )
    assert panel.figure.axes[0].get_ylim() == (-0.1, 0.1)
    assert panel.figure.axes[0].get_xlim() == (4, 20)
    viewer.timeline.set_visible_range(4, 20)
    time_panel = viewer.tabs.widget(1)
    time_panel.figure.axes[0].set_xlim(16, 28)
    time_panel.toolbar.navigated.emit()
    assert viewer.timeline.selection == (16, 28)
    assert viewer.timeline.visible_range == (12, 28)
    wait_for(
        app, lambda: panel._window is not None and panel._window.start == 16
    )
    assert panel.figure.axes[0].get_xlim() == (16, 28)
    assert panel.figure.axes[0].get_ylim() == (-0.1, 0.1)
    viewer.step_item(1)
    wait_for(
        app, lambda: panel._window is not None and panel._window.item == 1
    )
    assert panel.figure.axes[0].get_ylim() != (-0.1, 0.1)


def test_annotation_hover_json(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hover must show escaped source JSON without changing the selected tab.

    Args:
        app: Offscreen Qt application.
        monkeypatch: Fixture replacing tooltip display.
    """
    from PySide6.QtWidgets import QToolTip

    from sigmf_zarr.viewer.qt import SpectrogramPanel

    annotation = {
        "core:sample_start": 8,
        "core:sample_count": 16,
        "core:description": "Packet <test>",
        "core:freq_lower_edge": -0.2,
        "core:freq_upper_edge": 0.2,
    }
    metadata = {"global": {"core:sample_rate": 1}, "annotations": [annotation]}
    window = SampleWindow(
        "test", None, 0, {}, np.ones(32, dtype=complex), metadata, {}
    )
    panel = SpectrogramPanel()
    shown = []
    hidden = []

    def show_tooltip(position: object, text: str, widget: object) -> None:
        """Record tooltip contents.

        Args:
            position: Screen position.
            text: Escaped JSON in HTML preformatted text.
            widget: Tooltip parent.
        """
        shown.append(text)

    monkeypatch.setattr(QToolTip, "showText", show_tooltip)
    monkeypatch.setattr(QToolTip, "hideText", lambda: hidden.append(True))
    try:
        panel.resize(900, 600)
        panel.show()
        panel.set_window(window)
        panel.canvas.draw()
        x, y = panel.figure.axes[0].transData.transform((16, 0))
        event = MouseEvent("motion_notify_event", panel.canvas, x, y)
        panel.canvas.callbacks.process("motion_notify_event", event)
        assert "Packet &lt;test&gt;" in shown[-1]
        assert "core:sample_start" in shown[-1]
        assert "&quot;" in shown[-1]
        hidden.clear()
        event = MouseEvent("motion_notify_event", panel.canvas, 0, 0)
        panel.canvas.callbacks.process("motion_notify_event", event)
        assert hidden
    finally:
        panel.close()
        app.processEvents()
