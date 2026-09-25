"""Offscreen desktop integration, navigation, and worker lifecycle tests."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from sigmf_zarr.viewer import SampleWindow
from sigmf_zarr.viewer.plots import mean_power, peak_magnitude, rms, spectrum
from sigmf_zarr.viewer.qt import MatplotlibPanel


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
