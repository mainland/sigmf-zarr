"""Plot scaling, coordinates, and short-window behavior without Qt."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sigmf_zarr.viewer.data import SampleWindow
from sigmf_zarr.viewer.plots import (
    constellation,
)


def test_iq_square_and_equal_units() -> None:
    """I/Q must keep a square box and equal units for unequal sample spans."""
    window = SampleWindow(
        "rec", None, 0, {}, np.array([-10 - 1j, 10 + 1j]), {}, {}
    )
    figure = Figure(figsize=(10, 4), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    constellation(figure, window)
    axes = figure.axes[0]
    for width, height in [(10, 4), (4, 10)]:
        figure.set_size_inches(width, height)
        axes.set_xlim(-5, 5)
        axes.set_ylim(-1, 1)
        canvas.draw()
        bounds = axes.get_window_extent()
        assert bounds.width == pytest.approx(bounds.height)
        points = axes.transData.transform([(0, 0), (1, 0), (0, 1)])
        assert points[1, 0] - points[0, 0] == pytest.approx(
            points[2, 1] - points[0, 1]
        )
