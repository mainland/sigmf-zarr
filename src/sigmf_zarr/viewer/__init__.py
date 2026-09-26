"""Reusable dataset access for the optional desktop viewer.

Import GUI components explicitly from ``sigmf_zarr.viewer.qt``.
Importing this module requires neither Qt nor Matplotlib.
"""

from sigmf_zarr.viewer.data import (
    DatasetCatalog,
    DatasetSource,
    OverviewWindow,
    RecordingInfo,
    SampleWindow,
)

__all__ = [
    "DatasetCatalog",
    "DatasetSource",
    "OverviewWindow",
    "RecordingInfo",
    "SampleWindow",
]
