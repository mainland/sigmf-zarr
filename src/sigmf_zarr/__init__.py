"""Top-level package for the SigMF-Zarr extension project."""

from importlib.metadata import PackageNotFoundError, version

from sigmf_zarr.provenance import capture_inputs, verify_inputs
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.store import (
    ChecksumName,
    SigMFCollection,
    SigMFRecording,
    SigMFZarrStore,
    ZarrFormat,
)

__all__ = [
    "__version__",
    "ChecksumName",
    "SigMFCollection",
    "SigMFRecording",
    "SigMFZarrStore",
    "ZarrFormat",
    "ReadOnlyArray",
    "ReadOnlyGroup",
    "capture_inputs",
    "verify_inputs",
]

try:
    __version__ = version("sigmf-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0"
