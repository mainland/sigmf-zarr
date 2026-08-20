"""Top-level package for the SigMF-Zarr extension project."""

from sigmf_zarr._version import __version__
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.store import (
    ChecksumName,
    SigMFCollection,
    SigMFZarrStore,
    ZarrFormat,
)

__all__ = [
    "ChecksumName",
    "ReadOnlyArray",
    "ReadOnlyGroup",
    "SigMFCollection",
    "SigMFZarrStore",
    "ZarrFormat",
    "__version__",
]
