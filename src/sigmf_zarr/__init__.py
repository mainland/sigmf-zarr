"""Top-level package for the SigMF-Zarr extension project."""

from sigmf_zarr._version import __version__
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup

__all__ = [
    "ReadOnlyArray",
    "ReadOnlyGroup",
    "__version__",
]
