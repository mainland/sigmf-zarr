"""Top-level package for the SigMF-Zarr extension project."""

from importlib.metadata import PackageNotFoundError, version

from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup

__all__ = [
    "__version__",
    "ReadOnlyArray",
    "ReadOnlyGroup",
]

try:
    __version__ = version("sigmf-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0"
