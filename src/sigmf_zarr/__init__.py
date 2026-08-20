"""Top-level package for the SigMF-Zarr extension project."""

from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "__version__",
]

try:
    __version__ = version("sigmf-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0"
