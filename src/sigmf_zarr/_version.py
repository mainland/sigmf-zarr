"""Installed SigMF-Zarr package version."""

from importlib.metadata import PackageNotFoundError, version

UNKNOWN_VERSION = "0+unknown"
"""Version reported when installed package metadata is unavailable."""

try:
    __version__ = version("sigmf-zarr")
except PackageNotFoundError:
    # A source tree without installed metadata has no reliable version.
    __version__ = UNKNOWN_VERSION

__all__ = ["UNKNOWN_VERSION", "__version__"]
