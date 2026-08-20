"""Top-level package for the SigMF-Zarr extension project."""

from importlib.metadata import PackageNotFoundError, version

from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.sigmf import (
    calculate_sha512,
    export_sigmf,
    export_sigmf_archive,
    import_sigmf,
    import_sigmf_archive,
    verify_sha512,
)
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
    "calculate_sha512",
    "export_sigmf",
    "export_sigmf_archive",
    "import_sigmf",
    "import_sigmf_archive",
    "verify_sha512",
]

try:
    __version__ = version("sigmf-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0"
