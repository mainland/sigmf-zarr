"""Top-level package for the SigMF-Zarr extension project."""

from sigmf_zarr._version import __version__
from sigmf_zarr.provenance import capture_inputs, verify_inputs
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
    "ChecksumName",
    "ReadOnlyArray",
    "ReadOnlyGroup",
    "SigMFCollection",
    "SigMFRecording",
    "SigMFZarrStore",
    "ZarrFormat",
    "__version__",
    "calculate_sha512",
    "capture_inputs",
    "export_sigmf",
    "export_sigmf_archive",
    "import_sigmf",
    "import_sigmf_archive",
    "verify_inputs",
    "verify_sha512",
]
