"""Top-level package for the SigMF-Zarr extension project."""

from importlib.metadata import PackageNotFoundError, version

from sigmf_zarr.provenance import capture_inputs, verify_inputs
from sigmf_zarr.radioml2016 import (
    RadioML2016Dict,
    RadioML2016Key,
    RadioML2016Value,
    as_radioml2016_dict,
    flatten_radioml2016_dataset,
    import_radioml2016_dataset,
)
from sigmf_zarr.radioml2018 import (
    import_radioml2018_dataset,
)
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
from sigmf_zarr.validation import (
    ValidationIssue,
    ValidationReport,
    validate_store,
)

__all__ = [
    "__version__",
    "ChecksumName",
    "SigMFCollection",
    "SigMFRecording",
    "SigMFZarrStore",
    "ZarrFormat",
    "RadioML2016Dict",
    "RadioML2016Key",
    "RadioML2016Value",
    "ReadOnlyArray",
    "ReadOnlyGroup",
    "as_radioml2016_dict",
    "calculate_sha512",
    "capture_inputs",
    "export_sigmf",
    "export_sigmf_archive",
    "flatten_radioml2016_dataset",
    "import_sigmf",
    "import_sigmf_archive",
    "import_radioml2016_dataset",
    "import_radioml2018_dataset",
    "verify_sha512",
    "verify_inputs",
    "ValidationIssue",
    "ValidationReport",
    "validate_store",
]

try:
    __version__ = version("sigmf-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0"
