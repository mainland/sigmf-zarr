"""Top-level package for the SigMF-Zarr extension project."""

from importlib.metadata import PackageNotFoundError, version

from sigmf_zarr.cspb import (
    CSPB_MODULATION_CLASSES,
    CSPBSignalMetadata,
    CSPBTruth,
    CSPBTruthRecord,
    cspb_sample_shape,
    import_cspb_dataset,
    load_cspb_truth,
)
from sigmf_zarr.radioml2016 import (
    RadioML2016Dict,
    RadioML2016Key,
    RadioML2016Value,
    as_radioml2016_dict,
    flatten_radioml2016_dataset,
    import_radioml2016_dataset,
)
from sigmf_zarr.radioml2018 import (
    RADIOML2018_MODULATION_CLASSES,
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
from sigmf_zarr.tim import decode_tim, read_tim
from sigmf_zarr.validation import (
    ValidationIssue,
    ValidationReport,
    validate_store,
)

__all__ = [
    "__version__",
    "ChecksumName",
    "CSPB_MODULATION_CLASSES",
    "CSPBSignalMetadata",
    "CSPBTruth",
    "CSPBTruthRecord",
    "SigMFCollection",
    "SigMFRecording",
    "SigMFZarrStore",
    "ZarrFormat",
    "RADIOML2018_MODULATION_CLASSES",
    "RadioML2016Dict",
    "RadioML2016Key",
    "RadioML2016Value",
    "ReadOnlyArray",
    "ReadOnlyGroup",
    "as_radioml2016_dict",
    "calculate_sha512",
    "cspb_sample_shape",
    "decode_tim",
    "export_sigmf",
    "export_sigmf_archive",
    "flatten_radioml2016_dataset",
    "import_sigmf",
    "import_sigmf_archive",
    "import_radioml2016_dataset",
    "import_radioml2018_dataset",
    "import_cspb_dataset",
    "load_cspb_truth",
    "read_tim",
    "verify_sha512",
    "ValidationIssue",
    "ValidationReport",
    "validate_store",
]

try:
    __version__ = version("sigmf-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0"
