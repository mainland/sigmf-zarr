"""Top-level package for the SigMF-Zarr extension project."""

from sigmf_zarr._version import __version__
from sigmf_zarr.cspb import (
    CSPB_MODULATION_CLASSES,
    CSPBSignalMetadata,
    CSPBTruth,
    CSPBTruthRecord,
    cspb_sample_shape,
    import_cspb_dataset,
    load_cspb_truth,
)
from sigmf_zarr.export_plan import SigMFExportPlan, plan_sigmf_export
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
    "CSPB_MODULATION_CLASSES",
    "RADIOML2018_MODULATION_CLASSES",
    "CSPBSignalMetadata",
    "CSPBTruth",
    "CSPBTruthRecord",
    "ChecksumName",
    "RadioML2016Dict",
    "RadioML2016Key",
    "RadioML2016Value",
    "ReadOnlyArray",
    "ReadOnlyGroup",
    "SigMFCollection",
    "SigMFExportPlan",
    "SigMFRecording",
    "SigMFZarrStore",
    "ValidationIssue",
    "ValidationReport",
    "ZarrFormat",
    "__version__",
    "as_radioml2016_dict",
    "calculate_sha512",
    "capture_inputs",
    "cspb_sample_shape",
    "decode_tim",
    "export_sigmf",
    "export_sigmf_archive",
    "flatten_radioml2016_dataset",
    "import_cspb_dataset",
    "import_radioml2016_dataset",
    "import_radioml2018_dataset",
    "import_sigmf",
    "import_sigmf_archive",
    "load_cspb_truth",
    "plan_sigmf_export",
    "read_tim",
    "validate_store",
    "verify_inputs",
    "verify_sha512",
]
