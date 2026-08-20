"""Zarr-backed storage for SigMF-like recordings and collections."""

from zarr.core.common import AccessModeLiteral as _ZarrAccessMode

from sigmf_zarr.store._collection import SigMFCollection
from sigmf_zarr.store._common import ChecksumName, ZarrFormat
from sigmf_zarr.store._container import (
    SigMFCollections,
    SigMFRecordings,
    SigMFZarrStore,
)
from sigmf_zarr.store._recording import SigMFRecording

AccessModeLiteral = _ZarrAccessMode
"""Zarr access mode accepted by the `SigMFZarrStore` constructor.

The values are `"r"`, `"r+"`, `"a"`, `"w"`, and `"w-"`, as defined by
`zarr.core.common.AccessModeLiteral`. Modes `"r"` and `"r+"` require an
existing store. Mode `"a"` opens an existing store or initializes an empty
target. Modes `"w"` and `"w-"` create a store, and `"w"` replaces existing
content.
"""

__all__ = [
    "AccessModeLiteral",
    "ChecksumName",
    "SigMFCollection",
    "SigMFCollections",
    "SigMFRecording",
    "SigMFRecordings",
    "SigMFZarrStore",
    "ZarrFormat",
]
