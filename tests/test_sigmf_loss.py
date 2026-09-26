"""Interchange must not silently discard unsupported metadata."""

import json
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.sigmf import export_sigmf, import_sigmf
from sigmf_zarr.store import SigMFZarrStore


@pytest.mark.parametrize("kind", ["indexes", "extensions", "channels"])
def test_export_requires_permission_for_metadata_loss(
    tmp_path: Path, kind: str
) -> None:
    """Reject omissions even with force, and report explicitly allowed loss.

    Args:
        tmp_path: Temporary directory fixture.
        kind: Native metadata structure without a standard representation.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", sample_shape=(2, 4), sample_axes=("channel", "time"),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.set_samples(np.arange(8, dtype=np.float32).reshape(2, 4))
        if kind == "indexes":
            recording.add_index(
                "label", np.arange(4), axis="time", field="test:label"
            )
        elif kind == "extensions":
            recording.add_extension_array("test/calibration", [1, 2])
        else:
            recording.set_channel_metadata(0, {"test:antenna": "north"})
        target = tmp_path / "output.sigmf-meta"
        for force in (False, True):
            with pytest.raises(ValueError, match="allow_lossy=True"):
                export_sigmf(store, "rec", target, force=force)
            assert not target.exists()
            assert not target.with_suffix(".sigmf-data").exists()
        with pytest.warns(UserWarning, match="Standard SigMF export omits"):
            export_sigmf(store, "rec", target, allow_lossy=True)
        assert target.exists()
        assert recording.verify_sha512() is False


@pytest.mark.parametrize(
    "kind", ["top_level", "header", "trailing", "metadata_only", "dataset"]
)
def test_import_rejects_unsupported_structures_before_store_creation(
    tmp_path: Path, kind: str
) -> None:
    """Reject unsupported source layouts explicitly before any writes.

    Args:
        tmp_path: Temporary directory fixture.
        kind: Unsupported metadata or dataset structure.
    """
    metadata = {
        "global": {"core:datatype": "rf32_le", "core:version": "1.2.6"},
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }
    if kind == "top_level":
        metadata["test:objects"] = {"calibration": [1, 2]}
        error = "top-level metadata"
    elif kind == "header":
        metadata["captures"][0]["core:header_bytes"] = 4
        error = "header or trailing bytes"
    elif kind == "trailing":
        metadata["global"]["core:trailing_bytes"] = 4
        error = "header or trailing bytes"
    elif kind == "dataset":
        metadata["global"]["core:dataset"] = "../outside.dat"
        error = "same directory"
    else:
        metadata["global"]["core:metadata_only"] = True
        error = "Metadata-only"
    source = tmp_path / "source.sigmf-meta"
    source.write_text(json.dumps(metadata))
    store_path = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match=error):
        import_sigmf(store_path, source)
    assert not store_path.exists()


def test_export_rejects_unrepresented_header_bytes(tmp_path: Path) -> None:
    """Never publish header metadata alongside a sample-only dataset.

    Args:
        tmp_path: Temporary directory fixture.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        store.recordings.open(
            "rec", sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:datatype": "rf32_le"},
            captures=[{"core:sample_start": 0, "core:header_bytes": 4}],
        )
        target = tmp_path / "output.sigmf-meta"
        with pytest.raises(ValueError, match="header or trailing bytes"):
            export_sigmf(store, "rec", target, allow_lossy=True, force=True)
        assert not target.exists()
        assert not target.with_suffix(".sigmf-data").exists()
