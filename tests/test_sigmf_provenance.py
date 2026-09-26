"""SigMF conversion preserves source metadata values and ownership."""

import json
from pathlib import Path

import numpy as np

from sigmf_zarr.sigmf import export_sigmf, import_sigmf, strip_zarr_metadata
from sigmf_zarr.store import SigMFZarrStore


def test_recording_preserves_source_metadata(tmp_path: Path) -> None:
    """Keep source versions, nested extension values, and exact timestamps.

    Args:
        tmp_path: Temporary directory fixture.
    """
    source = tmp_path / "source.sigmf-meta"
    metadata = {
        "global": {
            "core:datatype": "rf32_le", "core:version": "1.0.0",
            "core:extensions": [
                {"name": "test", "version": "1.0.0", "optional": True}
            ],
            "test:nested": {"values": [None, True, {"name": "receiver"}]},
        },
        "captures": [{
            "core:sample_start": 0,
            "core:datetime": "2026-09-25T00:00:00.123456789Z",
            "test:gain": 7,
        }],
        "annotations": [{
            "core:sample_start": 1, "core:sample_count": 2,
            "test:label": {"name": "signal"},
        }],
    }
    source.write_text(json.dumps(metadata))
    original = source.read_bytes()
    data = np.arange(4, dtype="<f4").tobytes()
    source.with_suffix(".sigmf-data").write_bytes(data)
    store_path = tmp_path / "store.zarr"
    recording = import_sigmf(store_path, source)
    for key, value in metadata["global"].items():
        assert strip_zarr_metadata(recording.global_metadata)[key] == value
    assert "core:offset" not in recording.global_metadata
    assert "core:num_channels" not in recording.global_metadata
    before = recording.metadata()
    output = export_sigmf(
        SigMFZarrStore.open(store_path), "source",
        tmp_path / "output.sigmf-meta",
    )
    result = json.loads(output.read_text())
    for key, value in metadata["global"].items():
        assert result["global"][key] == value
    assert result["captures"] == metadata["captures"]
    assert result["annotations"] == metadata["annotations"]
    assert output.with_suffix(".sigmf-data").read_bytes() == data
    assert source.read_bytes() == original
    assert recording.metadata() == before
