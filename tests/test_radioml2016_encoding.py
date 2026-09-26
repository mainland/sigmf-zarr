"""RadioML imports declare the interchange encoding of stored components."""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.integrity import calculate_array_sha512
from sigmf_zarr.json import json_object, json_object_list
from sigmf_zarr.radioml2016 import import_radioml2016_dataset


def test_radioml2016_declares_complex_float_encoding(tmp_path: Path) -> None:
    """Set encoding without requiring callers to supply SigMF metadata.

    Args:
        tmp_path: Temporary directory.
    """
    dataset = {("BPSK", 0): np.ones((1, 2, 8), dtype=np.float32)}
    with import_radioml2016_dataset(tmp_path / "rml.zarr", dataset) as store:
        recording = store.recordings["radioml2016"]
        assert recording.global_metadata["core:datatype"] == "cf32_le"
        provenance = json_object(
            recording.global_metadata["sigmf-zarr-provenance:import"],
            name="provenance",
        )
        sources = json_object_list(provenance["sources"], name="sources")
        assert sources[0]["sha512"] == calculate_array_sha512(
            dataset[("BPSK", 0)]
        )
        assert sources[0]["target_start"] == 0
        parameters = json_object(provenance["parameters"], name="parameters")
        assert parameters["modulation_classes"] == ["BPSK"]


def test_radioml2016_rejects_conflicting_encoding(tmp_path: Path) -> None:
    """Reject conflicts before creating the destination.

    Args:
        tmp_path: Temporary directory.
    """
    dataset = {("BPSK", 0): np.ones((1, 2, 8), dtype=np.float32)}
    path = tmp_path / "rml.zarr"
    with pytest.raises(ValueError, match="core:datatype"):
        import_radioml2016_dataset(
            path, dataset, global_metadata={"core:datatype": "rf32_le"}
        )
    assert not path.exists()
