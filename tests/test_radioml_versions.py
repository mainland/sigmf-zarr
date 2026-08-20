"""Regression tests for RadioML dataset version metadata."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from sigmf_zarr.json import JSONObject
from sigmf_zarr.radioml2016 import import_radioml2016_dataset
from sigmf_zarr.radioml2018 import import_radioml2018_dataset
from sigmf_zarr.store import SigMFZarrStore


@pytest.mark.parametrize(
    ("dataset_version", "expected_version"),
    [(None, "2016"), ("2016.10B", "2016.10B")],
)
def test_radioml2016_dataset_version_metadata(
    tmp_path: Path,
    dataset_version: str | None,
    expected_version: str,
) -> None:
    """Persist the version argument without modifying caller metadata.

    Args:
        tmp_path: Pytest temporary directory.
        dataset_version: Explicit version, or None to use the default.
        expected_version: Version expected in the imported recording.
    """
    store_path = tmp_path / "radioml2016.sigmf-zarr"
    metadata: JSONObject = {
        "radioml:dataset_version": "caller-version",
        "core:description": "RadioML version regression fixture",
    }
    original_metadata = metadata.copy()
    version_options = (
        {} if dataset_version is None else {"dataset_version": dataset_version}
    )

    import_radioml2016_dataset(
        store_path,
        {("BPSK", 0): np.zeros((1, 2, 4), dtype=np.float32)},
        global_metadata=metadata,
        **version_options,
    )

    recording = SigMFZarrStore.open(store_path).recordings["radioml2016"]
    assert recording.global_metadata["radioml:dataset_version"] == (
        expected_version
    )
    assert recording.global_metadata["core:description"] == (
        original_metadata["core:description"]
    )
    assert metadata == original_metadata


@pytest.mark.parametrize(
    ("dataset_version", "expected_version"),
    [(None, "2018"), ("2018.01A", "2018.01A")],
)
def test_radioml2018_dataset_version_metadata(
    tmp_path: Path,
    dataset_version: str | None,
    expected_version: str,
) -> None:
    """Persist the version argument without modifying caller metadata.

    Args:
        tmp_path: Pytest temporary directory.
        dataset_version: Explicit version, or None to use the default.
        expected_version: Version expected in the imported recording.
    """
    source_path = tmp_path / "radioml2018.hdf5"
    store_path = tmp_path / "radioml2018.sigmf-zarr"
    with h5py.File(source_path, "w") as source:
        source.create_dataset("X", data=np.zeros((1, 4, 2), dtype=np.float32))
        source.create_dataset("Y", data=np.ones((1, 1), dtype=np.float32))
        source.create_dataset("Z", data=np.zeros((1, 1), dtype=np.int16))
    metadata: JSONObject = {
        "radioml:dataset_version": "caller-version",
        "core:description": "RadioML version regression fixture",
    }
    original_metadata = metadata.copy()
    version_options = (
        {} if dataset_version is None else {"dataset_version": dataset_version}
    )

    import_radioml2018_dataset(
        store_path,
        source_path,
        modulation_classes=("BPSK",),
        global_metadata=metadata,
        **version_options,
    )

    recording = SigMFZarrStore.open(store_path).recordings["radioml2018"]
    assert recording.global_metadata["radioml:dataset_version"] == (
        expected_version
    )
    assert recording.global_metadata["core:description"] == (
        original_metadata["core:description"]
    )
    assert metadata == original_metadata
