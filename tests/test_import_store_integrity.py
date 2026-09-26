"""Tests for store state after imports into new and existing stores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sigmf import SigMFFile

from sigmf_zarr import (
    SigMFZarrStore,
    import_sigmf,
)
from tests.testdata import TEST_FLOAT32_DATA, TEST_METADATA


def write_source(path: Path) -> Path:
    """Write a small standard SigMF recording.

    Args:
        path: Directory for the recording pair.

    Returns:
        Metadata file path.
    """
    data_path = path / "source.sigmf-data"
    TEST_FLOAT32_DATA.tofile(data_path)
    sigmf_file = SigMFFile(metadata=TEST_METADATA, data_file=data_path)
    meta_path = path / "source.sigmf-meta"
    sigmf_file.tofile(meta_path)
    return meta_path


def test_import_beside_unhashed_recording_leaves_store_hash_absent(
    tmp_path: Path,
) -> None:
    """Import must succeed when another recording has no hashes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    store.recordings.create("other", sample_shape=(2,))
    imported = import_sigmf(store_path, write_source(tmp_path))
    assert imported.verify_integrity()
    reopened = SigMFZarrStore.open(store_path)
    assert reopened.metadata_sha512 is None
    np.testing.assert_array_equal(
        reopened.recordings["source"].samples[:], TEST_FLOAT32_DATA
    )


def test_import_into_hashed_store_keeps_store_hash_current(
    tmp_path: Path,
) -> None:
    """Import into a fully hashed store must leave every hash current.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    store.recordings.create("other", sample_shape=(2,))
    store.update_integrity()
    import_sigmf(store_path, write_source(tmp_path))
    assert SigMFZarrStore.open(store_path).verify_integrity()
