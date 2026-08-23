"""Tests for store state after imports into new and existing stores."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from sigmf import SigMFFile

from sigmf_zarr import (
    SigMFZarrStore,
    import_radioml2016_dataset,
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


def _import_sigmf_with_format_2_shards(store_path: Path) -> None:
    """Import SigMF with sharding that fails after opening the store.

    Args:
        store_path: Target store path.
    """
    import_sigmf(
        store_path,
        write_source(store_path.parent),
        sample_shards=(64,),
        zarr_format=2,
    )


def _import_radioml_with_format_2_shards(store_path: Path) -> None:
    """Import RadioML with sharding that fails after opening the store.

    Args:
        store_path: Target store path.
    """
    import_radioml2016_dataset(
        store_path,
        {("BPSK", 0): np.zeros((2, 2, 4), dtype=np.float32)},
        sample_shards=(1, 2, 4),
        zarr_format=2,
    )


FAILING_IMPORTS = [
    _import_sigmf_with_format_2_shards,
    _import_radioml_with_format_2_shards,
]
"""Imports that open the target store and then raise `ValueError`."""


@pytest.mark.parametrize("failing_import", FAILING_IMPORTS)
def test_failed_import_preserves_existing_store(
    tmp_path: Path, failing_import: Callable[[Path], None]
) -> None:
    """A failed import must not remove a store that existed before it.

    Args:
        tmp_path: Pytest temporary path fixture.
        failing_import: Import that fails after opening its target.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=2)
    store.recordings.create("other", sample_shape=(2,))
    with pytest.raises(ValueError, match="requires Zarr format 3"):
        failing_import(store_path)
    assert SigMFZarrStore.open(store_path).list_recordings() == ("other",)


@pytest.mark.parametrize("failing_import", FAILING_IMPORTS)
def test_failed_import_preserves_existing_directory(
    tmp_path: Path, failing_import: Callable[[Path], None]
) -> None:
    """A failed import must not remove a directory that existed before it.

    Args:
        tmp_path: Pytest temporary path fixture.
        failing_import: Import that fails after opening its target.
    """
    store_path = tmp_path / "store.zarr"
    store_path.mkdir()
    with pytest.raises(ValueError, match="requires Zarr format 3"):
        failing_import(store_path)
    assert store_path.is_dir()
