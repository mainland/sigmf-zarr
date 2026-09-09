"""Tests for the target store that every importer opens or replaces."""

from __future__ import annotations

import os
import shutil
import struct
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from sigmf import SigMFFile

import sigmf_zarr._import_target as import_target_module
import sigmf_zarr.sigmf as sigmf_module
from sigmf_zarr import (
    SigMFZarrStore,
    import_cspb_dataset,
    import_panoradio_dataset,
    import_radioml2016_dataset,
    import_radioml2018_dataset,
    import_sigmf,
    import_sigmf_archive,
)
from sigmf_zarr.export_plan import validate_sigmf_schema
from tests.testdata import TEST_FLOAT32_DATA, TEST_METADATA

type Importer = Callable[..., object]
"""Function that imports sources from a directory into a store path."""


def _sigmf_pair(directory: Path) -> Path:
    """Write a small standard SigMF recording.

    Args:
        directory: Existing directory for the recording pair.

    Returns:
        Metadata file path.
    """
    data_path = directory / "source.sigmf-data"
    TEST_FLOAT32_DATA.tofile(data_path)
    meta_path = directory / "source.sigmf-meta"
    SigMFFile(metadata=TEST_METADATA, data_file=data_path).tofile(meta_path)
    return meta_path


def _import_sigmf(store_path: Path, directory: Path, **options: Any) -> None:
    """Import a standard SigMF recording written into `directory`.

    Args:
        store_path: Target store path.
        directory: Existing directory for the source files.
        **options: Importer keyword arguments.
    """
    import_sigmf(store_path, _sigmf_pair(directory), **options)


def _import_sigmf_archive(
    store_path: Path, directory: Path, **options: Any
) -> None:
    """Import a SigMF archive written into `directory`.

    Args:
        store_path: Target store path.
        directory: Existing directory for the source files.
        **options: Importer keyword arguments.
    """
    root = directory / "bundle"
    root.mkdir()
    _sigmf_pair(root)
    archive = directory / "bundle.sigmf"
    with tarfile.open(archive, "w") as output:
        output.add(root, arcname="bundle")
    shutil.rmtree(root)
    import_sigmf_archive(store_path, archive, **options)


def _import_cspb(store_path: Path, directory: Path, **options: Any) -> None:
    """Import a CSPB directory written into `directory`.

    Args:
        store_path: Target store path.
        directory: Existing directory for the source files.
        **options: Importer keyword arguments.
    """
    source = directory / "cspb"
    source.mkdir()
    components = np.arange(6, dtype="<f4")
    (source / "signal_1.tim").write_bytes(
        struct.pack("<ii", 2, 3) + components.tobytes()
    )
    import_cspb_dataset(store_path, source, **options)


def _import_radioml2016(
    store_path: Path, directory: Path, **options: Any
) -> None:
    """Import an in-memory RadioML 2016 mapping.

    Args:
        store_path: Target store path.
        directory: Unused source directory.
        **options: Importer keyword arguments.
    """
    del directory
    dataset = {("BPSK", 0): np.zeros((2, 2, 4), dtype=np.float32)}
    import_radioml2016_dataset(store_path, dataset, **options)


def _import_radioml2018(
    store_path: Path, directory: Path, **options: Any
) -> None:
    """Import a RadioML 2018 HDF5 file written into `directory`.

    Args:
        store_path: Target store path.
        directory: Existing directory for the source files.
        **options: Importer keyword arguments.
    """
    source = directory / "RML2018.hdf5"
    with h5py.File(source, "w") as output:
        output.create_dataset("X", data=np.zeros((2, 4, 2), dtype=np.float32))
        output.create_dataset(
            "Y", data=np.array([[1, 0], [0, 1]], dtype=np.float32)
        )
        output.create_dataset("Z", data=np.array([[0], [2]], dtype=np.int16))
    import_radioml2018_dataset(
        store_path, source, modulation_classes=("BPSK", "QPSK"), **options
    )


def _import_panoradio(
    store_path: Path, directory: Path, **options: Any
) -> None:
    """Import a Panoradio dataset written into `directory`.

    Args:
        store_path: Target store path.
        directory: Existing directory for the source files.
        **options: Importer keyword arguments.
    """
    source = directory / "pano.npy"
    np.save(source, np.ones((2, 4), dtype=np.complex64))
    tags = directory / "tags.csv"
    tags.write_text("idx, mode, snr\n0, AM, 0\n1, USB, 2\n", encoding="ascii")
    import_panoradio_dataset(store_path, source, tags, **options)


IMPORTERS: dict[str, Importer] = {
    "sigmf": _import_sigmf,
    "sigmf-archive": _import_sigmf_archive,
    "cspb": _import_cspb,
    "radioml2016": _import_radioml2016,
    "radioml2018": _import_radioml2018,
    "panoradio": _import_panoradio,
}
"""Importers with an `overwrite_store` option, by name."""

FILE_IMPORTERS = [name for name in IMPORTERS if name != "radioml2016"]
"""Importers that read source files."""


def _existing_store(path: Path) -> None:
    """Create a SigMF-Zarr store with one recording named `old`.

    Args:
        path: Store path.
    """
    store = SigMFZarrStore.create(path)
    store.recordings.create("old", sample_shape=(2,))


@pytest.mark.parametrize("importer", IMPORTERS)
def test_overwrite_store_refuses_a_directory_that_is_not_a_store(
    tmp_path: Path, importer: str
) -> None:
    """Replacing a store must never delete an unrelated directory.

    Args:
        tmp_path: Pytest temporary path fixture.
        importer: Importer name.
    """
    target = tmp_path / "Documents"
    target.mkdir()
    (target / "thesis.tex").write_text("precious")
    sources = tmp_path / "sources"
    sources.mkdir()
    with pytest.raises(ValueError, match="not a SigMF-Zarr store"):
        IMPORTERS[importer](target, sources, overwrite_store=True)
    assert [path.name for path in target.iterdir()] == ["thesis.tex"]


@pytest.mark.parametrize("importer", FILE_IMPORTERS)
def test_overwrite_store_refuses_a_store_that_contains_a_source(
    tmp_path: Path, importer: str
) -> None:
    """Replacing a store must never delete the files being imported.

    Args:
        tmp_path: Pytest temporary path fixture.
        importer: Importer name.
    """
    target = tmp_path / "store.zarr"
    _existing_store(target)
    before = sorted(path.name for path in target.iterdir())
    with pytest.raises(ValueError, match="contains the import source"):
        IMPORTERS[importer](target, target, overwrite_store=True)
    after = sorted(path.name for path in target.iterdir())
    assert set(before) < set(after)
    assert (target / "recordings" / "old" / "zarr.json").is_file()


@pytest.mark.parametrize("importer", IMPORTERS)
def test_overwrite_store_checks_chunks_before_deleting(
    tmp_path: Path, importer: str
) -> None:
    """Invalid sample chunks must leave the store to be replaced unchanged.

    Args:
        tmp_path: Pytest temporary path fixture.
        importer: Importer name.
    """
    target = tmp_path / "store.zarr"
    _existing_store(target)
    sources = tmp_path / "sources"
    sources.mkdir()
    with pytest.raises(ValueError, match="positive integers"):
        IMPORTERS[importer](
            target, sources, overwrite_store=True, sample_chunks=(0,)
        )
    assert SigMFZarrStore.open(target).list_recordings() == ("old",)


@pytest.mark.parametrize("importer", IMPORTERS)
@pytest.mark.parametrize("existing", ["store", "empty", "absent"])
def test_overwrite_store_replaces_a_store_or_empty_directory(
    tmp_path: Path, importer: str, existing: str
) -> None:
    """A store, an empty directory, or no entry may be replaced.

    Args:
        tmp_path: Pytest temporary path fixture.
        importer: Importer name.
        existing: What exists at the target before the import.
    """
    target = tmp_path / "store.zarr"
    if existing == "store":
        _existing_store(target)
    elif existing == "empty":
        target.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    IMPORTERS[importer](target, sources, overwrite_store=True)
    recordings = SigMFZarrStore.open(target).list_recordings()
    assert len(recordings) == 1
    assert "old" not in recordings


def test_overwrite_store_refuses_a_symbolic_link(tmp_path: Path) -> None:
    """A symbolic link at the target is not replaced or followed.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    real = tmp_path / "real.zarr"
    _existing_store(real)
    link = tmp_path / "link.zarr"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        import_sigmf(link, _sigmf_pair(tmp_path), overwrite_store=True)
    assert link.is_symlink()
    assert SigMFZarrStore.open(real).list_recordings() == ("old",)


def test_sigmf_zero_chunks_raise_value_error(tmp_path: Path) -> None:
    """A zero chunk extent is an invalid option, not a division error.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    target = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match="positive integers"):
        import_sigmf(target, _sigmf_pair(tmp_path), sample_chunks=(0,))
    assert not target.exists()


def test_failed_import_keeps_a_target_created_by_another_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup must not remove a target that the import did not create.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    target = tmp_path / "out.zarr"
    validate = validate_sigmf_schema

    def other_writer_then_reject(metadata: Any) -> None:
        """Create the target as another process would, then fail.

        Args:
            metadata: Metadata to validate.

        Raises:
            ValueError: Always.
        """
        validate(metadata)
        target.mkdir()
        (target / "results.csv").write_text("other process output")
        raise ValueError("injected validation failure")

    monkeypatch.setattr(
        sigmf_module, "validate_sigmf_schema", other_writer_then_reject
    )
    with pytest.raises(ValueError, match="injected"):
        import_sigmf(target, _sigmf_pair(tmp_path))
    assert (target / "results.csv").read_text() == "other process output"


def test_failed_import_keeps_a_replaced_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup removes only the directory identity that import created.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    target = tmp_path / "out.zarr"

    def replace_then_fail(*args: Any, **kwargs: Any) -> SigMFZarrStore:
        """Replace the new directory as another process would, then fail.

        Args:
            *args: Store arguments.
            **kwargs: Store keyword arguments.

        Raises:
            OSError: Always.
        """
        del args, kwargs
        # Create the replacement before removing the original, so that the
        # two directories cannot share an inode number.
        aside = tmp_path / "aside"
        target.rename(aside)
        target.mkdir()
        os.rmdir(aside)
        (target / "results.csv").write_text("other process output")
        raise OSError("injected store failure")

    monkeypatch.setattr(
        import_target_module, "SigMFZarrStore", replace_then_fail
    )
    with pytest.raises(OSError, match="injected"):
        import_sigmf(target, _sigmf_pair(tmp_path))
    assert (target / "results.csv").read_text() == "other process output"


def test_failed_import_removes_the_directory_it_created(
    tmp_path: Path,
) -> None:
    """A failure after store creation removes the new target directory.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    target = tmp_path / "out.zarr"
    with pytest.raises(ValueError, match="requires Zarr format 3"):
        import_sigmf(
            target, _sigmf_pair(tmp_path), sample_shards=(16,), zarr_format=2
        )
    assert not target.exists()
