"""Tests for export source selection and target file names."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore, export_sigmf, export_sigmf_archive


@pytest.fixture
def store(tmp_path: Path) -> SigMFZarrStore:
    """Create a writable, hashed store with three exportable recordings.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Store containing recordings `a`, `b`, and `c.v2`.
    """
    result = SigMFZarrStore.create(tmp_path / "store.zarr")
    for name in ("a", "b", "c.v2"):
        recording = result.recordings.create(
            name,
            sample_shape=(4,),
            global_metadata={"core:datatype": "rf32_le"},
            captures=[{"core:sample_start": 0}],
        )
        recording.set_samples(np.arange(4, dtype=np.float32))
    result.update_integrity()
    return result


def test_export_of_missing_recording_raises_key_error(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """Exporting a missing recording must not try to create it.

    Args:
        store: Writable source store.
        tmp_path: Pytest temporary path fixture.
    """
    with pytest.raises(KeyError):
        export_sigmf(store, "missing", tmp_path / "out")
    assert "missing" not in store.recordings
    assert store.verify_integrity()


def test_export_retains_dots_in_target_stems(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """A target stem keeps its dots, and SigMF suffixes are normalized.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    meta = export_sigmf(store, "c.v2", tmp_path / "out" / "c.v2")
    assert meta == tmp_path / "out" / "c.v2.sigmf-meta"
    assert (tmp_path / "out" / "c.v2.sigmf-data").is_file()
    data_target = export_sigmf(store, "a", tmp_path / "out" / "a.sigmf-data")
    assert data_target == tmp_path / "out" / "a.sigmf-meta"


def test_archive_export_of_missing_collection_raises_key_error(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """Archive export must not create a missing collection in its source.

    Args:
        store: Writable source store.
        tmp_path: Pytest temporary path fixture.
    """
    with pytest.raises(KeyError):
        export_sigmf_archive(
            store, tmp_path / "bundle.sigmf", collection_name="typo"
        )
    assert store.list_collections() == ()
    assert store.verify_integrity()
    assert not (tmp_path / "bundle.sigmf").exists()


def test_collection_archive_contains_only_its_recordings(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """A collection export contains each member once and no other recording.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    store.collections.create("pair", recording_ids=["b", "a", "b"])
    archive = export_sigmf_archive(
        store, tmp_path / "pair", collection_name="pair"
    )
    assert archive == tmp_path / "pair.sigmf"
    with tarfile.open(archive) as members:
        names = sorted(Path(name).name for name in members.getnames())
    assert names == [
        "a.sigmf-data",
        "a.sigmf-meta",
        "b.sigmf-data",
        "b.sigmf-meta",
        "pair",
        "pair.sigmf-collection",
    ]


def test_archive_retains_dots_in_target_stem(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """An archive target stem keeps its dots.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    archive = export_sigmf_archive(
        store, tmp_path / "bundle.2026-10-06", recording_names=["a"]
    )
    assert archive == tmp_path / "bundle.2026-10-06.sigmf"
    assert archive.is_file()
