"""Tests for export source selection and target file names."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore, export_sigmf


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
