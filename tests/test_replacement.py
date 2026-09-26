"""Resource replacement must preserve existing data when creation fails."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.storage import MemoryStore

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat


@pytest.fixture(
    params=[("local", 2), ("local", 3), ("memory", 2), ("memory", 3)]
)
def store(tmp_path: Path, request: pytest.FixtureRequest) -> SigMFZarrStore:
    """Create a hashed store with samples and collection metadata.

    Args:
        tmp_path: Temporary directory.
        request: Backend and physical-format parameters.

    Returns:
        Writable example store.
    """
    backend, version = request.param
    result = SigMFZarrStore.create(
        tmp_path / "data.zarr" if backend == "local" else MemoryStore(),
        zarr_format=cast(ZarrFormat, version),
    )
    recording = result.recordings.open("rec", sample_shape=(4,))
    recording.set_samples(np.arange(4, dtype=np.float32))
    recording.add_index("quality", [1, 2, 3, 4], axis="time", field="test:q")
    result.collections.open("group", recording_ids=["rec"])
    result.update_integrity()
    return result


@pytest.mark.parametrize(
    "settings",
    [
        {"sample_shape": (0,)},
        {"sample_shape": (4,), "sample_chunks": (1, 1)},
        {"sample_shape": (4,), "global_metadata": {"bad": float("nan")}},
        {"sample_shape": (4,), "create": False},
    ],
)
def test_rejected_recording_replacement_preserves_original(
    store: SigMFZarrStore, settings: dict[str, Any]
) -> None:
    """Rejected recordings must retain samples, indexes, and integrity hashes.

    Args:
        store: Hashed example store.
        settings: Invalid replacement options.
    """
    original = store.recordings["rec"].metadata()
    with pytest.raises((ValueError, KeyError)):
        store.recordings.open("rec", overwrite=True, **settings)
    recording = store.recordings["rec"]
    np.testing.assert_array_equal(recording.samples[:], np.arange(4))
    np.testing.assert_array_equal(recording.index("quality")[:], [1, 2, 3, 4])
    assert recording.metadata() == original
    assert store.verify_integrity()


def test_rejected_collection_replacement_preserves_original(
    store: SigMFZarrStore,
) -> None:
    """Collection validation after creation must also roll back.

    Args:
        store: Hashed example store.
    """
    with pytest.raises(ValueError, match="recording_ids"):
        store.collections.open("group", overwrite=True, recording_ids=[None])
    assert store.collections["group"].recording_ids == ("rec",)
    assert store.verify_integrity()


def test_successful_replacement_publishes_new_resource(
    store: SigMFZarrStore,
) -> None:
    """Successful replacements must publish the layout and invalidate hashes.

    Args:
        store: Hashed example store.
    """
    recording = store.recordings.open(
        "rec", overwrite=True, sample_shape=(6,), sample_dtype=np.int16
    )
    assert recording.samples.shape == (6,)
    assert recording.samples.dtype == np.dtype("int16")
    assert "quality" not in recording.indexes
    assert store.metadata_sha512 is None
    store.collections.open("group", overwrite=True, recording_ids=[])
    assert store.collections["group"].recording_ids == ()
    store.update_integrity()
    assert store.verify_integrity()


def test_replacement_rejects_read_only_backend(store: SigMFZarrStore) -> None:
    """Local directory moves must never bypass backend write protection.

    Args:
        store: Hashed example store.
    """
    readonly = SigMFZarrStore.open(store._group.store, mode="r")
    with pytest.raises(ValueError, match="writable"):
        readonly.recordings.open("rec", overwrite=True, sample_shape=(6,))
    assert store.verify_integrity()


def test_failed_rollback_retains_recovery_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second failure during restoration must retain the original data.

    Args:
        tmp_path: Temporary local store directory.
        monkeypatch: Filesystem failure injection.
    """
    import zarr

    from sigmf_zarr.store import _transaction

    store = SigMFZarrStore.create(tmp_path / "data.zarr")
    recording = store.recordings.open("rec", sample_shape=(4,))
    recording.set_samples(np.arange(4))
    store.update_integrity()

    def fail_restore(*args: object, **kwargs: object) -> None:
        """Simulate an unavailable destination during rollback."""
        raise OSError("restore failed")

    monkeypatch.setattr(_transaction.shutil, "copytree", fail_restore)
    with pytest.raises(OSError, match="Recovery files") as failure:
        store.recordings.open("rec", overwrite=True, sample_shape=(0,))
    backups = list(tmp_path.glob(".sigmf-zarr-backup-*"))
    assert len(backups) == 1
    assert str(backups[0]) in str(failure.value)
    assert (backups[0] / "parent-attributes.json").is_file()
    saved = zarr.open_group(backups[0] / "node", mode="r")
    np.testing.assert_array_equal(saved["samples"][:], np.arange(4))


@pytest.mark.parametrize("scope", ["recording", "store", "extension"])
def test_invalid_array_chunks_preserve_previous_values(
    store: SigMFZarrStore, scope: str
) -> None:
    """Invalid codec geometry must preserve existing arrays and hashes.

    Args:
        store: Hashed example store in either format and backend.
        scope: Array creation interface to exercise.
    """
    recording = store.recordings["rec"]
    store.add_index("nested/quality", [1, 2, 3, 4])
    recording.add_extension_array("nested/quality", [1, 2, 3, 4])
    store.update_integrity()
    with pytest.raises((ValueError, ZeroDivisionError)):
        if scope == "recording":
            recording.add_index(
                "quality", [5, 6, 7, 8], axis="time", field="test:q",
                overwrite=True, chunks=(0,),
            )
        elif scope == "store":
            store.add_index(
                "nested/quality", [5, 6, 7, 8], overwrite=True, chunks=(0,)
            )
        else:
            recording.add_extension_array(
                "nested/quality", [5, 6, 7, 8], overwrite=True, chunks=(0,)
            )
    for array in (
        recording.index("quality"),
        store.index("nested/quality"),
        recording.extensions["nested/quality"],
    ):
        np.testing.assert_array_equal(array[:], [1, 2, 3, 4])
    assert store.verify_integrity()


@pytest.mark.parametrize("scope", ["index", "item_metadata"])
def test_failed_array_write_restores_data_and_descriptors(
    store: SigMFZarrStore, scope: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after writing replacement data must restore the old array.

    Args:
        store: Hashed example store in either format and backend.
        scope: Array replacement path to exercise.
        monkeypatch: Inject a failure after a backend write.
    """
    from zarr.core.array import Array

    recording = store.recordings.open("batch", batched=True, sample_shape=(4,))
    recording.append_samples(np.zeros((2, 4)))
    recording.set_item_metadata([{"global": {"test:label": "old"}}, {}])
    recording.add_index("label", ["a", "b"], axis="item", field="test:label")
    store.update_integrity()
    write = Array.__setitem__

    def fail_after_write(array: Array, key: Any, value: Any) -> None:
        """Complete one array write, then simulate an I/O failure."""
        write(array, key, value)
        raise OSError("write interrupted")

    monkeypatch.setattr(Array, "__setitem__", fail_after_write)
    with pytest.raises(OSError, match="write interrupted"):
        if scope == "index":
            recording.add_index(
                "label", ["new", "new"], axis="item", field="test:new",
                overwrite=True,
            )
        else:
            recording.set_item_metadata([{}, {}], overwrite=True)
    assert recording.index("label")[:].tolist() == ["a", "b"]
    assert recording.index("label").attrs["field"] == "test:label"
    assert recording.get_item_metadata(0) == {"global": {"test:label": "old"}}
    assert store.verify_integrity()


def test_successful_array_replacement_removes_old_descriptors(
    store: SigMFZarrStore,
) -> None:
    """A replacement must publish its new values without stale attributes.

    Args:
        store: Hashed example store.
    """
    recording = store.recordings["rec"]
    recording.add_index(
        "quality", [9, 8, 7, 6], axis="time", field="test:new",
        overwrite=True, unit="dB",
    )
    index = recording.index("quality")
    np.testing.assert_array_equal(index[:], [9, 8, 7, 6])
    assert dict(index.attrs) == {
        "axis": "time", "field": "test:new", "kind": "metadata", "unit": "dB"
    }
    assert store.metadata_sha512 is None
    assert recording.metadata_sha512 is None
    store.update_integrity()
    assert store.verify_integrity()
