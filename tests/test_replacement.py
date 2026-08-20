"""Resource replacement must preserve existing data when creation fails."""

import errno
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.storage import MemoryStore

from sigmf_zarr.readonly import ReadOnlyArray
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
    recording = result.recordings.create("rec", sample_shape=(4,))
    recording.set_samples(np.arange(4, dtype=np.float32))
    recording.add_index("quality", [1, 2, 3, 4], axis="time", field="test:q")
    result.collections.create("group", recording_ids=["rec"])
    result.update_integrity()
    return result


@pytest.mark.parametrize(
    "settings",
    [
        {"sample_shape": (0,)},
        {"sample_shape": (4,), "sample_chunks": (1, 1)},
        {"sample_shape": (4,), "global_metadata": {"bad": float("nan")}},
        {"sample_shape": (4,), "sample_axes": ("item",)},
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
        store.recordings.create("rec", overwrite=True, **settings)
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
    with pytest.raises(ValueError, match="recording IDs"):
        store.collections.create(
            "group",
            overwrite=True,
            recording_ids=[None],  # type: ignore[list-item]
        )
    assert store.collections["group"].recording_ids == ("rec",)
    assert store.verify_integrity()


def test_successful_replacement_publishes_new_resource(
    store: SigMFZarrStore,
) -> None:
    """Successful replacements must publish the layout and invalidate hashes.

    Args:
        store: Hashed example store.
    """
    recording = store.recordings.create(
        "rec", overwrite=True, sample_shape=(6,), sample_dtype=np.int16
    )
    assert recording.samples.shape == (6,)
    assert recording.samples.dtype == np.dtype("int16")
    assert "quality" not in recording.indexes
    assert store.metadata_sha512 is None
    store.collections.create("group", overwrite=True, recording_ids=[])
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
        readonly.recordings.create("rec", overwrite=True, sample_shape=(6,))
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

    store = SigMFZarrStore.create(tmp_path / "data.zarr")
    recording = store.recordings.create("rec", sample_shape=(4,))
    recording.set_samples(np.arange(4))
    store.update_integrity()

    original_rename = Path.rename

    def fail_restore(self: Path, target: str | os.PathLike[str]) -> Path:
        """Simulate an unavailable destination when restoring the backup.

        Args:
            self: Path being renamed.
            target: Destination path.

        Returns:
            Renamed path for every rename except the backup restoration.

        Raises:
            OSError: When the saved node would move back into the store.
        """
        if self.name == "node" and self.parent.name.startswith(
            ".sigmf-zarr-backup-"
        ):
            raise OSError("restore failed")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_restore)
    with pytest.raises(OSError, match="Recovery files") as failure:
        store.recordings.create("rec", overwrite=True, sample_shape=(0,))
    backups = list(tmp_path.glob(".sigmf-zarr-backup-*"))
    assert len(backups) == 1
    assert str(backups[0]) in str(failure.value)
    assert (backups[0] / "parent-attributes.json").is_file()
    saved = zarr.open_group(backups[0] / "node", mode="r")
    saved_samples = saved["samples"]
    assert isinstance(saved_samples, zarr.Array)
    np.testing.assert_array_equal(saved_samples[:], np.arange(4))


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

    def replace() -> None:
        """Replace the selected array with invalid chunk geometry."""
        if scope == "recording":
            recording.add_index(
                "quality",
                [5, 6, 7, 8],
                axis="time",
                field="test:q",
                overwrite=True,
                chunks=(0,),
            )
        elif scope == "store":
            store.add_index(
                "nested/quality", [5, 6, 7, 8], overwrite=True, chunks=(0,)
            )
        else:
            recording.add_extension_array(
                "nested/quality", [5, 6, 7, 8], overwrite=True, chunks=(0,)
            )

    with pytest.raises((ValueError, ZeroDivisionError)):
        replace()
    extension = recording.extensions["nested/quality"]
    assert isinstance(extension, ReadOnlyArray)
    for array in (
        recording.index("quality"),
        store.index("nested/quality"),
        extension,
    ):
        np.testing.assert_array_equal(array[:], [1, 2, 3, 4])
    assert store.verify_integrity()


@pytest.mark.parametrize("scope", ["recording", "store", "extension"])
@pytest.mark.parametrize("existing_parent", [False, True])
def test_failed_nested_creation_restores_groups_and_integrity(
    store: SigMFZarrStore, scope: str, existing_parent: bool
) -> None:
    """Rollback removes new ancestors and retains existing sibling arrays.

    Args:
        store: Hashed example store in either format and backend.
        scope: Array creation interface to exercise.
        existing_parent: Whether a sibling already occupies the parent group.
    """
    recording = store.recordings["rec"]
    if scope == "store":
        group = store.indexes
    elif scope == "recording":
        group = recording.indexes
    else:
        group = recording.extensions
    if existing_parent:
        if scope == "store":
            store.add_index("nested/keep", [1, 2, 3, 4])
        elif scope == "recording":
            recording.add_index(
                "nested/keep", [1, 2, 3, 4], axis="time", field="test:keep"
            )
        else:
            recording.add_extension_array("nested/keep", [1, 2, 3, 4])
    store.update_integrity()
    root_hash = store.metadata_sha512
    recording_hash = recording.metadata_sha512

    def create() -> None:
        """Create a nested array with invalid chunk geometry."""
        if scope == "store":
            store.add_index("nested/new/quality", [1, 2, 3, 4], chunks=(0,))
        elif scope == "recording":
            recording.add_index(
                "nested/new/quality",
                [1, 2, 3, 4],
                axis="time",
                field="test:quality",
                chunks=(0,),
            )
        else:
            recording.add_extension_array(
                "nested/new/quality", [1, 2, 3, 4], chunks=(0,)
            )

    with pytest.raises((ValueError, ZeroDivisionError)):
        create()
    assert ("nested" in group) == existing_parent
    assert "nested/new" not in group
    if existing_parent:
        keep = group["nested/keep"]
        assert isinstance(keep, ReadOnlyArray)
        np.testing.assert_array_equal(keep[:], [1, 2, 3, 4])
    assert store.metadata_sha512 == root_hash
    assert recording.metadata_sha512 == recording_hash
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

    recording = store.recordings.create(
        "batch", batched=True, sample_shape=(4,)
    )
    recording.append_samples(np.zeros((2, 4)))
    recording.set_item_metadata([{"global": {"test:label": "old"}}, {}])
    recording.add_index("label", ["a", "b"], axis="item", field="test:label")
    store.update_integrity()
    write = Array.__setitem__

    def fail_after_write(array: Array[Any], key: Any, value: Any) -> None:
        """Complete one array write, then simulate an I/O failure."""
        write(array, key, value)
        raise OSError("write interrupted")

    monkeypatch.setattr(Array, "__setitem__", fail_after_write)

    def replace() -> None:
        """Replace the selected array through the interrupted write."""
        if scope == "index":
            recording.add_index(
                "label",
                ["new", "new"],
                axis="item",
                field="test:new",
                overwrite=True,
            )
        else:
            recording.set_item_metadata([{}, {}], overwrite=True)

    with pytest.raises(OSError, match="write interrupted"):
        replace()
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
        "quality",
        [9, 8, 7, 6],
        axis="time",
        field="test:new",
        overwrite=True,
        unit="dB",
    )
    index = recording.index("quality")
    np.testing.assert_array_equal(index[:], [9, 8, 7, 6])
    assert dict(index.attrs) == {
        "axis": "time",
        "field": "test:new",
        "kind": "metadata",
        "unit": "dB",
    }
    assert store.metadata_sha512 is None
    assert recording.metadata_sha512 is None
    store.update_integrity()
    assert store.verify_integrity()


@pytest.mark.parametrize(
    "settings",
    [
        {"sample_shape": (4,), "global_metadata": {"bad": float("nan")}},
        {"sample_shape": (4,), "captures": [{"bad": float("inf")}]},
        {"sample_shape": (4,), "sample_chunks": (1, 1)},
    ],
)
def test_rejected_recording_creation_leaves_name_available(
    store: SigMFZarrStore, settings: dict[str, Any]
) -> None:
    """A failed creation must not leave a partial recording or stale hashes.

    Args:
        store: Hashed example store.
        settings: Invalid creation options.
    """
    with pytest.raises(ValueError, match=r"to be finite|shape"):
        store.recordings.create("new", **settings)
    assert "new" not in store.recordings
    assert store.verify_integrity()
    created = store.recordings.create("new", sample_shape=(4,))
    assert created.sample_shape == (4,)


@pytest.mark.parametrize(
    "settings",
    [
        {"metadata": {"bad": float("nan")}},
        {"recording_ids": ["missing"]},
        {"recording_ids": [""]},
        {"recording_ids": "rec"},
    ],
)
def test_rejected_collection_creation_leaves_name_available(
    store: SigMFZarrStore, settings: dict[str, Any]
) -> None:
    """Creation must apply the managed setter rules before writing.

    Args:
        store: Hashed example store.
        settings: Invalid creation options.
    """
    with pytest.raises(ValueError, match=r"Collection|to be finite"):
        store.collections.create("new", **settings)
    assert "new" not in store.collections
    assert store.verify_integrity()


def test_referenced_recording_cannot_be_removed(
    store: SigMFZarrStore,
) -> None:
    """Removal must not leave a collection with a dangling reference.

    Args:
        store: Hashed example store.
    """
    with pytest.raises(ValueError, match="referenced by collections"):
        store.recordings.remove("rec")
    assert "rec" in store.recordings
    assert store.verify_integrity()
    store.collections["group"].set_recording_ids([])
    store.recordings.remove("rec")
    assert "rec" not in store.recordings


def _unreferenced(store: SigMFZarrStore) -> SigMFZarrStore:
    """Drop the collection reference to `rec` and rehash the store.

    Args:
        store: Hashed example store.

    Returns:
        The same store with `rec` removable and every hash current.
    """
    store.collections["group"].set_recording_ids([])
    store.update_integrity()
    return store


def test_failed_removal_hash_write_restores_recording(
    store: SigMFZarrStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed root metadata write must leave the removed recording intact.

    Args:
        store: Hashed example store in either format and backend.
        monkeypatch: Failure injection for the store hash write.
    """
    _unreferenced(store)

    def fail(self: SigMFZarrStore) -> None:
        """Simulate a failed root attribute write.

        Args:
            self: Store whose hash would be invalidated.

        Raises:
            OSError: Always.
        """
        raise OSError("root write failed")

    monkeypatch.setattr(SigMFZarrStore, "_invalidate_metadata_integrity", fail)
    with pytest.raises(OSError, match="root write failed"):
        store.recordings.remove("rec")
    monkeypatch.undo()
    recording = store.recordings["rec"]
    np.testing.assert_array_equal(recording.samples[:], np.arange(4))
    np.testing.assert_array_equal(recording.index("quality")[:], [1, 2, 3, 4])
    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_failed_nonlocal_removal_is_reported_and_retryable(
    zarr_format: ZarrFormat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial delete on a nonlocal backend must be detectable.

    Nonlocal removal deletes without a backup, so a failure can leave part
    of the recording. The recording group's own metadata goes last, so the
    recording stays listed. Validation must report it, and a second removal
    must finish the deletion.

    Args:
        zarr_format: Physical Zarr format to create.
        monkeypatch: Failure injection for the backend delete.
    """
    from sigmf_zarr.validation import validate_store

    store = SigMFZarrStore.create(MemoryStore(), zarr_format=zarr_format)
    recording = store.recordings.create("rec", sample_shape=(4,))
    recording.set_samples(np.arange(4, dtype=np.float32))
    store.update_integrity()
    original_delete = MemoryStore.delete
    group_document = "recordings/rec/" + (
        ".zgroup" if zarr_format == 2 else "zarr.json"
    )

    async def partial_delete(self: MemoryStore, key: str) -> None:
        """Delete every object except the recording group's document.

        Args:
            self: Backend being modified.
            key: Object key to delete.

        Raises:
            OSError: If `key` is the recording group's document.
        """
        if key == group_document:
            raise OSError("delete failed")
        await original_delete(self, key)

    monkeypatch.setattr(MemoryStore, "delete", partial_delete)
    with pytest.raises(OSError, match="delete failed"):
        store.recordings.remove("rec")
    monkeypatch.undo()
    assert store.metadata_sha512 is None
    assert "rec" in store.recordings
    report = validate_store(store)
    assert report.valid is False
    assert any(issue.path == "recordings/rec" for issue in report.issues)
    store.recordings.remove("rec")
    assert "rec" not in store.recordings
    assert validate_store(store).valid is True


def test_successful_removal_leaves_no_backup(tmp_path: Path) -> None:
    """A local removal must delete its backup after the transaction.

    Args:
        tmp_path: Temporary local store directory.
    """
    store = SigMFZarrStore.create(tmp_path / "data.zarr")
    store.recordings.create("rec", sample_shape=(4,))
    store.update_integrity()
    store.recordings.remove("rec")
    assert "rec" not in store.recordings
    assert store.metadata_sha512 is None
    assert list(tmp_path.iterdir()) == [tmp_path / "data.zarr"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_replacement_rollback_renames_without_free_space(
    tmp_path: Path, zarr_format: ZarrFormat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback after ENOSPC must move the backup back instead of copying.

    Args:
        tmp_path: Temporary local store directory.
        zarr_format: Physical Zarr format to create.
        monkeypatch: Failure injection for the full disk.
    """
    from sigmf_zarr.store import _transaction

    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=zarr_format
    )
    recording = store.recordings.create("rec", sample_shape=(4,))
    recording.add_extension_array("table", np.arange(6).reshape(2, 3))
    store.update_integrity()

    def no_space(*args: object, **kwargs: object) -> None:
        """Simulate a full disk for every write after the backup.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            OSError: Always, with `ENOSPC`.
        """
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(SigMFZarrStore, "create_array", no_space)
    monkeypatch.setattr(f"{_transaction.__name__}.shutil.copytree", no_space)
    with pytest.raises(OSError, match=os.strerror(errno.ENOSPC)) as failure:
        recording.add_extension_array("table", np.zeros(3), overwrite=True)
    monkeypatch.undo()
    assert failure.value.errno == errno.ENOSPC
    table = recording.extensions["table"]
    assert isinstance(table, ReadOnlyArray)
    np.testing.assert_array_equal(table[:], np.arange(6).reshape(2, 3))
    assert store.verify_integrity()
    assert list(tmp_path.iterdir()) == [tmp_path / "data.zarr"]
