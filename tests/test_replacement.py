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

    from sigmf_zarr.store import _transaction

    store = SigMFZarrStore.create(tmp_path / "data.zarr")
    recording = store.recordings.create("rec", sample_shape=(4,))
    recording.set_samples(np.arange(4))
    store.update_integrity()

    def fail_restore(*args: object, **kwargs: object) -> None:
        """Simulate an unavailable destination during rollback."""
        raise OSError("restore failed")

    monkeypatch.setattr(
        f"{_transaction.__name__}.shutil.copytree", fail_restore
    )
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
