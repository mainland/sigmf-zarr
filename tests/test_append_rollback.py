"""Tests for consistent recordings after interrupted resizes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr.store import SigMFRecording, SigMFZarrStore


class Interrupted(BaseException):
    """Injected failure that ordinary exception handlers do not catch."""


def fail_on(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    predicate: Callable[[Array[Any]], bool],
) -> None:
    """Make the next matching call of one Zarr array method raise.

    Later calls succeed, so rollback can use the same method.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        method: `Array` method to replace.
        predicate: Selects the arrays that fail.
    """
    original = getattr(Array, method)
    pending = [True]

    def replacement(self: Array[Any], *args: Any, **kwargs: Any) -> Any:
        """Raise for selected arrays and delegate otherwise.

        Args:
            self: Array whose method is called.
            *args: Positional method arguments.
            **kwargs: Keyword method arguments.

        Returns:
            Result of the original method.

        Raises:
            Interrupted: On the first call for a matching array.
        """
        if pending and predicate(self):
            pending.clear()
            raise Interrupted
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Array, method, replacement)


@pytest.fixture
def batched(tmp_path: Path) -> SigMFZarrStore:
    """Create a hashed store with a batched recording and item metadata.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Store containing recording `batch`.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "batch", batched=True, sample_shape=(4,), sample_chunks=(2, 4)
    )
    recording.append_samples(
        np.ones((3, 4), dtype=np.float32),
        item_metadata=[{"global": {"test:n": n}} for n in range(3)],
    )
    recording.add_index(
        "snr", np.arange(3, dtype=np.int16), axis="item", field="test:snr"
    )
    store.update_integrity()
    return store


def assert_batched_unchanged(store: SigMFZarrStore) -> None:
    """Check that a fresh handle sees the recording before the append.

    Args:
        store: Store whose recording was appended to.
    """
    reopened = SigMFZarrStore.open(store._group.store_path.store)
    recording = reopened.recordings["batch"]
    assert len(recording) == 3
    np.testing.assert_array_equal(recording.samples[:], np.ones((3, 4)))
    assert recording.get_item_metadata(2) == {"global": {"test:n": 2}}
    np.testing.assert_array_equal(recording.index("snr")[:], [0, 1, 2])
    assert reopened.verify_integrity()


def test_interrupted_item_metadata_resize_restores_append(
    batched: SigMFZarrStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item metadata failure after the sample resize restores both lengths.

    Args:
        batched: Hashed store with a batched recording.
        monkeypatch: Pytest monkeypatch fixture.
    """
    fail_on(
        monkeypatch, "resize", lambda array: array.basename == "item_metadata"
    )
    with pytest.raises(Interrupted):
        batched.recordings["batch"].append_samples(
            np.zeros((2, 4), dtype=np.float32)
        )
    monkeypatch.undo()
    assert_batched_unchanged(batched)


def test_interrupted_item_metadata_creation_restores_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while creating item metadata removes the partial array.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "batch", batched=True, sample_shape=(4,)
    )
    recording.append_samples(np.ones((3, 4), dtype=np.float32))
    store.update_integrity()

    def interrupt(*args: object, **kwargs: object) -> None:
        """Interrupt item metadata creation.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            Interrupted: Always.
        """
        raise Interrupted

    monkeypatch.setattr(
        SigMFRecording, "_create_item_metadata_array", interrupt
    )
    with pytest.raises(Interrupted):
        recording.append_samples(
            np.zeros((1, 4), dtype=np.float32),
            item_metadata=[{"global": {"test:n": 3}}],
        )
    monkeypatch.undo()
    reopened = store.recordings["batch"]
    assert len(reopened) == 3
    assert not reopened.has_item_metadata
    assert store.verify_integrity()


def test_interrupted_unbatched_append_restores_shape_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after the time axis grows restores metadata and hashes.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec",
        sample_shape=(4,),
        global_metadata={"core:datatype": "rf32_le"},
    )
    recording.set_samples(np.arange(4, dtype=np.float32))
    # Rollback restores attributes, so any declared digest must survive.
    recording.set_global_field("core:sha512", "a" * 128)
    store.update_integrity()

    def interrupt(*args: object, **kwargs: object) -> None:
        """Interrupt the final capture update.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            Interrupted: Always.
        """
        raise Interrupted

    monkeypatch.setattr(SigMFRecording, "_append_capture", interrupt)
    with pytest.raises(Interrupted):
        recording.append_samples(
            np.arange(4, 6, dtype=np.float32),
            capture={"core:frequency": 1.0},
        )
    monkeypatch.undo()
    reopened = SigMFZarrStore.open(tmp_path / "store.zarr")["rec"]
    assert isinstance(reopened, SigMFRecording)
    assert reopened.sample_shape == (4,)
    np.testing.assert_array_equal(reopened.samples[:], np.arange(4))
    assert reopened.sha512 == "a" * 128
    assert SigMFZarrStore.open(tmp_path / "store.zarr").verify_integrity()


def test_interrupted_set_samples_leaves_consistent_unhashed_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replaced values keep lengths consistent but drop the sample hashes.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec",
        sample_shape=(4,),
        global_metadata={"core:datatype": "rf32_le"},
    )
    recording.set_samples(np.arange(4, dtype=np.float32))
    recording.add_index(
        "level", np.arange(4, dtype=np.int16), axis="time", field="test:level"
    )
    recording.set_global_field("core:sha512", "a" * 128)
    store.update_integrity()
    fail_on(
        monkeypatch, "__setitem__", lambda array: array.basename == "samples"
    )
    with pytest.raises(Interrupted):
        recording.set_samples(np.zeros(6, dtype=np.float32))
    monkeypatch.undo()
    reopened = SigMFZarrStore.open(tmp_path / "store.zarr")
    restored = reopened.recordings["rec"]
    assert restored.sample_shape == (4,)
    np.testing.assert_array_equal(restored.index("level")[:], np.arange(4))
    assert restored.sha512 is None
    assert restored.sample_sha512 is None
    assert restored.metadata_sha512 is None
    assert reopened.metadata_sha512 is None


def test_mutate_samples_restores_resized_array(tmp_path: Path) -> None:
    """Resizing inside a sample mutation is rejected and undone.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create("rec", sample_shape=(4,))
    with (
        pytest.raises(ValueError, match="cannot resize"),
        recording.mutate_samples() as samples,
    ):
        samples.resize((8,))
    reopened = SigMFZarrStore.open(tmp_path / "store.zarr")["rec"]
    assert isinstance(reopened, SigMFRecording)
    assert reopened.samples.shape == (4,)
