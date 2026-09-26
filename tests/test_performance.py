"""Deterministic performance regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr.store import SigMFRecording, SigMFZarrStore


class CountingArray:
    """Record logical reads and writes delegated to a Zarr array."""

    _array: Array
    """Backing Zarr array."""

    reads: list[object]
    """Selections read through this wrapper."""

    writes: list[object]
    """Selections written through this wrapper."""

    def __init__(self, array: Array) -> None:
        """Initialize an access-counting array wrapper.

        Args:
            array: Backing Zarr array.
        """
        self._array = array
        self.reads = []
        self.writes = []

    def __getattr__(self, name: str) -> Any:
        """Delegate array metadata and non-indexing operations.

        Args:
            name: Backing array member name.

        Returns:
            Backing array member.
        """
        return getattr(self._array, name)

    def __getitem__(self, selection: object) -> Any:
        """Record and execute one array read.

        Args:
            selection: Zarr selection expression.

        Returns:
            Selected values.
        """
        self.reads.append(selection)
        return self._array[selection]  # type: ignore[index]

    def __setitem__(self, selection: object, value: object) -> None:
        """Record and execute one array write.

        Args:
            selection: Zarr selection expression.
            value: Values to write.
        """
        self.writes.append(selection)
        self._array[selection] = value  # type: ignore[index]


def _item_metadata_recording(
    path: Path,
    *,
    item_count: int = 8193,
) -> SigMFRecording:
    """Create a recording spanning three item-metadata chunks.

    Args:
        path: Store path.
        item_count: Number of batch items to create.

    Returns:
        Recording with an initialized item metadata array.
    """
    store = SigMFZarrStore.create(path, overwrite=True)
    recording = store.recordings.open(
        "batch",
        create=True,
        batched=True,
        sample_shape=(2, 1),
        sample_axes=("iq", "time"),
        sample_chunks=(1024, 2, 1),
        sample_compressor=None,
        sample_checksum=None,
    )
    recording.append_samples(
        np.zeros((item_count, 2, 1), dtype=np.float32),
        item_metadata=[None] * item_count,
    )
    return recording


def _count_item_metadata_accesses(
    recording: SigMFRecording,
    monkeypatch: pytest.MonkeyPatch,
) -> CountingArray:
    """Replace the private item array property with a counting wrapper.

    Args:
        recording: Recording whose item array should be observed.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Installed counting wrapper.
    """
    counting = CountingArray(recording._item_metadata_array)

    def item_metadata_array(unused_recording: SigMFRecording) -> Array:
        """Return the installed counting wrapper.

        Args:
            unused_recording: Recording accessing its private item array.

        Returns:
            Counting wrapper presented as a Zarr array.
        """
        del unused_recording
        return cast(Array, counting)

    monkeypatch.setattr(
        SigMFRecording,
        "_item_metadata_array",
        property(item_metadata_array),
    )
    return counting


def test_item_metadata_validation_reads_each_chunk_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation I/O should scale with chunks rather than item count.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    recording = _item_metadata_recording(tmp_path / "store.zarr")
    counting = _count_item_metadata_accesses(recording, monkeypatch)

    recording._validate_item_metadata_array()

    # Count logical I/O operations instead of wall time so this regression
    # remains deterministic across machines and storage backends.
    assert counting.reads == [
        slice(0, 4096),
        slice(4096, 8192),
        slice(8192, 8193),
    ]
    assert counting.writes == []


def test_item_metadata_entry_setter_does_not_scan_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated entry update should issue one write and no reads.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    recording = _item_metadata_recording(tmp_path / "store.zarr")
    counting = _count_item_metadata_accesses(recording, monkeypatch)

    recording.set_item_metadata_entry(
        5000,
        {"global": {"example:value": 1}},
    )

    assert counting.reads == []
    assert counting.writes == [5000]


def test_item_metadata_slice_setter_uses_one_contiguous_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated slice update should not degrade to scalar writes.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    recording = _item_metadata_recording(tmp_path / "store.zarr")
    counting = _count_item_metadata_accesses(recording, monkeypatch)

    recording.set_item_metadata_slice(
        slice(4090, 4100),
        [{"global": {"example:value": index}} for index in range(10)],
    )

    assert counting.reads == []
    assert counting.writes == [slice(4090, 4100)]
