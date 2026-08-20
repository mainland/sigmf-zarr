"""Tests for reusable RadioML helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

import sigmf_zarr.radioml2016 as radioml2016
import sigmf_zarr.radioml2018 as radioml2018
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore
from sigmf_zarr.store._common import ZarrFormat


def test_flatten_radioml_dataset_returns_aligned_arrays() -> None:
    """Flattening should preserve sorted key order and aligned labels."""
    dataset = {
        ("BPSK", 0): np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        ("AM-DSB", 10): np.array(
            [
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            dtype=np.float32,
        ),
    }

    iq, mod_class_id, snr_db, mod_classes = (
        radioml2016.flatten_radioml2016_dataset(dataset)
    )

    assert iq.shape == (3, 2, 2)
    np.testing.assert_array_equal(
        mod_class_id,
        np.array([0, 0, 1], dtype=np.int16),
    )
    assert mod_classes == ["AM-DSB", "BPSK"]
    np.testing.assert_array_equal(
        snr_db,
        np.array([10, 10, 0], dtype=np.int16),
    )


def test_import_radioml_dataset_delegates_to_write(monkeypatch) -> None:
    """Import helper should write RadioML data into a plain store.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    captured: dict[str, object] = {}

    class FakeArray:
        def __init__(self, name: str) -> None:
            """Initialize a fake array.

            Args:
                name: Array name.
            """
            self.name = name
            self.value = None

        def __setitem__(self, key: object, value: object) -> None:
            """Store assigned array values.

            Args:
                key: Assignment key.
                value: Assigned value.
            """
            del key
            self.value = np.asarray(value)

    class FakeExtension:
        def __init__(self) -> None:
            """Initialize a fake extension group."""
            self.groups: dict[str, object] = {}
            self.arrays: dict[str, FakeArray] = {}

        def __contains__(self, name: object) -> bool:
            """Return whether the fake group contains a member.

            Args:
                name: Candidate member name.

            Returns:
                True when `name` exists as a group or array.
            """
            return isinstance(name, str) and (
                name in self.groups or name in self.arrays
            )

        def __delitem__(self, name: str) -> None:
            """Delete a fake member.

            Args:
                name: Member name to delete.
            """
            self.groups.pop(name, None)
            self.arrays.pop(name, None)

        def create_group(self, name: str):
            """Create a fake subgroup.

            Args:
                name: Subgroup name.

            Returns:
                Fake subgroup.
            """
            group = FakeExtension()
            self.groups[name] = group
            return group

        def create_array(self, name: str, **kwargs: object) -> FakeArray:
            """Create a fake array.

            Args:
                name: Array name.
                **kwargs: Array creation options.

            Returns:
                Created fake array.
            """
            captured.setdefault("arrays", {})[name] = kwargs
            array = FakeArray(name)
            if "data" in kwargs:
                array.value = np.asarray(kwargs["data"])
            self.arrays[name] = array
            return array

    class FakeRecording:
        def __init__(self) -> None:
            """Initialize a fake recording."""
            self.group = type("Group", (), {"attrs": {}})()
            self.extensions = FakeExtension()

        def append_samples(self, samples: object) -> None:
            """Capture appended samples.

            Args:
                samples: Samples to append.
            """
            captured["iq"] = np.asarray(samples)

        def update_integrity(self) -> dict[str, object]:
            """Record an internal integrity update.

            Returns:
                Empty fake integrity object.
            """
            captured["recording_integrity_updated"] = True
            return {}

        def add_index(
            self,
            index_name: str,
            values: object,
            *,
            axis: str,
            field: str,
            kind: str = "metadata",
            unit: str | None = None,
            labels: list[object] | None = None,
            overwrite: bool = False,
        ) -> FakeArray:
            """Capture recording-level index creation.

            Args:
                index_name: Index array name.
                values: Index values.
                axis: Runtime sample axis indexed by the values.
                field: Metadata field represented by the index.
                kind: Index kind.
                unit: Optional value unit.
                labels: Optional index labels.
                overwrite: Whether replacement is allowed.

            Returns:
                Created fake index array.
            """
            del overwrite
            captured.setdefault("indexes", {})[index_name] = {
                "values": np.asarray(values),
                "axis": axis,
                "field": field,
                "kind": kind,
                "unit": unit,
                "labels": labels,
            }
            array = FakeArray(index_name)
            array.value = np.asarray(values)
            return array

        def create_array(
            self,
            group: FakeExtension,
            name: str,
            data: object,
            *,
            overwrite: bool,
        ) -> FakeArray:
            """Create a fake nested array.

            Args:
                group: Parent fake extension group.
                name: Array name or nested path.
                data: Array data.
                overwrite: Whether replacement is allowed.

            Returns:
                Created fake array.
            """
            del overwrite
            if "/" not in name:
                return group.create_array(name, data=data)

            group_name, array_name = name.rsplit("/", 1)
            target_group = group.groups.get(group_name)
            if target_group is None:
                target_group = group.create_group(group_name)
            return target_group.create_array(array_name, data=data)

    class FakeRecordings:
        def __init__(self) -> None:
            """Initialize a fake recordings view."""
            self.recording = FakeRecording()

        def __contains__(self, recording_name: object) -> bool:
            """Report that no recording exists before conversion.

            Args:
                recording_name: Candidate recording name.

            Returns:
                Always false.
            """
            del recording_name
            return False

        def open(self, recording_name: str, **kwargs: object) -> FakeRecording:
            """Open the fake recording.

            Args:
                recording_name: Recording name.
                **kwargs: Recording open options.

            Returns:
                Fake recording.
            """
            captured["recording_name"] = recording_name
            captured["recording_kwargs"] = kwargs
            self.recording.group.attrs["global"] = kwargs["global_metadata"]
            return self.recording

    class FakeStore:
        def __init__(self) -> None:
            """Initialize a fake store."""
            self.recordings = FakeRecordings()

        def update_metadata_integrity(self) -> dict[str, object]:
            """Record a store metadata integrity update.

            Returns:
                Empty fake integrity object.
            """
            captured["store_integrity_updated"] = True
            return {}

        def info(self) -> str:
            """Return fake store info.

            Returns:
                Fake store info string.
            """
            return "fake-store"

    def fake_create(
        store_path: str | Path,
        *,
        overwrite: bool = False,
        zarr_format: int = 3,
    ):
        """Create a fake store.

        Args:
            store_path: Target store path.
            overwrite: Whether to recreate the store.
            zarr_format: Physical Zarr format.

        Returns:
            Fake store.
        """
        captured["store_path"] = store_path
        captured["overwrite"] = overwrite
        captured["zarr_format"] = zarr_format
        return FakeStore()

    monkeypatch.setattr(radioml2016.SigMFZarrStore, "create", fake_create)

    dataset = {
        ("BPSK", 0): np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            dtype=np.float32,
        ),
    }
    result = radioml2016.import_radioml2016_dataset(
        "store.zarr",
        dataset,
        source_dataset="RML2016.10a",
        recording_name="demo",
        overwrite_store=True,
        overwrite_recording=True,
        sample_shards=(32, 2, 3),
        zarr_format=3,
    )

    assert result.info() == "fake-store"
    assert captured["store_path"] == "store.zarr"
    np.testing.assert_array_equal(
        captured["iq"],
        np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            dtype=np.float32,
        ),
    )
    assert captured["recording_name"] == "demo"
    assert captured["overwrite"] is True
    assert captured["zarr_format"] == 3
    assert captured["recording_kwargs"]["batched"] is True
    assert captured["recording_kwargs"]["sample_shape"] == (2, 3)
    assert captured["recording_kwargs"]["sample_axes"] == ("iq", "time")
    assert captured["recording_kwargs"]["sample_shards"] == (32, 2, 3)
    recording = result.recordings.recording
    assert recording.group.attrs["global"]["radioml:source_dataset"] == (
        "RML2016.10a"
    )
    indexes = captured["indexes"]
    assert indexes["snr_db"]["axis"] == "item"
    assert indexes["snr_db"]["field"] == "radioml:snr"
    assert indexes["snr_db"]["kind"] == "metadata"
    assert indexes["snr_db"]["unit"] == "dB"
    np.testing.assert_array_equal(
        indexes["snr_db"]["values"],
        np.array([0], dtype=np.int16),
    )
    assert indexes["mod_class_id"]["axis"] == "item"
    assert indexes["mod_class_id"]["field"] == "radioml:mod_class"
    assert indexes["mod_class_id"]["kind"] == "metadata"
    assert indexes["mod_class_id"]["labels"] == ["BPSK"]
    np.testing.assert_array_equal(
        indexes["mod_class_id"]["values"],
        np.array([0], dtype=np.int16),
    )


def test_import_radioml2018_dataset_streams_hdf5_batches(tmp_path) -> None:
    """The 2018 importer should transpose and append bounded HDF5 batches.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "RML2018.hdf5"
    store_path = tmp_path / "RML2018.sigmf-zarr"
    source_samples = np.arange(40, dtype=np.float32).reshape(5, 4, 2)
    source_labels = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source_snr = np.array([[-2], [0], [2], [4], [6]], dtype=np.int16)
    with h5py.File(source_path, "w") as source:
        source.create_dataset("X", data=source_samples)
        source.create_dataset("Y", data=source_labels)
        source.create_dataset("Z", data=source_snr)

    store = radioml2018.import_radioml2018_dataset(
        store_path,
        source_path,
        modulation_classes=("BPSK", "QPSK"),
        batch_size=2,
        zarr_format=2,
    )

    recording = store.recordings["radioml2018"]
    assert store.zarr_format == 2
    assert recording.samples.shape == (5, 2, 4)
    np.testing.assert_array_equal(
        recording.samples[:],
        np.moveaxis(source_samples, 2, 1),
    )
    np.testing.assert_array_equal(
        recording.index("mod_class_id")[:],
        np.array([0, 1, 0, 1, 1], dtype=np.int16),
    )
    assert recording.index("mod_class_id").attrs["labels"] == [
        "BPSK",
        "QPSK",
    ]
    np.testing.assert_array_equal(
        recording.index("snr_db")[:],
        source_snr[:, 0],
    )
    assert recording.global_metadata["radioml:dataset_version"] == "2018"
    assert recording.global_metadata["radioml:source_dataset"] == "RML2018"


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_radioml2016_streams_sorted_bounded_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zarr_format: ZarrFormat,
) -> None:
    """Keep conversion writes bounded while preserving sample-label order.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
        zarr_format: Physical Zarr format to exercise.
    """
    dataset = {
        ("QPSK", -2): np.arange(20, dtype=np.float64).reshape(5, 2, 2),
        ("BPSK", 10): np.arange(12, dtype=np.float64).reshape(3, 2, 2),
    }
    batches: list[int] = []
    original_append = SigMFRecording.append_samples

    def append(recording: SigMFRecording, samples: npt.ArrayLike) -> None:
        """Record the batch size before performing the real append.

        Args:
            recording: Destination recording.
            samples: Batch passed by the importer.
        """
        batches.append(len(np.asarray(samples)))
        original_append(recording, samples)

    monkeypatch.setattr(SigMFRecording, "append_samples", append)
    store = radioml2016.import_radioml2016_dataset(
        tmp_path / "store.zarr",
        dataset,
        batch_size=2,
        zarr_format=zarr_format,
    )
    recording = store.recordings["radioml2016"]
    assert batches == [2, 1, 2, 2, 1]
    expected, labels, snr, classes = (
        radioml2016.flatten_radioml2016_dataset(dataset)
    )
    np.testing.assert_array_equal(recording.samples[:], expected)
    np.testing.assert_array_equal(recording.index("mod_class_id")[:], labels)
    np.testing.assert_array_equal(recording.index("snr_db")[:], snr)
    assert recording.index("mod_class_id").attrs["labels"] == classes
    assert store.verify_integrity()


def test_radioml2016_failed_overwrite_restores_recording(
    tmp_path: Path,
) -> None:
    """Restore the original recording when replacement conversion fails.

    Args:
        tmp_path: Pytest temporary directory.
    """
    store_path = tmp_path / "store.zarr"
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    store = radioml2016.import_radioml2016_dataset(
        store_path, {("BPSK", 0): values}
    )
    original_hash = store.metadata_sha512
    with pytest.raises(ValueError, match="convert"):
        radioml2016.import_radioml2016_dataset(
            store_path,
            {("BPSK", 0): np.full((2, 2, 2), "bad")},
            overwrite_recording=True,
        )
    reopened = SigMFZarrStore.open(store_path)
    np.testing.assert_array_equal(
        reopened.recordings["radioml2016"].samples[:], values
    )
    assert reopened.metadata_sha512 == original_hash
    assert reopened.verify_integrity()
