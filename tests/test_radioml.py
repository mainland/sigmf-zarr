"""Tests for reusable RadioML helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

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


def test_import_radioml2016_defaults_to_sharded_zarr_format_3(
    tmp_path: Path,
) -> None:
    """The 2016 importer should avoid one physical file per item.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    dataset = {
        ("BPSK", 0): np.zeros((8192, 2, 128), dtype=np.float32),
    }

    store = radioml2016.import_radioml2016_dataset(
        tmp_path / "dataset.zarr",
        dataset,
    )

    recording = store.recordings["radioml2016"]
    assert store.zarr_format == 3
    assert recording.samples.chunks == (256, 2, 128)
    assert recording.samples.shards == (4096, 2, 128)


def test_import_radioml_dataset_writes_store(tmp_path: Path) -> None:
    """Import should write samples, labels, storage options, and hashes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    dataset = {
        ("BPSK", 0): np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            dtype=np.float32,
        ),
    }
    store = radioml2016.import_radioml2016_dataset(
        tmp_path / "store.zarr",
        dataset,
        source_dataset="RML2016.10a",
        recording_name="demo",
        overwrite_store=True,
        overwrite_recording=True,
        sample_shards=(32, 2, 3),
        zarr_format=3,
    )

    recording = store.recordings["demo"]
    assert store.zarr_format == 3
    assert recording.batched is True
    assert recording.sample_shape == (2, 3)
    assert recording.sample_axes == ("iq", "time")
    assert recording.samples.shards == (32, 2, 3)
    np.testing.assert_array_equal(recording.samples[:], dataset[("BPSK", 0)])
    np.testing.assert_array_equal(recording.index("mod_class_id")[:], [0])
    assert recording.index("mod_class_id").attrs["labels"] == ["BPSK"]
    assert recording.index("mod_class_id").attrs["field"] == (
        "rfml-dataset:modulation"
    )
    np.testing.assert_array_equal(recording.index("snr_db")[:], [0])
    assert recording.index("snr_db").attrs["field"] == "radioml:snr"
    assert store.verify_integrity()


def test_import_radioml2018_dataset_streams_hdf5_batches(
    tmp_path: Path,
) -> None:
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
    assert recording.samples.chunks == (5, 2, 4)
    assert recording.samples.shards is None
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
    expected, labels, snr, classes = radioml2016.flatten_radioml2016_dataset(
        dataset
    )
    np.testing.assert_array_equal(recording.samples[:], expected)
    np.testing.assert_array_equal(recording.index("mod_class_id")[:], labels)
    np.testing.assert_array_equal(recording.index("snr_db")[:], snr)
    assert recording.index("mod_class_id").attrs["labels"] == classes
    assert store.verify_integrity()


def test_radioml2016_failed_overwrite_restores_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore the original recording when a replacement import fails.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store_path = tmp_path / "store.zarr"
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    store = radioml2016.import_radioml2016_dataset(
        store_path, {("BPSK", 0): values}
    )
    original_hash = store.metadata_sha512

    def fail_index(*args: Any, **kwargs: Any) -> None:
        """Fail after the replacement samples have been written.

        Args:
            *args: Index arguments.
            **kwargs: Index options.

        Raises:
            OSError: Always.
        """
        raise OSError("injected index failure")

    monkeypatch.setattr(SigMFRecording, "add_index", fail_index)
    with pytest.raises(OSError, match="injected index failure"):
        radioml2016.import_radioml2016_dataset(
            store_path,
            {("BPSK", 0): np.zeros((3, 2, 2), dtype=np.float32)},
            overwrite_recording=True,
        )
    monkeypatch.undo()
    reopened = SigMFZarrStore.open(store_path)
    np.testing.assert_array_equal(
        reopened.recordings["radioml2016"].samples[:], values
    )
    assert reopened.metadata_sha512 == original_hash
    assert reopened.verify_integrity()


def test_pickle_loader_accepts_legacy_numpy_module_names(
    tmp_path: Path,
) -> None:
    """Python 2 era pickles name numpy.core rather than numpy._core.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    dataset = {("QPSK", 2): np.arange(16, dtype=np.float32).reshape(2, 2, 4)}
    legacy = pickle.dumps(dataset, protocol=2).replace(
        b"numpy._core.multiarray", b"numpy.core.multiarray"
    )
    assert b"numpy.core.multiarray" in legacy
    path = tmp_path / "legacy.pkl"
    path.write_bytes(legacy)

    loaded = radioml2016.load_radioml2016_pickle(path, encoding="latin1")

    assert isinstance(loaded, dict)
    np.testing.assert_array_equal(loaded[("QPSK", 2)], dataset[("QPSK", 2)])


class _Payload:
    """Object whose pickle would call an arbitrary function."""

    def __reduce__(self) -> tuple[object, tuple[str]]:
        """Request a call to a function outside the NumPy allowlist.

        Returns:
            Callable and arguments recorded in the pickle.
        """
        return (print, ("unpickled payload executed",))


def test_pickle_loader_rejects_other_globals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Loading must not resolve globals outside the NumPy allowlist.

    Args:
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest output capture fixture.
    """
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps({("QPSK", 2): _Payload()}, protocol=2))

    # Protocol 2 records Python 3 builtins under their Python 2 module name.
    unsupported = r"unsupported global (__builtin__|builtins)\.print"
    with pytest.raises(ValueError, match=unsupported):
        radioml2016.load_radioml2016_pickle(path, encoding="latin1")
    assert "executed" not in capsys.readouterr().out


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_pickle_loader_accepts_each_protocol(
    tmp_path: Path, protocol: int
) -> None:
    """Pickles that store each array once must load at every protocol.

    Args:
        tmp_path: Pytest temporary path fixture.
        protocol: Pickle protocol used to write the file.
    """
    dataset = {
        ("BPSK", 0): np.zeros((4, 2, 8), dtype=np.float32),
        ("QPSK", 2): np.ones((3, 2, 8), dtype=np.float64),
    }
    path = tmp_path / "dataset.pkl"
    path.write_bytes(pickle.dumps(dataset, protocol=protocol))

    loaded = radioml2016.load_radioml2016_pickle(path, encoding="latin1")

    assert isinstance(loaded, dict)
    for key, value in dataset.items():
        np.testing.assert_array_equal(loaded[key], value)


def test_pickle_loader_translates_empty_input(tmp_path: Path) -> None:
    """Empty input must follow the documented malformed-pickle contract.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "empty.pkl"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Invalid RadioML pickle") as failure:
        radioml2016.load_radioml2016_pickle(path, encoding="latin1")
    assert isinstance(failure.value.__cause__, EOFError)


def test_pickle_loader_rejects_repeated_array_data(tmp_path: Path) -> None:
    """A small pickle must not expand one array into many imported buckets.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    shared = np.zeros((100, 2, 128), dtype=np.float32)
    path = tmp_path / "aliased.pkl"
    # Pickle memoizes the array, so the file stores its data only once.
    path.write_bytes(
        pickle.dumps({("BPSK", snr): shared for snr in range(4)}, protocol=2)
    )

    with pytest.raises(ValueError, match="more than the"):
        radioml2016.load_radioml2016_pickle(path, encoding="latin1")


@pytest.mark.parametrize(
    "values",
    [
        np.empty((1, 2, 4), dtype=object),
        np.ones((1, 2, 4), dtype=np.complex64),
        np.zeros((1, 2, 4), dtype=[("i", "f4")]),
    ],
)
def test_radioml2016_mapping_requires_real_numeric_arrays(
    values: npt.NDArray[Any],
) -> None:
    """Object, complex, and structured arrays must be rejected.

    Args:
        values: Array with an unsupported dtype.
    """
    with pytest.raises(TypeError, match="real numeric"):
        radioml2016.as_radioml2016_dict({("BPSK", 0): values})
