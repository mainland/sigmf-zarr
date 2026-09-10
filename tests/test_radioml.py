"""Tests for reusable RadioML helpers."""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pytest

import sigmf_zarr
import sigmf_zarr._import_limits as import_limits
import sigmf_zarr.radioml2016 as radioml2016
import sigmf_zarr.radioml2018 as radioml2018
from sigmf_zarr.cli.sigmf import SigMFCommand
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
    assert recording.global_metadata["radioml:source_dataset"] is None


def test_radioml2018_command_line_defaults_source_dataset_to_stem(
    tmp_path: Path,
) -> None:
    """Only the command line derives a source name from the file name.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "RML2018.hdf5"
    _write_radioml2018_labels(source_path, item_count=2, num_classes=24)
    with h5py.File(source_path, "a") as source:
        source.create_dataset("X", data=np.ones((2, 4, 2), dtype=np.float32))
    store_path = tmp_path / "store.zarr"

    assert (
        SigMFCommand().run(
            ["import", "radioml2018", str(source_path), str(store_path)]
        )
        == 0
    )

    recording = SigMFZarrStore.open(store_path).recordings["radioml2018"]
    assert recording.global_metadata["radioml:source_dataset"] == "RML2018"


def _write_radioml2018_labels(
    path: Path,
    *,
    item_count: int,
    num_classes: int = 2,
) -> None:
    """Write valid RadioML 2018 `Y` and `Z` datasets to an HDF5 file.

    Args:
        path: HDF5 file to create.
        item_count: Number of items.
        num_classes: Number of one-hot columns.
    """
    labels = np.zeros((item_count, num_classes), dtype=np.float32)
    labels[:, 0] = 1.0
    with h5py.File(path, "w") as source:
        source.create_dataset("Y", data=labels)
        source.create_dataset("Z", data=np.zeros((item_count, 1), np.int64))


def _write_external_samples(directory: Path, kind: str) -> Path:
    """Write an HDF5 file whose `X` data lives in another file.

    Args:
        directory: Directory for the input and the other file.
        kind: Way in which `X` refers to the other file.

    Returns:
        Path of the HDF5 input file.
    """
    other_path = directory / "other.h5"
    secret_path = directory / "secret.bin"
    samples = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    secret_path.write_bytes(samples.tobytes())
    with h5py.File(other_path, "w") as other:
        other.create_dataset("data/X", data=samples)
    source_path = directory / "input.h5"
    _write_radioml2018_labels(source_path, item_count=1)
    with h5py.File(source_path, "a") as source:
        if kind == "external-storage":
            source.create_dataset(
                "X",
                shape=(1, 4, 2),
                dtype=np.float32,
                external=[(str(secret_path), 0, samples.nbytes)],
            )
        elif kind == "external-link":
            source["X"] = h5py.ExternalLink(str(other_path), "data/X")
        elif kind == "external-group-link":
            source["data"] = h5py.ExternalLink(str(other_path), "data")
            source["X"] = h5py.SoftLink("/data/X")
        elif kind == "soft-link-through-external-link":
            source["remote"] = h5py.ExternalLink(str(other_path), "/")
            source["X"] = h5py.SoftLink("remote/data/X")
        else:
            layout = h5py.VirtualLayout(shape=(1, 4, 2), dtype=np.float32)
            layout[:] = h5py.VirtualSource(
                str(other_path), "data/X", shape=(1, 4, 2)
            )
            source.create_virtual_dataset("X", layout)
    return source_path


@pytest.mark.parametrize(
    "kind",
    [
        "external-storage",
        "external-link",
        "external-group-link",
        "soft-link-through-external-link",
        "virtual-dataset",
    ],
)
def test_radioml2018_reads_only_its_input_file(
    tmp_path: Path,
    kind: str,
) -> None:
    """Reject samples stored in, or linked from, any other file.

    Args:
        tmp_path: Pytest temporary path fixture.
        kind: Way in which `X` refers to another file.
    """
    source_path = _write_external_samples(tmp_path, kind)
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match="input file"):
        radioml2018.import_radioml2018_dataset(
            store_path, source_path, modulation_classes=("A", "B")
        )

    assert not store_path.exists()


@pytest.mark.parametrize("target", ["/data/X", "X", "./X"])
def test_radioml2018_follows_soft_links_within_its_input_file(
    tmp_path: Path,
    target: str,
) -> None:
    """Resolve absolute and relative soft links inside the input file.

    Args:
        tmp_path: Pytest temporary path fixture.
        target: Soft-link target of `data/samples`.
    """
    source_path = tmp_path / "input.h5"
    _write_radioml2018_labels(source_path, item_count=1)
    samples = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    with h5py.File(source_path, "a") as source:
        source.create_dataset("data/X", data=samples)
        source["data/samples"] = h5py.SoftLink(target)

    store = radioml2018.import_radioml2018_dataset(
        tmp_path / "store.zarr",
        source_path,
        modulation_classes=("A", "B"),
        samples_dataset="/data/samples",
    )

    np.testing.assert_array_equal(
        store.recordings["radioml2018"].samples[:],
        np.moveaxis(samples, 2, 1),
    )


def test_radioml2018_rejects_soft_link_cycles(tmp_path: Path) -> None:
    """Stop resolving a soft-link cycle with `ValueError`.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "input.h5"
    _write_radioml2018_labels(source_path, item_count=1)
    with h5py.File(source_path, "a") as source:
        source["X"] = h5py.SoftLink("/X")

    with pytest.raises(ValueError, match="soft links"):
        radioml2018.import_radioml2018_dataset(
            tmp_path / "store.zarr",
            source_path,
            modulation_classes=("A", "B"),
        )


@pytest.mark.parametrize(
    ("shape", "chunks", "message"),
    [
        ((1, 2**24, 2), None, "per item"),
        ((64, 2**18, 2), (64, 2**18, 2), "chunks"),
    ],
    ids=["item", "chunk"],
)
def test_radioml2018_rejects_large_items_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int] | None,
    message: str,
) -> None:
    """Reject 128 MiB declared items or chunks without reading samples.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest patch fixture.
        shape: Declared sample shape, stored only as fill values.
        chunks: Declared HDF5 chunk shape.
        message: Expected error text.
    """
    source_path = tmp_path / "input.h5"
    _write_radioml2018_labels(source_path, item_count=shape[0])
    with h5py.File(source_path, "a") as source:
        source.create_dataset("X", shape=shape, dtype="<f4", chunks=chunks)
    store_path = tmp_path / "store.zarr"

    def read(*_args: object) -> None:
        """Fail if the importer reads any HDF5 data.

        Args:
            *_args: Ignored selection arguments.
        """
        raise AssertionError("HDF5 data was read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    with pytest.raises(ValueError, match=message):
        radioml2018.import_radioml2018_dataset(
            store_path, source_path, modulation_classes=("A", "B")
        )

    assert not store_path.exists()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_radioml2018_bounds_sample_batches_by_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zarr_format: ZarrFormat,
) -> None:
    """Split reads by the byte budget when `batch_size` allows more.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest patch fixture.
        zarr_format: Physical Zarr format to exercise.
    """
    source_path = tmp_path / "input.h5"
    _write_radioml2018_labels(source_path, item_count=7)
    samples = np.arange(7 * 4 * 2, dtype=np.float32).reshape(7, 4, 2)
    with h5py.File(source_path, "a") as source:
        source.create_dataset("X", data=samples)
    # One item holds 32 bytes, so a 100-byte budget allows three items.
    monkeypatch.setattr(import_limits, "IMPORT_BATCH_BYTES", 100)
    batches: list[int] = []
    original_append = SigMFRecording.append_samples

    def append(recording: SigMFRecording, batch: npt.ArrayLike) -> None:
        """Record each batch length before a real append.

        Args:
            recording: Destination recording.
            batch: Sample batch passed by the importer.
        """
        batches.append(len(np.asarray(batch)))
        original_append(recording, batch)

    monkeypatch.setattr(SigMFRecording, "append_samples", append)
    store = radioml2018.import_radioml2018_dataset(
        tmp_path / "store.zarr",
        source_path,
        modulation_classes=("A", "B"),
        zarr_format=zarr_format,
    )

    assert batches == [3, 3, 1]
    np.testing.assert_array_equal(
        store.recordings["radioml2018"].samples[:],
        np.moveaxis(samples, 2, 1),
    )
    assert store.verify_integrity()


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
    # The 3-item BPSK batch and the 5-item QPSK batch share an append.
    assert batches == [2, 2, 2, 2]
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


def _assert_same_arrays(loaded: object, expected: object) -> None:
    """Check that a loaded mapping matches a reference exactly.

    Args:
        loaded: Mapping returned by the restricted loader.
        expected: Reference mapping.
    """
    assert isinstance(loaded, dict)
    assert isinstance(expected, dict)
    assert list(loaded) == list(expected)
    for key, value in expected.items():
        array = loaded[key]
        assert type(array) is np.ndarray
        assert array.dtype == value.dtype
        assert array.shape == value.shape
        assert array.flags.writeable
        assert array.flags.f_contiguous == value.flags.f_contiguous
        assert array.tobytes(order="A") == value.tobytes(order="A")


def _sample_dataset() -> dict[tuple[str, int], npt.NDArray[Any]]:
    """Return arrays covering each byte order and storage order.

    Returns:
        Mapping of small arrays.
    """
    pattern = bytes(range(256))
    return {
        ("QPSK", 2): np.frombuffer(pattern[:64], dtype="<f4").reshape(2, 2, 4),
        ("BPSK", -4): np.frombuffer(pattern[:16], dtype="u1").reshape(2, 2, 4),
        ("8PSK", 0): np.asfortranarray(
            np.frombuffer(pattern[:48], dtype=">i2").reshape(3, 2, 4)
        ),
        ("AM-SSB", 300): np.frombuffer(pattern[64:], dtype="<f8").reshape(
            3, 2, 4
        ),
    }


def _native(
    dataset: dict[tuple[str, int], npt.NDArray[Any]],
) -> dict[tuple[str, int], npt.NDArray[Any]]:
    """Return arrays in native byte order, as NumPy unpickles them.

    NumPy converts array state to native byte order, except in protocol 5
    pickles, which rebuild arrays with `numpy.frombuffer`.

    Args:
        dataset: Mapping of arrays.

    Returns:
        Mapping of native arrays.
    """
    return {
        key: value.astype(value.dtype.newbyteorder("="))
        for key, value in dataset.items()
    }


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_loader_matches_pickle_at_each_protocol(
    tmp_path: Path, protocol: int
) -> None:
    """Every protocol must load exactly as `pickle.loads` does.

    Args:
        tmp_path: Pytest temporary path fixture.
        protocol: Pickle protocol used to write the file.
    """
    data = pickle.dumps(_sample_dataset(), protocol=protocol)
    path = tmp_path / "dataset.pkl"
    path.write_bytes(data)

    loaded = radioml2016.load_radioml2016_pickle(path, encoding="latin1")

    # The test wrote these bytes, so the unrestricted loader is safe here.
    _assert_same_arrays(loaded, pickle.loads(data))


def _python2_repr(data: bytes) -> bytes:
    """Quote bytes as Python 2 `repr` quotes a `str`.

    Args:
        data: String bytes.

    Returns:
        Quoted and escaped string.
    """
    escapes = {
        ord("\\"): b"\\\\",
        ord("'"): b"\\'",
        ord("\t"): b"\\t",
        ord("\n"): b"\\n",
        ord("\r"): b"\\r",
    }
    quoted = bytearray(b"'")
    for byte in data:
        if byte in escapes:
            quoted += escapes[byte]
        elif 0x20 <= byte < 0x7F:
            quoted.append(byte)
        else:
            quoted += b"\\x%02x" % byte
    return bytes(quoted + b"'")


def _python2_pickle(
    dataset: dict[tuple[str, int], npt.NDArray[Any]], protocol: int
) -> bytes:
    """Pickle C-ordered arrays as Python 2 with NumPy 1 does.

    Python 2 pickles strings as `str`, and NumPy 1 passes integer dtype flags
    and names `numpy.core.multiarray`. Repeated globals and dtypes are
    memoized.

    Args:
        dataset: Mapping to pickle.
        protocol: Pickle protocol 0 or 2.

    Returns:
        Pickle bytes.
    """
    memo: dict[str, int] = {}

    def ref(name: str, value: bytes) -> bytes:
        """Write a value once and refer to the memo afterward."""
        if name in memo:
            index = memo[name]
            return b"g%d\n" % index if protocol == 0 else b"h%c" % index
        memo[name] = index = len(memo) + 1
        return value + (b"p%d\n" % index if protocol == 0 else b"q%c" % index)

    def text(value: bytes) -> bytes:
        """Write a Python 2 string."""
        if protocol == 0:
            return b"S" + _python2_repr(value) + b"\n"
        if len(value) < 256:
            return b"U%c" % len(value) + value
        return b"T" + len(value).to_bytes(4, "little") + value

    def integer(value: int) -> bytes:
        """Write a Python 2 integer."""
        if protocol == 0:
            return b"I%d\n" % value
        if 0 <= value < 256:
            return b"K%c" % value
        return b"J" + value.to_bytes(4, "little", signed=True)

    def items(*values: bytes) -> bytes:
        """Write a tuple of encoded values."""
        if protocol == 2 and 1 <= len(values) <= 3:
            return b"".join(values) + bytes([0x84 + len(values)])
        return b"(" + b"".join(values) + b"t"

    def dtype(value: np.dtype[Any]) -> bytes:
        """Write a dtype with its version 3 state once."""
        if value.str in memo:
            return ref(value.str, b"")
        constructed = (
            ref("dtype", b"cnumpy\ndtype\n")
            + items(text(value.str[1:].encode()), integer(0), integer(1))
            + b"R"
        )
        state = items(
            integer(3),
            text(value.str[:1].encode()),
            b"N",
            b"N",
            b"N",
            integer(-1),
            integer(-1),
            integer(0),
        )
        return ref(value.str, constructed) + state + b"b"

    result = bytearray(b"(dp0\n" if protocol == 0 else b"\x80\x02}q\x00(")
    for (modulation, snr), values in dataset.items():
        result += items(text(modulation.encode()), integer(snr))
        result += ref("reconstruct", b"cnumpy.core.multiarray\n_reconstruct\n")
        result += items(
            ref("ndarray", b"cnumpy\nndarray\n"),
            items(integer(0)),
            text(b"b"),
        )
        result += b"R"
        result += items(
            integer(1),
            items(*(integer(size) for size in values.shape)),
            dtype(values.dtype),
            b"I00\n" if protocol == 0 else b"\x89",
            text(values.tobytes()),
        )
        result += b"b"
        if protocol == 0:
            result += b"s"
    result += b"." if protocol == 0 else b"u."
    return bytes(result)


@pytest.mark.parametrize("protocol", [0, 2])
def test_pickle_loader_accepts_python2_pickles(
    tmp_path: Path, protocol: int
) -> None:
    """Python 2 pickles like the DeepSig releases must load exactly.

    Args:
        tmp_path: Pytest temporary path fixture.
        protocol: Pickle protocol used to write the file.
    """
    dataset = {
        key: np.ascontiguousarray(value)
        for key, value in _sample_dataset().items()
    }
    data = _python2_pickle(dataset, protocol)
    path = tmp_path / "dataset.pkl"
    path.write_bytes(data)

    loaded = radioml2016.load_radioml2016_pickle(path, encoding="latin1")

    _assert_same_arrays(loaded, _native(dataset))
    # NumPy warns about the integer dtype flags that Python 2 pickles hold.
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", np.exceptions.VisibleDeprecationWarning
        )
        # The test wrote these bytes, so the unrestricted loader is safe here.
        expected = pickle.loads(data, encoding="latin1")
    _assert_same_arrays(loaded, expected)


def test_pickle_loader_accepts_string_paths(tmp_path: Path) -> None:
    """The loader must accept `str` paths like the other loaders.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "dataset.pkl"
    path.write_bytes(pickle.dumps(_sample_dataset(), protocol=4))

    loaded = sigmf_zarr.load_radioml2016_pickle(str(path), encoding="latin1")

    _assert_same_arrays(loaded, _native(_sample_dataset()))


class _Reduce:
    """Object that pickles as an arbitrary reduce value."""

    def __init__(self, *value: object) -> None:
        """Record the reduce value.

        Args:
            *value: Callable, arguments, and optional state.
        """
        self.value = value

    def __reduce__(self) -> tuple[object, ...]:
        """Return the recorded reduce value.

        Returns:
            Callable, arguments, and optional state.
        """
        return self.value


_RECONSTRUCT = np.zeros(1).__reduce__()[0]
_F4_STATE = (3, "<", None, None, None, -1, -1, 0)


def _dtype(
    code: str = "f4",
    state: tuple[object, ...] | None = _F4_STATE,
    *,
    copy: object = True,
) -> _Reduce:
    """Return a pickled dtype with arbitrary arguments and state.

    Args:
        code: Type code.
        state: `__setstate__` argument, or `None` for no state.
        copy: Copy flag.

    Returns:
        Reduce value.
    """
    if state is None:
        return _Reduce(np.dtype, (code, False, copy))
    return _Reduce(np.dtype, (code, False, copy), state)


def _array(
    dtype: object = None,
    shape: tuple[int, ...] = (1, 2, 4),
    data: object = None,
) -> _Reduce:
    """Return a pickled array with arbitrary state.

    Args:
        dtype: Pickled dtype, by default `float32`.
        shape: Declared shape.
        data: Array bytes, by default zeros of the declared size.

    Returns:
        Reduce value.
    """
    if data is None:
        data = bytes(math.prod(shape) * 4)
    return _Reduce(
        _RECONSTRUCT,
        (np.ndarray, (0,), b"b"),
        (1, shape, _dtype() if dtype is None else dtype, False, data),
    )


_SHARED_DATA = bytes(4096)

_SUBARRAY_PICKLE = (
    b"\x80\x02"
    b"cnumpy\ndtype\nX\x02\x00\x00\x00f4\x89\x88\x87R"
    b"(K\x03X\x01\x00\x00\x00<"
    b"cnumpy\ndtype\nX\x01\x00\x00\x00O\x89\x88\x87RK\x01\x85\x86"
    b"NNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb"
    b"q\x00}q\x01X\x04\x00\x00\x00BPSKK\x00\x86"
    b"cnumpy.core.multiarray\n_reconstruct\ncnumpy\nndarray\n"
    b"K\x00\x85C\x01b\x87R"
    b"(K\x01K\x01K\x02K\x04\x87h\x00\x89B\x20\x00\x00\x00"
    + b"\x41" * 32
    + b"tbs."
)
"""The review's 208-byte pickle, which crashed the importer."""

_P2_DTYPE = b"\x80\x02cnumpy\ndtype\nX\x02\x00\x00\x00f4\x89\x88\x87R"
_P2_DTYPE_STATE = (
    b"(K\x03X\x01\x00\x00\x00<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb"
)
_P2_ARRAY = (
    b"\x80\x02cnumpy.core.multiarray\n_reconstruct\ncnumpy\nndarray\n"
    b"K\x00\x85C\x01b\x87R"
)
_P2_ARRAY_STATE = (
    b"(K\x01K\x01K\x02K\x04\x87"
    + _P2_DTYPE[2:]
    + _P2_DTYPE_STATE
    + b"\x89C\x20"
    + bytes(32)
    + b"tb"
)


def _codecs_amplification() -> bytes:
    """Return the review's nested `_codecs.encode(x, "hex")` pickle.

    Returns:
        A 346-byte pickle that would build a 64 MiB value.
    """
    expression = b"C\x01A"
    for _ in range(26):
        expression = b"h\x00" + expression + b"X\x03\x00\x00\x00hex\x86R"
    return (
        b"\x80\x02c_codecs\nencode\nq\x00}q\x01X\x01\x00\x00\x00k"
        + expression
        + b"s."
    )


def _repeated_encoding() -> bytes:
    """Return a pickle that encodes one memoized string many times.

    Returns:
        Pickle that would build 2,000 copies of a 64 KiB string.
    """
    text = b"X" + (1 << 16).to_bytes(4, "little") + b"a" * (1 << 16)
    call = b"h\x00h\x01X\x06\x00\x00\x00latin1\x86R"
    return (
        b"\x80\x02c_codecs\nencode\nq\x00"
        + text
        + b"q\x01("
        + call * 2000
        + b"t."
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        # The review's 208-byte pickle hides an object subarray in float32.
        (
            {
                ("BPSK", 0): _array(
                    _dtype(
                        state=(
                            3,
                            "<",
                            (np.dtype("O"), (1,)),
                            None,
                            None,
                            -1,
                            -1,
                            0,
                        )
                    )
                )
            },
            "dtype arguments",
        ),
        (
            {
                ("BPSK", 0): _array(
                    _dtype(state=(3, "<", ("f4", (1,)), None, None, -1, -1, 0))
                )
            },
            "nested more than 2",
        ),
        # Reference-count flags and an enlarged item size.
        (
            {("BPSK", 0): _array(_dtype(state=(*_F4_STATE[:-1], 1)))},
            "state for dtype f4",
        ),
        (
            {("BPSK", 0): _array(_dtype(state=(*_F4_STATE[:-1], 0x3F)))},
            "state for dtype f4",
        ),
        (
            {
                ("BPSK", 0): _array(
                    _dtype(state=(3, "<", None, None, None, 4096, 4096, 0))
                )
            },
            "state for dtype f4",
        ),
        (
            {("BPSK", 0): _array(_dtype(state=(4, *_F4_STATE[1:], None)))},
            "state for dtype f4",
        ),
        (
            {("BPSK", 0): _array(_dtype("u1", state=_F4_STATE))},
            "state for dtype u1",
        ),
        (
            {("BPSK", 0): _array(_dtype(state=(3, "=", *_F4_STATE[2:])))},
            "state for dtype f4",
        ),
        (
            {("BPSK", 0): _array(_dtype(state=None))},
            "no recorded state",
        ),
        # Without the copy flag, NumPy would change its global dtype.
        ({("BPSK", 0): _array(_dtype(copy=False))}, "dtype arguments"),
        ({("BPSK", 0): _array(_dtype("V8"))}, "dtype arguments"),
        ({("BPSK", 0): _array(_dtype("O"))}, "dtype arguments"),
        ({("BPSK", 0): _array(_dtype("c8"))}, "dtype arguments"),
        ({("BPSK", 0): _array(_dtype("b1"))}, "dtype arguments"),
        ({("BPSK", 0): _array(_dtype("f16"))}, "dtype arguments"),
        ({("BPSK", 0): _array(_dtype("M8[s]"))}, "dtype arguments"),
        ({("BPSK", 0): _array(np.dtype("O"))}, "dtype arguments"),
        # Declared sizes are checked before allocation.
        (
            {("BPSK", 0): _array(shape=(80_000_000, 2, 1), data=bytes(32))},
            "need 640000000",
        ),
        ({("BPSK", 0): _array(data=bytes(31))}, "need 32"),
        ({("BPSK", 0): _array(data="\u0100" * 32)}, "latin"),
        ({("BPSK", 0): _array(shape=(1, 2, -4), data=b"x")}, "shape"),
        ({("BPSK", 0): _array(data=(0,))}, "array data is a tuple"),
        # Each array is built from a separate reference to shared data.
        (
            {
                ("BPSK", snr): _array(shape=(128, 2, 4), data=_SHARED_DATA)
                for snr in range(8)
            },
            "more data than the file",
        ),
        (
            {("BPSK", 0): _Reduce(_RECONSTRUCT, (np.ndarray, (0,), b"b"))},
            "no recorded state",
        ),
        (
            {("BPSK", 0): _Reduce(_RECONSTRUCT, (np.ndarray, (1,), b"b"))},
            "_reconstruct arguments",
        ),
        ({("BPSK", 0): None}, "not an array"),
        ({(("BPSK",), 0): _array()}, "dictionary key"),
        ({("BPSK", 0): np.zeros((1, 2, 4), dtype=object)}, "dtype"),
        (np.zeros((1, 2, 4), dtype=np.float32), "holds an array"),
        ([np.zeros((1, 2, 4), dtype=np.float32)], "opcode EMPTY_LIST"),
        ({("BPSK", 0.5): _array()}, "opcode BINFLOAT"),
    ],
)
def test_pickle_loader_rejects_unsafe_numpy_state(
    tmp_path: Path, payload: object, message: str
) -> None:
    """Dtype and array state must be plain, numeric, and bounded.

    Args:
        tmp_path: Pytest temporary path fixture.
        payload: Object whose pickle holds the unsafe state.
        message: Expected error text.
    """
    path = tmp_path / "unsafe.pkl"
    path.write_bytes(pickle.dumps(payload, protocol=2))
    with pytest.raises(ValueError, match="Invalid RadioML pickle") as failure:
        radioml2016.load_radioml2016_pickle(path, encoding="latin1")
    assert message in str(failure.value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "truncated"),
        (b"\x80\x02}", "truncated"),
        # A key made of 10**6 nested tuples overflowed the C stack.
        (b"\x80\x02}N" + b"\x85" * 10**6 + b"Ns.", "nested more than 2"),
        (b"\x80\x02(" + b"N" * 33 + b"t.", "more than 32 items"),
        (b"\x80\x02" + b"(" * 17, "more than 16 marks"),
        (b"\x80\x02" + b"N" * (1 << 20) + b".", "more than 1048576 opcodes"),
        (b"\x80\x04\x8e" + (1 << 62).to_bytes(8, "little"), "truncated"),
        (b"\x80\x04\x96" + (1 << 62).to_bytes(8, "little"), "truncated"),
        (b"\x80\x02\x8b\x00\x00\x10\x00.", "integer is too large"),
        (b"\x80\x02T\xff\xff\xff\xff.", "negative length"),
        (b"S'\\q'\n.", "invalid escape"),
        (b"S'unquoted\n.", "not quoted"),
        (b"\x80\x02g7\n.", "memo index 7"),
        (b"\x80\x02N\x85t.", "needs a MARK"),
        (b"\x80\x02s.", "empty stack"),
        (b"\x80\x06}.", "protocol 6"),
        (b"\x80\x02}K\x01a.", "opcode APPEND"),
        (
            b"\x80\x02c_codecs\nencode\nK\x01X\x05\x00\x00\x00utf-8\x86R.",
            "_codecs.encode arguments",
        ),
        (_codecs_amplification(), "_codecs.encode arguments"),
        (_repeated_encoding(), "encoded strings hold more data"),
        # A declared object array of 320 MB.
        (
            b"\x80\x02}X\x01\x00\x00\x00k"
            b"cnumpy.core.multiarray\n_reconstruct\ncnumpy\nndarray\n"
            b"J" + (40_000_000).to_bytes(4, "little") + b"\x85C\x01O\x87Rs.",
            "_reconstruct arguments",
        ),
        (
            _P2_DTYPE + _P2_DTYPE_STATE + _P2_DTYPE_STATE + b".",
            "cannot set the state of a dtype",
        ),
        (
            _P2_ARRAY + _P2_ARRAY_STATE + b"N\x85b.",
            "cannot set the state of an array",
        ),
        (_P2_DTYPE + b"N\x85R.", "cannot call a dtype"),
        (b"\x80\x02cnumpy\nndarray\n)R.", "cannot call global numpy.ndarray"),
        (b"\x80\x02cnumpy\nndarray\nNR.", "arguments are a NoneType"),
        (b"\x80\x02}Nb.", "BUILD state is a NoneType"),
        (b"\x80\x02}N)s.", "not an array"),
        (b"\x80\x02N(}t.", "cannot hold a dictionary"),
        (b"\x80\x02N\x8c\x05numpyN\x93.", "two strings"),
        (b"\x80\x02\xff.", "opcode 0xff"),
        (_SUBARRAY_PICKLE, "dtype arguments"),
    ],
)
def test_pickle_loader_rejects_malformed_or_unbounded_streams(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    """Malformed and unbounded opcode streams must raise `ValueError`.

    Args:
        tmp_path: Pytest temporary path fixture.
        payload: Pickle bytes.
        message: Expected error text.
    """
    path = tmp_path / "malformed.pkl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="Invalid RadioML pickle") as failure:
        radioml2016.load_radioml2016_pickle(path, encoding="latin1")
    assert message in str(failure.value)


def test_pickle_loader_rejects_unknown_encoding(tmp_path: Path) -> None:
    """An unknown string encoding must raise `ValueError`.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "dataset.pkl"
    path.write_bytes(pickle.dumps({("BPSK", 0): np.zeros((1, 2, 4))}))
    with pytest.raises(ValueError, match="no-such-codec") as failure:
        radioml2016.load_radioml2016_pickle(path, encoding="no-such-codec")
    assert isinstance(failure.value.__cause__, LookupError)


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


class _CraftedDTypeArray(np.ndarray[Any, Any]):
    """Array that reports a substitute dtype, as an unpickled dtype could."""

    crafted: SimpleNamespace

    @property
    def dtype(self) -> Any:
        """Return the substitute dtype.

        Returns:
            Object with dtype attributes.
        """
        return self.crafted


@pytest.mark.parametrize(
    "attributes",
    [
        {"subdtype": (np.dtype("O"), (1,))},
        {"hasobject": True},
        {"flags": 1},
    ],
)
def test_radioml2016_mapping_rejects_hidden_object_dtypes(
    attributes: dict[str, object],
) -> None:
    """A numeric kind must not hide subarrays, objects, or flags.

    Args:
        attributes: Dtype attributes that differ from plain `float32`.
    """
    values = np.zeros((1, 2, 4), dtype=np.float32).view(_CraftedDTypeArray)
    plain = {
        "kind": "f",
        "fields": None,
        "subdtype": None,
        "hasobject": False,
        "flags": 0,
    }
    values.crafted = SimpleNamespace(**(plain | attributes))
    with pytest.raises(TypeError, match="real numeric"):
        radioml2016.as_radioml2016_dict({("BPSK", 0): values})
    values.crafted = SimpleNamespace(**plain)
    assert radioml2016.as_radioml2016_dict({("BPSK", 0): values})


def test_import_radioml2016_rejects_empty_batches_before_writing(
    tmp_path: Path,
) -> None:
    """An empty batch must fail before the destination is created.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    dataset = {
        ("BPSK", 0): np.ones((1, 2, 4), dtype=np.float32),
        ("BPSK", 2): np.ones((0, 2, 4), dtype=np.float32),
    }
    with pytest.raises(ValueError, match=r"\('BPSK', 2\).*one item"):
        radioml2016.import_radioml2016_dataset(store_path, dataset)
    assert not store_path.exists()
