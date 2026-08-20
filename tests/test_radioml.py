"""Tests for reusable RadioML helpers."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import sigmf_zarr.radioml2016 as radioml2016
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
    np.testing.assert_array_equal(recording.index("snr_db")[:], [0])
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
