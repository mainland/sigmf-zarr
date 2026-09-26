"""Tests for RadioML 2016 key rules and per-bucket resource limits."""

from __future__ import annotations

import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import sigmf_zarr.radioml2016 as radioml2016
from sigmf_zarr.json import json_object, json_object_list
from sigmf_zarr.store import SigMFRecording

_SHARED_ARRAY = (
    b"cnumpy.core.multiarray\n_reconstruct\ncnumpy\nndarray\n"
    b"K\x00\x85C\x01b\x87R"
    b"(K\x01K\x01K\x02K\x01\x87"
    b"cnumpy\ndtype\nX\x02\x00\x00\x00i1K\x00K\x01\x87R"
    b"(K\x03X\x01\x00\x00\x00|NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb"
    b"\x89C\x02\x01\x02tb"
)
"""Protocol 2 opcodes for one `(1, 2, 1)` `int8` array of 2 bytes."""


def _bucket_pickle(count: int) -> bytes:
    """Return a pickle whose keys all refer to one memoized array.

    Each key costs about 14 bytes and each reference 2 bytes of array data,
    so the arrays stay within the file size for any key count.

    Args:
        count: Number of distinct keys.

    Returns:
        Protocol 2 pickle bytes.
    """
    data = bytearray(b"\x80\x02" + _SHARED_ARRAY + b"q\x00}(")
    for index in range(count):
        label, snr = divmod(index, 1 << 16)
        data += b"X\x01\x00\x00\x00" + b"AB"[label : label + 1]
        data += b"J" + struct.pack("<i", snr - (1 << 15)) + b"\x86h\x00"
        if index % 1000 == 999:
            data += b"u("
    return bytes(data + b"u.")


def _load(tmp_path: Path, data: bytes) -> object:
    """Write pickle bytes and load them with the restricted loader.

    Args:
        tmp_path: Pytest temporary directory.
        data: Pickle bytes.

    Returns:
        Loaded mapping.
    """
    path = tmp_path / "dataset.pkl"
    path.write_bytes(data)
    return radioml2016.load_radioml2016_pickle(path, encoding="latin1")


def test_pickle_loader_limits_dictionary_entries(tmp_path: Path) -> None:
    """A dictionary may hold 65,536 entries, but no more.

    Args:
        tmp_path: Pytest temporary directory.
    """
    loaded = _load(tmp_path, _bucket_pickle(1 << 16))
    assert isinstance(loaded, dict)
    assert len(loaded) == 1 << 16
    with pytest.raises(ValueError, match="more than 65536 entries"):
        _load(tmp_path, _bucket_pickle((1 << 16) + 1))


_ARRAY = np.zeros((1, 2, 4), dtype=np.float32)


@pytest.mark.parametrize(
    ("key", "protocol"),
    [
        # Python 2 writes Booleans as the text integers 00 and 01.
        (("BPSK", True), 0),
        (("BPSK", True), 2),
        (("BPSK", 1 << 15), 2),
        (("BPSK", -(1 << 15) - 1), 2),
        (("A" * 257, 0), 2),
        ((b"BPSK", 0), 3),
        (("BPSK", 0, 1), 2),
        (("BPSK",), 2),
        ("BPSK", 2),
        (0, 2),
        (None, 2),
        # Tuples of -1 and -2 share hash values.
        ((-1, -2, -1, -2), 2),
    ],
)
def test_pickle_loader_rejects_other_keys(
    tmp_path: Path, key: object, protocol: int
) -> None:
    """Keys must be a bounded label and a non-Boolean `int16` SNR.

    Args:
        tmp_path: Pytest temporary directory.
        key: Dictionary key to pickle.
        protocol: Pickle protocol.
    """
    with pytest.raises(ValueError, match="dictionary key must be"):
        _load(tmp_path, pickle.dumps({key: _ARRAY}, protocol=protocol))


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_loader_accepts_boundary_keys(
    tmp_path: Path, protocol: int
) -> None:
    """The longest label and the extreme SNR values must load unchanged.

    Args:
        tmp_path: Pytest temporary directory.
        protocol: Pickle protocol.
    """
    dataset = {("A" * 256, -(1 << 15)): _ARRAY, ("", (1 << 15) - 1): _ARRAY}
    loaded = _load(tmp_path, pickle.dumps(dataset, protocol=protocol))
    assert isinstance(loaded, dict)
    assert list(loaded) == list(dataset)
    for key, value in loaded.items():
        np.testing.assert_array_equal(value, dataset[key])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        # Integers that differ by 2**61 - 1 share one hash value.
        (b"\x80\x02I9223372036854775808\n.", "integer is too large"),
        (b"\x80\x02I-9223372036854775809\n.", "integer is too large"),
        (b"\x80\x02L9223372036854775808L\n.", "integer is too large"),
        (b"\x80\x02I9223372036854775807\n.", "holds a int"),
        (b"\x80\x02L-9223372036854775808L\n.", "holds a int"),
        (b"\x80\x02Np4294967296\n.", "memo index 4294967296 is out of range"),
        (b"\x80\x02Np-1\n.", "memo index -1 is out of range"),
        (b"\x80\x02Np4294967295\n.", "holds a NoneType"),
        (b"\x80\x02g4294967296\n.", "memo index 4294967296 is out of range"),
    ],
)
def test_pickle_loader_bounds_text_integers(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    """Text integers and memo indexes must fit their binary opcodes.

    Args:
        tmp_path: Pytest temporary directory.
        payload: Pickle bytes.
        message: Expected error text.
    """
    with pytest.raises(ValueError, match="Invalid RadioML pickle") as failure:
        _load(tmp_path, payload)
    assert message in str(failure.value)


def test_pickle_loader_rejects_unhashable_encoding(tmp_path: Path) -> None:
    """A `bytearray` encoding name must raise `ValueError`.

    Args:
        tmp_path: Pytest temporary directory.
    """
    payload = (
        b"\x80\x05c_codecs\nencode\nX\x01\x00\x00\x00a"
        b"\x96\x01\x00\x00\x00\x00\x00\x00\x00x\x86R."
    )
    assert len(payload) == 37
    with pytest.raises(ValueError, match=r"_codecs\.encode arguments"):
        _load(tmp_path, payload)


def test_import_radioml2016_rejects_boolean_snr(tmp_path: Path) -> None:
    """A Boolean SNR must not reach the provenance metadata.

    Args:
        tmp_path: Pytest temporary directory.
    """
    store_path = tmp_path / "store.zarr"
    with pytest.raises(TypeError, match="snr_db: int"):
        radioml2016.import_radioml2016_dataset(
            store_path, {("BPSK", True): _ARRAY}
        )
    assert not store_path.exists()


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        ({("A" * 257, 0): _ARRAY}, "at most 256 characters"),
        (
            {("BPSK", snr): _ARRAY for snr in range((1 << 12) + 1)},
            "at most 4096 batches",
        ),
    ],
)
def test_import_radioml2016_bounds_labels_and_batches(
    tmp_path: Path,
    dataset: dict[tuple[str, int], npt.NDArray[Any]],
    message: str,
) -> None:
    """Labels and batches, each copied into provenance, must be bounded.

    Args:
        tmp_path: Pytest temporary directory.
        dataset: Mapping that exceeds a limit.
        message: Expected error text.
    """
    store_path = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match=message):
        radioml2016.import_radioml2016_dataset(store_path, dataset)
    assert not store_path.exists()


def test_import_radioml2016_coalesces_small_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-item batches must share appends of `batch_size` items.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
    """
    dataset = {
        ("BPSK", snr): np.full((1, 2, 4), snr, dtype=np.float64)
        for snr in range(10)
    }
    appends: list[int] = []
    original_append = SigMFRecording.append_samples

    def append(recording: SigMFRecording, samples: npt.ArrayLike) -> None:
        """Record the item count before performing the real append.

        Args:
            recording: Destination recording.
            samples: Items passed by the importer.
        """
        appends.append(len(np.asarray(samples)))
        original_append(recording, samples)

    monkeypatch.setattr(SigMFRecording, "append_samples", append)
    store = radioml2016.import_radioml2016_dataset(
        tmp_path / "store.zarr", dataset, batch_size=4
    )
    assert appends == [4, 4, 2]
    recording = store.recordings["radioml2016"]
    expected, _, snr, _ = radioml2016.flatten_radioml2016_dataset(dataset)
    np.testing.assert_array_equal(recording.samples[:], expected)
    np.testing.assert_array_equal(recording.index("snr_db")[:], snr)
    provenance = json_object(
        recording.global_metadata["sigmf-zarr-provenance:import"],
        name="provenance",
    )
    sources = json_object_list(provenance["sources"], name="sources")
    assert [source["target_start"] for source in sources] == list(range(10))
    assert store.verify_integrity()


def test_pickle_loader_bounds_repeated_label_text(tmp_path: Path) -> None:
    """A memoized label must not be copied into more keys than the file holds.

    Args:
        tmp_path: Pytest temporary directory.
    """
    label = b"A" * 256
    data = bytearray(b"\x80\x02" + _SHARED_ARRAY + b"q\x00")
    data += b"X" + struct.pack("<I", len(label)) + label + b"q\x01}("
    for snr in range(64):
        data += b"h\x01K" + bytes([snr]) + b"\x86h\x00"
    data += b"u."
    # 64 copies of the label hold 16,384 characters, more than 16 times the
    # file size.
    assert len(data) < 1024
    with pytest.raises(ValueError, match="more label text than the file"):
        _load(tmp_path, bytes(data))
