"""RadioML 2018 import reads untrusted HDF5 files with bounded memory."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pytest
from h5py import h5pl

import sigmf_zarr._hdf5 as hdf5
from sigmf_zarr.radioml2018 import import_radioml2018_dataset


def _write_source(
    path: Path,
    samples: npt.NDArray[Any],
    *,
    num_classes: int = 2,
    **options: Any,
) -> None:
    """Write a RadioML 2018 file whose datasets share HDF5 options.

    Args:
        path: HDF5 file to create.
        samples: Sample tensor with shape `(N, T, 2)`.
        num_classes: Number of one-hot label columns.
        **options: `create_dataset` options for `X`, `Y`, and `Z`.
    """
    count = len(samples)
    labels = np.zeros((count, num_classes), dtype=np.int64)
    labels[np.arange(count), np.arange(count) % num_classes] = 1
    snr = (np.arange(count, dtype=np.int64) * 2 - 20).reshape(count, 1)
    label_options = dict(options)
    if "chunks" in label_options:
        label_options["chunks"] = True
    with h5py.File(path, "w") as source:
        source.create_dataset("X", data=samples, **options)
        source.create_dataset("Y", data=labels, **label_options)
        source.create_dataset("Z", data=snr, **label_options)


def _forbid_hdf5_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if anything reads HDF5 data through slicing.

    Args:
        monkeypatch: Pytest patch fixture.
    """

    def read(*_args: object) -> None:
        """Reject a read.

        Args:
            *_args: Ignored selection arguments.
        """
        raise AssertionError("HDF5 data was read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)


@pytest.mark.parametrize(
    "options",
    [
        {"compression": "gzip"},
        {"compression": "gzip", "shuffle": True},
        {"compression": "gzip", "shuffle": True, "fletcher32": True},
        {"compression": "gzip", "fletcher32": True},
        {"shuffle": True, "fletcher32": True},
    ],
    ids=[
        "gzip",
        "gzip-shuffle",
        "gzip-shuffle-fletcher32",
        "gzip-fletcher32",
        "shuffle-fletcher32",
    ],
)
@pytest.mark.parametrize(
    "chunks",
    [(3, 16, 2), (4, 5, 1)],
    ids=["item-chunks", "split-chunks"],
)
def test_radioml2018_imports_filtered_datasets(
    tmp_path: Path,
    options: dict[str, Any],
    chunks: tuple[int, int, int],
) -> None:
    """Import allowed filters exactly as HDF5 itself decodes them.

    Batches of four items cross chunk boundaries of three items, and the
    split chunks also divide the time and I/Q axes unevenly.

    Args:
        tmp_path: Pytest temporary path fixture.
        options: HDF5 filter options.
        chunks: Sample chunk shape.
    """
    source_path = tmp_path / "input.h5"
    samples = (
        np.random.default_rng(1).standard_normal((10, 16, 2)).astype(">f4")
    )
    _write_source(source_path, samples, chunks=chunks, **options)

    store = import_radioml2018_dataset(
        tmp_path / "store.zarr",
        source_path,
        modulation_classes=("A", "B"),
        batch_size=4,
    )

    recording = store.recordings["radioml2018"]
    np.testing.assert_array_equal(
        recording.samples[:], np.moveaxis(samples, 2, 1)
    )
    np.testing.assert_array_equal(
        recording.index("mod_class_id")[:], np.arange(10) % 2
    )
    np.testing.assert_array_equal(
        recording.index("snr_db")[:], np.arange(10) * 2 - 20
    )


def _write_raw_sample_chunk(path: Path, chunk: bytes) -> None:
    """Write a one-item gzip file whose only sample chunk is `chunk`.

    Args:
        path: HDF5 file to create.
        chunk: Stored bytes of the sample chunk.
    """
    _write_source(path, np.zeros((1, 128, 2), dtype=np.float32))
    with h5py.File(path, "a") as source:
        del source["X"]
        samples = source.create_dataset(
            "X",
            shape=(1, 128, 2),
            dtype=np.float32,
            chunks=(1, 128, 2),
            compression="gzip",
        )
        samples.id.write_direct_chunk((0, 0, 0), chunk, 0)


@pytest.mark.parametrize(
    ("chunk", "message"),
    [
        (zlib.compress(bytes(256 * 1024), 9), "inflates past"),
        (zlib.compress(bytes(1024))[:-8], "truncated"),
        (zlib.compress(bytes(1000)), "instead of 1024"),
        (b"not a zlib stream", "corrupt"),
        (bytes(4096), "stores a 4096-byte HDF5 chunk"),
    ],
    ids=["bomb", "truncated", "short", "corrupt", "oversized"],
)
def test_radioml2018_bounds_gzip_chunks(
    tmp_path: Path,
    chunk: bytes,
    message: str,
) -> None:
    """Reject a 1 KiB gzip chunk that does not decode to exactly 1 KiB.

    Args:
        tmp_path: Pytest temporary path fixture.
        chunk: Stored chunk bytes.
        message: Expected error text.
    """
    source_path = tmp_path / "input.h5"
    _write_raw_sample_chunk(source_path, chunk)
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match=message):
        import_radioml2018_dataset(
            store_path, source_path, modulation_classes=("A", "B")
        )

    assert not store_path.exists()


def test_radioml2018_checks_fletcher32_on_gzip_chunks(tmp_path: Path) -> None:
    """Reject a gzip chunk whose Fletcher-32 checksum does not match.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "input.h5"
    _write_source(
        source_path,
        np.ones((2, 16, 2), dtype=np.float32),
        chunks=(2, 16, 2),
        compression="gzip",
        fletcher32=True,
    )
    with h5py.File(source_path, "a") as source:
        samples = source["X"]
        mask, chunk = samples.id.read_direct_chunk((0, 0, 0))
        damaged = bytes(chunk[:-1]) + bytes([chunk[-1] ^ 0xFF])
        samples.id.write_direct_chunk((0, 0, 0), damaged, mask)

    with pytest.raises(ValueError, match="Fletcher-32"):
        import_radioml2018_dataset(
            tmp_path / "store.zarr", source_path, modulation_classes=("A", "B")
        )


@pytest.mark.parametrize("size", [1, 2, 359, 360, 361, 721, 3_000_001])
@pytest.mark.parametrize("fill", ["random", "ones"])
def test_fletcher32_matches_hdf5(tmp_path: Path, size: int, fill: str) -> None:
    """Compute the checksum that HDF5 stores, for odd and long inputs.

    Args:
        tmp_path: Pytest temporary path fixture.
        size: Chunk size in bytes.
        fill: Chunk contents.
    """
    data = (
        np.random.default_rng(size).integers(0, 256, size, dtype=np.uint8)
        if fill == "random"
        else np.full(size, 0xFF, dtype=np.uint8)
    )
    with h5py.File(tmp_path / "checksum.h5", "w") as source:
        dataset = source.create_dataset(
            "data", data=data, chunks=(size,), fletcher32=True
        )
        _mask, chunk = dataset.id.read_direct_chunk((0,))

    assert hdf5.fletcher32(chunk[:-4]) == int.from_bytes(chunk[-4:], "little")


def test_radioml2018_rejects_other_hdf5_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a filter outside the allowed list before reading data.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest patch fixture.
    """
    source_path = tmp_path / "input.h5"
    _write_source(
        source_path,
        np.ones((2, 16, 2), dtype=np.float32),
        chunks=(2, 16, 2),
        compression="lzf",
    )
    store_path = tmp_path / "store.zarr"
    _forbid_hdf5_reads(monkeypatch)

    with pytest.raises(ValueError, match="HDF5 filter 32000"):
        import_radioml2018_dataset(
            store_path, source_path, modulation_classes=("A", "B")
        )

    assert not store_path.exists()


def test_radioml2018_disables_hdf5_plugins_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty the HDF5 plugin search path during import and restore it.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest patch fixture.
    """
    source_path = tmp_path / "input.h5"
    _write_source(source_path, np.ones((2, 16, 2), dtype=np.float32))
    plugin_paths: list[int] = []
    original_read = hdf5.HDF5RowReader.read

    def read(
        reader: hdf5.HDF5RowReader, start: int, stop: int
    ) -> npt.NDArray[Any]:
        """Record the plugin search path length before a real read.

        Args:
            reader: Row reader.
            start: First row.
            stop: One past the last row.

        Returns:
            Rows read by the original method.
        """
        plugin_paths.append(h5pl.size())
        return original_read(reader, start, stop)

    monkeypatch.setattr(hdf5.HDF5RowReader, "read", read)
    before = [h5pl.get(index) for index in range(h5pl.size())]
    h5pl.append(b"/nonexistent/sigmf-zarr-plugins")
    try:
        import_radioml2018_dataset(
            tmp_path / "store.zarr", source_path, modulation_classes=("A", "B")
        )
        after = [h5pl.get(index) for index in range(h5pl.size())]
    finally:
        h5pl.remove(h5pl.size() - 1)

    assert plugin_paths
    assert set(plugin_paths) == {0}
    assert after == [*before, b"/nonexistent/sigmf-zarr-plugins"]


@pytest.mark.parametrize(
    ("dataset", "dtype"),
    [
        ("Y", np.dtype((np.float32, (4,)))),
        ("Z", np.dtype([("snr", np.int16), ("pad", np.int16)])),
        ("Y", np.dtype("S4")),
    ],
    ids=["array-labels", "compound-snr", "string-labels"],
)
def test_radioml2018_checks_label_types_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    dtype: np.dtype[Any],
) -> None:
    """Reject non-scalar label and SNR elements before reading them.

    An HDF5 array type can hide any number of values in one element.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest patch fixture.
        dataset: Dataset to replace.
        dtype: Element type of the replacement.
    """
    source_path = tmp_path / "input.h5"
    _write_source(source_path, np.ones((1, 16, 2), dtype=np.float32))
    with h5py.File(source_path, "a") as source:
        shape = source[dataset].shape
        del source[dataset]
        source.create_dataset(dataset, shape=shape, dtype=dtype)
    store_path = tmp_path / "store.zarr"
    _forbid_hdf5_reads(monkeypatch)

    with pytest.raises(ValueError, match="must be scalar"):
        import_radioml2018_dataset(
            store_path, source_path, modulation_classes=("A", "B")
        )

    assert not store_path.exists()


def test_radioml2018_reports_unallocatable_item_counts(tmp_path: Path) -> None:
    """Report a declared item count too large for memory as `ValueError`.

    The file declares 2**48 items, whose two `int16` label arrays exceed
    any 64-bit address space.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    count = 2**48
    source_path = tmp_path / "input.h5"
    with h5py.File(source_path, "w") as source:
        source.create_dataset("X", shape=(count, 1, 2), dtype=np.float32)
        source.create_dataset("Y", shape=(count, 2), dtype=np.float32)
        source.create_dataset("Z", shape=(count,), dtype=np.int16)
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match="too many"):
        import_radioml2018_dataset(
            store_path, source_path, modulation_classes=("A", "B")
        )

    assert not store_path.exists()


def test_radioml2018_bounds_class_count_by_identifier_type(
    tmp_path: Path,
) -> None:
    """Reject more classes than `int16` identifiers can number.

    Column 32768 would otherwise be stored as -32768.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    classes = [f"C{index}" for index in range(32769)]
    source_path = tmp_path / "input.h5"
    labels = np.zeros((1, len(classes)), dtype=np.uint8)
    labels[0, -1] = 1
    with h5py.File(source_path, "w") as source:
        source.create_dataset("X", data=np.ones((1, 4, 2), dtype=np.float32))
        source.create_dataset("Y", data=labels)
        source.create_dataset("Z", data=np.zeros(1, dtype=np.int16))
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match="at most 32768"):
        import_radioml2018_dataset(
            store_path, source_path, modulation_classes=classes
        )

    assert not store_path.exists()
