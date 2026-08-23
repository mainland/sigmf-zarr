"""Tests for Chad Spooner's CSPB dataset importer."""

from __future__ import annotations

import struct
import tracemalloc
import zlib
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, BadZipFile, ZipFile

import numpy as np
import numpy.typing as npt
import pytest

import sigmf_zarr._import_limits as import_limits
from sigmf_zarr.cli.sigmf import SigMFCommand
from sigmf_zarr.cspb import import_cspb_dataset, load_cspb_truth
from sigmf_zarr.json import JSONObject, json_object, json_object_list
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore
from sigmf_zarr.store._common import ZarrFormat


def _complex_tim_bytes(
    samples: np.ndarray,
    *,
    byte_order: str = "<",
) -> bytes:
    """Encode complex test samples as ``.tim``.

    Args:
        samples: One-dimensional complex values.
        byte_order: ``struct`` and NumPy byte-order prefix.

    Returns:
        Encoded ``.tim`` bytes.
    """
    values = np.asarray(samples, dtype=np.complex64)
    components = np.empty((values.size * 2,), dtype=f"{byte_order}f4")
    components[0::2] = values.real
    components[1::2] = values.imag
    return (
        struct.pack(f"{byte_order}ii", 2, values.size) + components.tobytes()
    )


def test_load_cspb_ml_truth_file(tmp_path: Path) -> None:
    """The nine-field challenge layout should expose all scalar values.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "signal_record.txt"
    path.write_text(
        "1 bpsk 11 -7.4433467080e-04 9.8977795076e-01 "
        "10 9 7.8834556169e+00 0.0\n",
        encoding="utf-8",
    )

    truth = load_cspb_truth(path)
    record = truth.records[1]
    signal = record.signals[0]

    assert truth.format == "cspb-ml"
    assert signal.modulation == "BPSK"
    assert signal.base_symbol_period == 11
    assert signal.upsample_factor == 10
    assert signal.downsample_factor == 9
    assert signal.symbol_rate == (1 / 11) * (9 / 10)
    assert signal.carrier_offset == -7.4433467080e-04
    assert signal.excess_bandwidth == 9.8977795076e-01
    assert signal.inband_snr_db == 7.8834556169
    assert signal.noise_spectral_density_db == 0.0


def test_load_psk_mixtures_single_and_two_signal_truth(tmp_path: Path) -> None:
    """PSK Mixtures rows should decode component modulation metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "PM_truth.txt"
    path.write_text(
        "Index_1 1 1 .25 -.1 3 1 10.0\n"
        "Index_2 2 2 .25 -.1 0 1 10.0\n"
        "Index_60001 1 2 .25 -.1 3 1 10.0 .2 .1 4 2 5.0\n",
        encoding="utf-8",
    )

    truth = load_cspb_truth(path)

    assert truth.format == "psk-mixtures"
    assert truth.records[1].signals[0].modulation == "8PSK"
    assert truth.records[1].signals[0].source_signal_index == 1
    assert truth.records[2].signals[0].modulation == "UNKNOWN-1-0"
    assert tuple(
        signal.modulation for signal in truth.records[60001].signals
    ) == ("8PSK", "16QAM")
    assert tuple(
        signal.source_signal_index for signal in truth.records[60001].signals
    ) == (1, 2)


def test_import_cspb_directory_with_dense_truth_indexes(
    tmp_path: Path,
) -> None:
    """Challenge data should stream into a labeled batched recording.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source = tmp_path / "CSPB.ML.2018R2"
    source.mkdir()
    first = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    second = np.array([-1 + 1j, -2 + 2j, -3 + 3j], dtype=np.complex64)
    (source / "signal_1.tim").write_bytes(_complex_tim_bytes(first))
    (source / "signal_2.tim").write_bytes(_complex_tim_bytes(second))
    truth_path = tmp_path / "signal_record.txt"
    truth_path.write_text(
        "1 bpsk 11 -.001 .35 1 1 7.5 0.0\n2 qpsk 8 .002 .5 3 2 9.0 0.0\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "dataset.zarr"

    store = import_cspb_dataset(
        store_path,
        source,
        truth_paths=(truth_path,),
        source_dataset="CSPB.ML.2018R2",
        batch_size=1,
    )

    recording = store.recordings["cspb"]
    assert recording.samples.shape == (2, 2, 3)
    assert recording.samples.chunks == (2, 2, 3)
    assert recording.samples.shards is None
    assert recording.sample_axes == ("iq", "time")
    assert recording.global_metadata["core:datatype"] == "cf32_le"
    assert recording.global_metadata["cspb:source_dataset"] == "CSPB.ML.2018R2"
    np.testing.assert_array_equal(recording.samples[0, 0], first.real)
    np.testing.assert_array_equal(recording.samples[0, 1], first.imag)
    np.testing.assert_array_equal(
        recording.index("signal_id")[:],
        np.array([1, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        recording.index("mod_class_id")[:],
        np.array([0, 1], dtype=np.int16),
    )
    assert recording.index("mod_class_id").attrs["labels"] == [
        "BPSK",
        "QPSK",
    ]
    np.testing.assert_allclose(
        recording.index("inband_snr_db")[:],
        np.array([7.5, 9.0]),
    )
    assert recording.has_item_metadata is False
    assert recording.verify_integrity() is True


def test_import_cspb_zip_preserves_cochannel_metadata_in_zarr_2(
    tmp_path: Path,
) -> None:
    """A ZIP batch should import mixtures into an existing format-2 store.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    archive_path = tmp_path / "PM_Two_Batch_1.zip"
    values = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "Batch/psk_mixtures_60001.tim",
            _complex_tim_bytes(values),
        )
    truth_path = tmp_path / "PM_two_truth.txt"
    truth_path.write_text(
        "Index_60001 1 2 .25 -.1 3 1 10.0 .2 .1 4 2 5.0\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "dataset-v2.zarr"
    SigMFZarrStore.create(store_path, zarr_format=2)

    store = import_cspb_dataset(
        store_path,
        archive_path,
        truth_paths=(truth_path,),
    )

    recording = store.recordings["cspb"]
    assert store.zarr_format == 2
    assert recording.samples.shape == (1, 2, 2)
    assert recording.samples.chunks == (1, 2, 2)
    assert recording.samples.shards is None
    assert recording.has_item_metadata is True
    metadata = recording.get_item_metadata(0)
    signals = json_object_list(
        json_object(metadata["global"], name="global")["cspb:signals"],
        name="cspb:signals",
    )
    assert [signal["cspb:modulation"] for signal in signals] == [
        "8PSK",
        "16QAM",
    ]
    np.testing.assert_array_equal(
        recording.index("signal_count")[:],
        np.array([2], dtype=np.int8),
    )
    assert "mod_class_id" not in recording.indexes
    assert recording.index("source_file")[0] == (
        "PM_Two_Batch_1.zip:Batch/psk_mixtures_60001.tim"
    )


def test_import_cspb_detects_each_file_byte_order(tmp_path: Path) -> None:
    """A recording may combine little- and big-endian source files.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source = tmp_path / "source"
    source.mkdir()
    first = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    second = np.array([5 + 6j, 7 + 8j], dtype=np.complex64)
    (source / "signal_1.tim").write_bytes(
        _complex_tim_bytes(first, byte_order="<")
    )
    (source / "signal_2.tim").write_bytes(
        _complex_tim_bytes(second, byte_order=">")
    )

    store = import_cspb_dataset(tmp_path / "dataset.zarr", source)

    recording = store.recordings["cspb"]
    assert recording.global_metadata["core:datatype"] == "cf32_le"
    assert "cspb:source_byte_order" not in recording.global_metadata
    np.testing.assert_array_equal(recording.samples[0, 0], first.real)
    np.testing.assert_array_equal(recording.samples[1, 1], second.imag)


def test_import_cspb_rejects_missing_truth_rows(tmp_path: Path) -> None:
    """Supplying truth should require a row for every imported file.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source = tmp_path / "source"
    source.mkdir()
    (source / "signal_2.tim").write_bytes(
        _complex_tim_bytes(np.array([1 + 1j], dtype=np.complex64))
    )
    truth_path = tmp_path / "truth.txt"
    truth_path.write_text(
        "1 bpsk 11 -.001 .35 1 1 7.5 0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="missing 1 imported signal indexes: 2"
    ):
        import_cspb_dataset(
            tmp_path / "dataset.zarr",
            source,
            truth_paths=(truth_path,),
        )


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_import_cspb_preserves_mixed_single_signal_truth(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Retain fields that cannot form dense indexes across truth formats.

    Args:
        tmp_path: Pytest temporary directory.
        zarr_format: Physical Zarr format to exercise.
    """
    source = tmp_path / "source"
    source.mkdir()
    values = np.arange(4, dtype=np.complex64)
    for signal_index in (1, 2):
        (source / f"signal_{signal_index}.tim").write_bytes(
            _complex_tim_bytes(values)
        )
    first_truth = tmp_path / "challenge.txt"
    first_truth.write_text(
        "1 bpsk 11 -.001 .35 1 1 7.5 0.0\n", encoding="ascii"
    )
    second_truth = tmp_path / "mixture.txt"
    second_truth.write_text("Index_2 2 2 .25 -.1 3 1 10.0\n", encoding="ascii")
    store = import_cspb_dataset(
        tmp_path / "store.zarr",
        source,
        truth_paths=(first_truth, second_truth),
        batch_size=1,
        zarr_format=zarr_format,
    )
    recording = store.recordings["cspb"]
    assert recording.has_item_metadata
    for position, truth_path in enumerate((first_truth, second_truth)):
        expected = load_cspb_truth(truth_path).records[position + 1]
        assert recording.get_item_metadata(position) == (
            expected.to_item_metadata()
        )
    assert recording.index("signal_count")[:].tolist() == [1, 1]
    provenance = json_object(
        recording.global_metadata["sigmf-zarr-provenance:import"],
        name="provenance",
    )
    sources = json_object_list(provenance["sources"], name="sources")
    assert sources[0]["count"] == 2
    assert [value["role"] for value in sources[1:]] == [
        "truth",
        "truth",
    ]
    assert store.verify_integrity()


def test_import_cspb_cleans_up_failed_recording_creation(
    tmp_path: Path,
) -> None:
    """A rejected shard configuration must allow a valid retry.

    Args:
        tmp_path: Pytest temporary directory.
    """
    source = tmp_path / "signal_1.tim"
    source.write_bytes(_complex_tim_bytes(np.arange(4, dtype=np.complex64)))
    store_path = tmp_path / "store.zarr"
    SigMFZarrStore.create(store_path)
    with pytest.raises(ValueError, match="inner chunk size"):
        import_cspb_dataset(store_path, source, sample_shards=(1, 1, 1))
    assert SigMFZarrStore.open(store_path).list_recordings() == ()
    store = import_cspb_dataset(store_path, source)
    assert store.recordings["cspb"].samples.shape == (1, 2, 4)
    assert store.verify_integrity()


def test_import_cspb_failed_overwrite_restores_recording(
    tmp_path: Path,
) -> None:
    """Restore an existing recording after a later input file fails.

    Args:
        tmp_path: Pytest temporary directory.
    """
    source = tmp_path / "source"
    source.mkdir()
    (source / "signal_1.tim").write_bytes(
        _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    )
    store_path = tmp_path / "store.zarr"
    store = import_cspb_dataset(store_path, source)
    original_samples = store.recordings["cspb"].samples[:]
    original_hash = store.metadata_sha512
    (source / "signal_2.tim").write_bytes(
        _complex_tim_bytes(np.arange(3, dtype=np.complex64))
    )
    with pytest.raises(ValueError, match="shape"):
        import_cspb_dataset(
            store_path, source, batch_size=1, overwrite_recording=True
        )
    reopened = SigMFZarrStore.open(store_path)
    np.testing.assert_array_equal(
        reopened.recordings["cspb"].samples[:], original_samples
    )
    assert reopened.metadata_sha512 == original_hash
    assert reopened.verify_integrity()


def test_import_cspb_zip_checks_member_size_before_reading(
    tmp_path: Path,
) -> None:
    """A member whose size disagrees with its header must be rejected.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    archive_path = tmp_path / "PM_Two_Batch_1.zip"
    encoded = _complex_tim_bytes(np.ones(4, dtype=np.complex64))
    # Declare one million samples while storing only four.
    truncated = struct.pack("<ii", 2, 1_000_000) + encoded[8:]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Batch/psk_mixtures_60001.tim", truncated)
    truth_path = tmp_path / "PM_two_truth.txt"
    truth_path.write_text(
        "Index_60001 1 2 .25 -.1 3 1 10.0 .2 .1 4 2 5.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="header declares 8000008"):
        import_cspb_dataset(
            tmp_path / "dataset.zarr",
            archive_path,
            truth_paths=(truth_path,),
        )


def test_import_cspb_rejects_signal_index_beyond_int64(
    tmp_path: Path,
) -> None:
    """Reject an int64 overflow before creating the store.

    Args:
        tmp_path: Pytest temporary directory.
    """
    source = tmp_path / "source"
    source.mkdir()
    encoded = _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    (source / "signal_1.tim").write_bytes(encoded)
    (source / "signal_99999999999999999999.tim").write_bytes(encoded)
    store_path = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match="does not fit in int64"):
        import_cspb_dataset(store_path, source)
    assert not store_path.exists()


@pytest.mark.parametrize(
    "row",
    [
        "0_1 bpsk 11 -.001 .35 1 1 7.5 0.0",
        "\u0661 bpsk 11 -.001 .35 1 1 7.5 0.0",
        "1 bpsk 11 -.001 .35 1_0 1 7.5 0.0",
        "1 bpsk 1_1 -.001 .35 1 1 7.5 0.0",
        "1 bpsk \u0661\u0661 -.001 .35 1 1 7.5 0.0",
        "Index_\u0661 1 1 .25 -.1 3 1 10.0",
        "Index_1 0_1 1 .25 -.1 3 1 10.0",
    ],
    ids=[
        "underscore-index",
        "non-ascii-index",
        "underscore-upsample",
        "underscore-float",
        "non-ascii-float",
        "non-ascii-mixture-index",
        "underscore-mixture-index",
    ],
)
def test_load_cspb_truth_requires_ascii_numbers(
    tmp_path: Path,
    row: str,
) -> None:
    """Reject numbers that only Python's own literal syntax accepts.

    Args:
        tmp_path: Pytest temporary directory.
        row: Truth row with one invalid number.
    """
    path = tmp_path / "truth.txt"
    path.write_text(f"{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Truth line 1|Index_N|layout"):
        load_cspb_truth(path)


def test_import_cspb_requires_ascii_file_index(tmp_path: Path) -> None:
    """Reject a `.tim` file name whose index uses non-ASCII digits.

    Args:
        tmp_path: Pytest temporary directory.
    """
    source = tmp_path / "source"
    source.mkdir()
    encoded = _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    (source / "signal_\u0661.tim").write_bytes(encoded)
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match="numeric index"):
        import_cspb_dataset(store_path, source)

    assert not store_path.exists()


@pytest.mark.parametrize("container", ["directory", "zip"])
def test_import_cspb_rejects_large_items_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container: str,
) -> None:
    """Check each file's declared size against the per-item limit.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
        container: Whether the `.tim` file is extracted or archived.
    """
    encoded = _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    if container == "zip":
        source = tmp_path / "batch.zip"
        with ZipFile(source, "w") as archive:
            archive.writestr("signal_1.tim", encoded)
    else:
        source = tmp_path / "source"
        source.mkdir()
        (source / "signal_1.tim").write_bytes(encoded)
    # The file holds four complex samples, or 32 bytes of float32 data.
    monkeypatch.setattr(import_limits, "MAX_IMPORT_ITEM_BYTES", 31)
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match="importer limit"):
        import_cspb_dataset(store_path, source)

    assert not store_path.exists()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_import_cspb_bounds_batches_by_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zarr_format: ZarrFormat,
) -> None:
    """Split writes by the byte budget when `batch_size` allows more.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
        zarr_format: Physical Zarr format to exercise.
    """
    source = tmp_path / "source"
    source.mkdir()
    signals = np.arange(15, dtype=np.complex64).reshape(5, 3)
    for index, signal in enumerate(signals, start=1):
        (source / f"signal_{index}.tim").write_bytes(
            _complex_tim_bytes(signal)
        )
    # Each item holds three complex samples, or 24 bytes.
    monkeypatch.setattr(import_limits, "IMPORT_BATCH_BYTES", 50)
    original_append = SigMFRecording.append_samples
    batches: list[int] = []

    def append(
        recording: SigMFRecording,
        samples: npt.ArrayLike,
        *,
        item_metadata: Sequence[JSONObject | None] | None = None,
    ) -> None:
        """Record each batch length before a real append.

        Args:
            recording: Destination recording.
            samples: Sample batch passed by the importer.
            item_metadata: Optional per-item metadata for the real append.
        """
        batches.append(len(np.asarray(samples)))
        original_append(recording, samples, item_metadata=item_metadata)

    monkeypatch.setattr(SigMFRecording, "append_samples", append)
    store = import_cspb_dataset(
        tmp_path / "store.zarr", source, zarr_format=zarr_format
    )

    assert batches == [2, 2, 1]
    samples = store.recordings["cspb"].samples[:]
    np.testing.assert_array_equal(samples[:, 0], signals.real)
    np.testing.assert_array_equal(samples[:, 1], signals.imag)


def _zip_with_bad_crc(path: Path) -> None:
    """Write a ZIP batch whose second member fails its CRC check.

    Args:
        path: ZIP archive path.
    """
    first = _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    second = _complex_tim_bytes(np.arange(4, 8, dtype=np.complex64))
    with ZipFile(path, "w") as archive:
        archive.writestr("signal_1.tim", first)
        archive.writestr("signal_2.tim", second)
    contents = bytearray(path.read_bytes())
    # Stored members hold their bytes verbatim. Change the final sample.
    offset = contents.rindex(second) + len(second) - 1
    contents[offset] ^= 0xFF
    path.write_bytes(bytes(contents))


def test_import_cspb_rejects_corrupt_zip_member(tmp_path: Path) -> None:
    """A CRC failure must raise `ValueError` and remove the new store.

    Args:
        tmp_path: Pytest temporary directory.
    """
    archive_path = tmp_path / "batch.zip"
    _zip_with_bad_crc(archive_path)
    store_path = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match="Corrupt CSPB archive") as failure:
        import_cspb_dataset(store_path, archive_path)
    assert isinstance(failure.value.__cause__, BadZipFile)
    assert not store_path.exists()


def test_cspb_command_reports_corrupt_zip_member(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command line must report a CRC failure without a traceback.

    Args:
        tmp_path: Pytest temporary directory.
        capsys: Pytest output capture fixture.
    """
    archive_path = tmp_path / "batch.zip"
    _zip_with_bad_crc(archive_path)
    store_path = tmp_path / "store.zarr"
    with pytest.raises(SystemExit) as failure:
        SigMFCommand().run(
            ["import", "cspb", str(archive_path), str(store_path)]
        )
    assert failure.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "Corrupt CSPB archive member 'signal_2.tim'" in diagnostic
    assert "Traceback" not in diagnostic
    assert not store_path.exists()


def _deflate_member_with_extra_data(
    path: Path, member: bytes, extra: int
) -> None:
    """Write a ZIP whose deflate stream holds more than its member declares.

    The local and central headers declare the size and CRC of `member`, but
    the compressed stream continues with `extra` zero bytes.

    Args:
        path: ZIP archive path.
        member: Declared member contents.
        extra: Number of undeclared bytes after `member`.
    """
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("signal_1.tim", member + bytes(extra))
    contents = bytearray(path.read_bytes())
    fields = struct.pack("<I", zlib.crc32(member))
    size = struct.pack("<I", len(member))
    central = contents.index(b"PK\x01\x02")
    # The local header holds the CRC at 14 and the size at 22. The central
    # header holds them at 16 and 24.
    contents[14:18] = fields
    contents[22:26] = size
    contents[central + 16 : central + 20] = fields
    contents[central + 24 : central + 28] = size
    path.write_bytes(bytes(contents))


def test_import_cspb_inflates_zip_members_only_to_declared_size(
    tmp_path: Path,
) -> None:
    """Stop inflating a deflate member at the size that its header declares.

    The member's stream holds 32 MiB beyond the declared `.tim` file. The
    import must not decompress it.

    Args:
        tmp_path: Pytest temporary directory.
    """
    values = np.arange(4, dtype=np.complex64)
    archive_path = tmp_path / "batch.zip"
    _deflate_member_with_extra_data(
        archive_path, _complex_tim_bytes(values), 32 * 1024 * 1024
    )

    tracemalloc.start()
    try:
        store = import_cspb_dataset(tmp_path / "store.zarr", archive_path)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 8 * 1024 * 1024
    samples = store.recordings["cspb"].samples[:]
    np.testing.assert_array_equal(samples[0, 0], values.real)
    np.testing.assert_array_equal(samples[0, 1], values.imag)


def _zip_member_with_fields(
    path: Path, *, method: int | None = None, flags: int = 0
) -> None:
    """Write a stored ZIP member, then change its method and flag fields.

    Args:
        path: ZIP archive path.
        method: Compression method to declare, or `None` to keep it.
        flags: General-purpose flag bits to set.
    """
    encoded = _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    with ZipFile(path, "w") as archive:
        archive.writestr("signal_1.tim", encoded)
    contents = bytearray(path.read_bytes())
    central = contents.index(b"PK\x01\x02")
    # The local header holds the flags at 6 and the method at 8. The
    # central header holds them at 8 and 10.
    for flag_offset in (6, central + 8):
        (current,) = struct.unpack_from("<H", contents, flag_offset)
        struct.pack_into("<H", contents, flag_offset, current | flags)
        if method is not None:
            struct.pack_into("<H", contents, flag_offset + 2, method)
    path.write_bytes(bytes(contents))


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("bzip2", "method 12"),
        ("lzma", "method 14"),
        ("corrupt-lzma", "method 14"),
        ("deflate64", "method 9"),
        ("encrypted", "encrypted"),
    ],
)
def test_import_cspb_rejects_unbounded_zip_members(
    tmp_path: Path,
    kind: str,
    message: str,
) -> None:
    """Reject members that cannot be read with bounded memory.

    CPython decompresses a whole bzip2 or LZMA stream before truncating it,
    and cannot read deflate64 or encrypted members. Every case must raise
    `ValueError` before the store is created.

    Args:
        tmp_path: Pytest temporary directory.
        kind: Kind of member to write.
        message: Expected error text.
    """
    archive_path = tmp_path / "batch.zip"
    encoded = _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    if kind in {"bzip2", "lzma"}:
        compression = ZIP_BZIP2 if kind == "bzip2" else ZIP_LZMA
        with ZipFile(archive_path, "w", compression=compression) as archive:
            archive.writestr("signal_1.tim", encoded)
    elif kind == "corrupt-lzma":
        _zip_member_with_fields(archive_path, method=ZIP_LZMA)
    elif kind == "deflate64":
        _zip_member_with_fields(archive_path, method=9)
    else:
        _zip_member_with_fields(archive_path, flags=0x01)
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match=message):
        import_cspb_dataset(store_path, archive_path)

    assert not store_path.exists()


@pytest.mark.parametrize("target", ["file", "directory", "missing"])
def test_import_cspb_rejects_symbolic_links(
    tmp_path: Path,
    target: str,
) -> None:
    """Refuse to follow a link out of a source directory.

    Args:
        tmp_path: Pytest temporary directory.
        target: Kind of object that the link names.
    """
    encoded = _complex_tim_bytes(np.arange(4, dtype=np.complex64))
    source = tmp_path / "source"
    outside = tmp_path / "outside"
    source.mkdir()
    outside.mkdir()
    (source / "signal_1.tim").write_bytes(encoded)
    (outside / "signal_2.tim").write_bytes(encoded)
    link, destination = {
        "file": (source / "signal_2.tim", outside / "signal_2.tim"),
        "directory": (source / "more", outside),
        "missing": (source / "signal_3.tim", outside / "missing.tim"),
    }[target]
    try:
        link.symlink_to(destination)
    except OSError as exc:
        pytest.skip(f"cannot create symbolic links: {exc}")
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match="symbolic link"):
        import_cspb_dataset(store_path, source)

    assert not store_path.exists()


@pytest.mark.parametrize(
    "row",
    [
        "1 bpsk 11 -.001 .35 99999999999999999999 1 7.5 0.0",
        "1 bpsk 11 -.001 .35 9223372036854775808 1 7.5 0.0",
        "1 bpsk 11 -.001 .35 1 -9223372036854775809 7.5 0.0",
        "Index_1 1 1 .25 -.1 9223372036854775808 1 10.0",
        "Index_9223372036854775808 1 1 .25 -.1 3 1 10.0",
    ],
    ids=["upsample", "upsample-uint64", "downsample", "variant", "index"],
)
def test_load_cspb_truth_requires_int64_integers(
    tmp_path: Path,
    row: str,
) -> None:
    """Reject integer truth values that an `int64` index cannot hold.

    Args:
        tmp_path: Pytest temporary directory.
        row: Truth row with one out-of-range integer.
    """
    path = tmp_path / "truth.txt"
    path.write_text(f"{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="int64 range"):
        load_cspb_truth(path)


def test_import_cspb_bounds_modulations_before_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject more modulation labels than `int16` identifiers can number.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
    """
    count = 32769
    archive_path = tmp_path / "batch.zip"
    encoded = struct.pack("<ii", 1, 2) + struct.pack("<ff", 1.0, 2.0)
    with ZipFile(archive_path, "w") as archive:
        for index in range(1, count + 1):
            archive.writestr(f"signal_{index}.tim", encoded)
    truth_path = tmp_path / "truth.txt"
    truth_path.write_text(
        "".join(
            f"{index} M{index} 4.0 0.01 0.35 1 1 10 0\n"
            for index in range(1, count + 1)
        ),
        encoding="utf-8",
    )

    def append(*_args: object, **_kwargs: object) -> None:
        """Fail if any samples are copied.

        Args:
            *_args: Ignored arguments.
            **_kwargs: Ignored keyword arguments.
        """
        raise AssertionError("samples were copied")

    monkeypatch.setattr(SigMFRecording, "append_samples", append)
    store_path = tmp_path / "store.zarr"

    with pytest.raises(ValueError, match="at most 32768"):
        import_cspb_dataset(
            store_path, archive_path, truth_paths=(truth_path,)
        )

    assert not store_path.exists()
