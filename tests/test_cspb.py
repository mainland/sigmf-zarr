"""Tests for Chad Spooner's CSPB dataset importer."""

from __future__ import annotations

import struct
from zipfile import ZipFile

import numpy as np

from sigmf_zarr.cspb import import_cspb_dataset, load_cspb_truth
from sigmf_zarr.store import SigMFZarrStore


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
        struct.pack(f"{byte_order}ii", 2, values.size)
        + components.tobytes()
    )


def test_load_cspb_ml_truth_file(tmp_path) -> None:
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


def test_load_psk_mixtures_single_and_two_signal_truth(tmp_path) -> None:
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
        signal.source_signal_index
        for signal in truth.records[60001].signals
    ) == (1, 2)


def test_import_cspb_directory_with_dense_truth_indexes(tmp_path) -> None:
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
        "1 bpsk 11 -.001 .35 1 1 7.5 0.0\n"
        "2 qpsk 8 .002 .5 3 2 9.0 0.0\n",
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
    assert recording.sample_axes == ("iq", "time")
    assert recording.global_metadata["core:datatype"] == "cf32_le"
    assert (
        recording.global_metadata["cspb:source_dataset"]
        == "CSPB.ML.2018R2"
    )
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
    tmp_path,
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
    assert recording.has_item_metadata is True
    metadata = recording.get_item_metadata(0)
    signals = metadata["global"]["cspb:signals"]
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


def test_import_cspb_detects_each_file_byte_order(tmp_path) -> None:
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


def test_import_cspb_rejects_missing_truth_rows(tmp_path) -> None:
    """Supplying truth should require a row for every imported file.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If incomplete truth data is accepted.
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

    try:
        import_cspb_dataset(
            tmp_path / "dataset.zarr",
            source,
            truth_paths=(truth_path,),
        )
    except ValueError as exc:
        assert "missing 1 imported signal indexes: 2" in str(exc)
    else:
        raise AssertionError("Expected incomplete CSPB truth data to fail")
