"""Tests for Chad Spooner's CSPB dataset importer."""

from __future__ import annotations

from sigmf_zarr.cspb import load_cspb_truth


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
