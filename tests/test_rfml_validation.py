"""RFML validation checks numerical contracts without certifying source
truth.
"""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.rfml import validate_rfml
from sigmf_zarr.store import SigMFZarrStore

PROFILE = {"name": "rfml-dataset", "version": "0.1.0", "optional": True}
MEASUREMENT = {
    "definition": "fixture-v1", "signal_reference": "scene",
    "time_support": "item", "noise_bandwidth": "full_sample_band",
    "interference": "excluded",
}


@pytest.mark.parametrize(
    ("field", "values", "unit", "attributes", "labels"),
    [
        ("modulation", [0, 2], None, {}, ["BPSK", "QPSK"]),
        ("class", [0, 1], None, {}, ["same", "same"]),
        ("snr", [1.0, np.nan], "dB", {"measurement": MEASUREMENT}, None),
        ("snr", [1.0, 2.0], "dB", {}, None),
        ("carrier_offset", [-0.5, 0.5], "cycles/sample", {}, None),
        ("symbol_rate", [1.0, 0.0], "symbols/sample", {}, None),
        ("symbol_rate", [1.0, 2.0], "Hz", {}, None),
        ("signal_count", [1, -1], None, {}, None),
        ("signal_count", [True, False], None, {}, None),
        ("source_id", ["known", ""], None, {}, None),
    ],
)
def test_profile_rejects_invalid_measurements(
    tmp_path: Path, field: str, values: list, unit: str | None,
    attributes: dict, labels: list | None,
) -> None:
    """Reject distinct descriptor and payload failures through the public API.

    Args:
        tmp_path: Temporary directory.
        field: Profile field being tested.
        values: Invalid payload or payload with invalid descriptors.
        unit: Declared measurement unit.
        attributes: Additional descriptors.
        labels: Optional category lookup.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(2,), sample_axes=("time",),
            global_metadata={"core:extensions": [PROFILE]},
        )
        recording.append_samples(np.ones((2, 2)))
        recording.add_index(
            "target", values, axis="item", field=f"rfml-dataset:{field}",
            unit=unit, attributes=attributes, labels=labels,
        )
        before = recording.metadata()
        report = validate_rfml(recording, batch_size=1)
        assert not report.valid
        assert report.issues[0].path == "indexes/target"
        assert recording.metadata() == before


def test_profile_conformance_does_not_certify_truth(tmp_path: Path) -> None:
    """Valid profile values do not certify native SNR or physical identities.

    Args:
        tmp_path: Temporary directory.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(2,), sample_axes=("time",),
            global_metadata={"core:extensions": [PROFILE]},
        )
        recording.append_samples(np.ones((2, 2)))
        recording.add_index(
            "target", [0, 1], axis="item", field="rfml-dataset:modulation",
            labels=["BPSK", "QPSK"],
        )
        recording.add_index(
            "snr", [0.0, 10.0], axis="item", field="rfml-dataset:snr",
            unit="dB", attributes={"measurement": MEASUREMENT},
        )
        recording.add_index("native", [0, 2], axis="item", field="native:snr")
        report = validate_rfml(recording)
        assert report.valid, report.as_dict()
        assert report.indexes_checked == 2
        assert report.unvalidated_fields == ("native:snr",)
        assert report.as_dict()["source_truth_verified"] is False


def test_rfml_cli_reports_missing_declaration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI exposes profile failures separately from store conformance.

    Args:
        tmp_path: Temporary directory.
        capsys: Captured command output.
    """
    import json

    from sigmf_zarr.cli.sigmf import SigMFCommand

    path = tmp_path / "store.zarr"
    with SigMFZarrStore.create(path) as store:
        store.recordings.open(
            "rec", batched=True, sample_shape=(2,), sample_axes=("time",)
        )
    status = SigMFCommand().run([
        "store", str(path), "recording", "rec", "validate-rfml",
        "--format", "json",
    ])
    assert status == 1
    report = json.loads(capsys.readouterr().out)
    assert report["issues"][0]["path"] == "global/core:extensions"
    assert report["source_truth_verified"] is False
