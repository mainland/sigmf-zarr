"""Export preflight makes sample checks and metadata dispositions explicit."""

import json
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.export_plan import INDEX_NAMESPACE, plan_sigmf_export
from sigmf_zarr.radioml2016 import import_radioml2016_dataset
from sigmf_zarr.sigmf import export_sigmf, import_sigmf
from sigmf_zarr.store import SigMFZarrStore


def test_default_import_export_preserves_declared_provenance(
    tmp_path: Path,
) -> None:
    """Export native RadioML fields with explicit namespace declarations.

    Args:
        tmp_path: Temporary source and destination paths.
    """
    with import_radioml2016_dataset(
        tmp_path / "source.zarr",
        {("BPSK", 0): np.ones((1, 2, 8), dtype=np.float32)},
    ) as store:
        names = ["mod_class_id", "snr_db"]
        plan = plan_sigmf_export(
            store, "radioml2016", item_index=0, project_indexes=names
        )
        assert plan.lossless, plan.as_dict()
        output = export_sigmf(
            store, "radioml2016", tmp_path / "signal",
            item_index=0, project_indexes=names,
        )
        restored = import_sigmf(tmp_path / "restored.zarr", output)
        assert restored.global_metadata["sigmf-zarr-provenance:import"] == (
            store.recordings["radioml2016"].global_metadata[
                "sigmf-zarr-provenance:import"
            ]
        )


def test_projected_indexes_preserve_values_and_descriptors(
    tmp_path: Path,
) -> None:
    """Selected labels and measurements survive ordinary SigMF interchange.

    Args:
        tmp_path: Temporary directory.
    """
    with SigMFZarrStore.create(tmp_path / "source.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.append_samples(np.ones((2, 4)))
        recording.add_index(
            "labels/modulation", [0, 1], axis="item", field="test:modulation",
            labels=["BPSK", "QPSK"],
        )
        recording.add_index(
            "snr", [0.0, 10.0], axis="item", field="test:snr", unit="dB",
        )
        before = recording.metadata()
        partial = plan_sigmf_export(
            store, "rec", item_index=1,
            project_indexes=("labels/modulation",), verify_samples=False,
        )
        assert partial.omitted == ("indexes/snr",)
        assert not partial.samples_verified
        assert not partial.lossless
        names = ("labels/modulation", "snr")
        plan = plan_sigmf_export(
            store, "rec", item_index=1, project_indexes=names,
        )
        assert plan.lossless, plan.as_dict()
        output = export_sigmf(
            store, "rec", tmp_path / "out", item_index=1,
            project_indexes=names,
        )
        metadata = json.loads(output.read_text())
        values = metadata["global"][f"{INDEX_NAMESPACE}:values"]
        assert values["labels/modulation"]["value"] == 1
        assert values["labels/modulation"]["label"] == "QPSK"
        assert values["snr"]["value"] == 10.0
        assert values["snr"]["attributes"]["unit"] == "dB"
        assert metadata["global"]["core:sha512"] == (
            plan.metadata["global"]["core:sha512"]
        )
        reimported = import_sigmf(tmp_path / "reimport.zarr", output)
        assert (
            reimported.global_metadata[f"{INDEX_NAMESPACE}:values"] == values
        )
        assert recording.metadata() == before


def test_preflight_reports_encoding_failure(tmp_path: Path) -> None:
    """Lossy sample conversion is a rejection, not an authorized omission.

    Args:
        tmp_path: Temporary directory.
    """
    with SigMFZarrStore.create(tmp_path / "source.zarr") as store:
        recording = store.recordings.open(
            "rec", sample_shape=(1,), sample_axes=("time",),
            global_metadata={"core:datatype": "ri8"},
        )
        with recording.mutate_samples() as samples:
            samples[:] = 0.5
        before = recording.metadata()
        plan = plan_sigmf_export(store, "rec")
        assert "cannot be represented" in plan.rejected[0]
        assert not plan.lossless
        assert recording.metadata() == before
        missing = plan_sigmf_export(store, "missing")
        assert missing.rejected
        assert "missing" not in store.recordings


def test_projection_rejects_conflicting_json(tmp_path: Path) -> None:
    """Explicit projection cannot overwrite an existing meaning silently.

    Args:
        tmp_path: Temporary directory.
    """
    with SigMFZarrStore.create(tmp_path / "source.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(2,), sample_axes=("time",),
            global_metadata={
                "core:datatype": "rf32_le", f"{INDEX_NAMESPACE}:values": {},
            },
        )
        recording.append_samples(np.ones((1, 2)))
        recording.add_index("label", [0], axis="item", field="test:label")
        with pytest.raises(ValueError, match="conflicts"):
            export_sigmf(
                store, "rec", tmp_path / "out", item_index=0,
                project_indexes=("label",),
            )
        assert not (tmp_path / "out.sigmf-meta").exists()


def test_preflight_rejects_invalid_core_metadata(tmp_path: Path) -> None:
    """A sample-compatible export must still satisfy the SigMF core schema.

    Args:
        tmp_path: Temporary directory.
    """
    with SigMFZarrStore.create(tmp_path / "source.zarr") as store:
        store.recordings.open(
            "rec", sample_shape=(2,), sample_axes=("time",),
            global_metadata={
                "core:datatype": "rf32_le", "core:sample_rate": "unknown",
            },
        )
        plan = plan_sigmf_export(store, "rec")
        assert plan.rejected
        assert not plan.samples_verified
