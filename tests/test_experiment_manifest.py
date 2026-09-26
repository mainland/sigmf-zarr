"""Experiment manifests bind selections and reject stale split derivations."""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore, capture_inputs, verify_inputs
from sigmf_zarr.experiments import dataset_manifest


def test_manifest_binds_targets_selection_and_split_inputs(
    tmp_path: Path,
) -> None:
    """Record exact lookup tables and reject changed split-generation inputs.

    Args:
        tmp_path: Temporary writable store directory.
    """
    with SigMFZarrStore.create(tmp_path / "data.zarr") as store:
        rec = store.recordings.open(
            "rec", batched=True, sample_shape=(4,), sample_axes=("time",)
        )
        rec.append_samples(np.zeros((4, 4), dtype=np.float32))
        rec.add_index(
            "class",
            [0, 1, 0, 1],
            axis="item",
            field="test:class",
            labels=["BPSK", "QPSK"],
        )
        rec.add_index(
            "session", [0, 0, 1, 1], axis="item", field="test:session"
        )
        rec.add_split(
            "holdout",
            [0, 0, 1, 1],
            labels=["train", "test"],
            group_index="session",
            input_binding=capture_inputs(rec, indexes=["class"]),
        )
        options = dict(
            targets={"modulation": "class"},
            split="holdout",
            partitions=["test", "test"],
            transforms={"samples": "identity"},
            parameters={"seed": 42},
        )
        manifest = dataset_manifest(rec, **options)
        assert manifest["selection"]["item_counts"] == {"test": 2}
        assert manifest["inputs"]["indexes"]["class"]["attributes"][
            "labels"
        ] == ["BPSK", "QPSK"]
        assert set(manifest["inputs"]["indexes"]) == {
            "class",
            "holdout",
            "session",
        }
        assert verify_inputs(rec, manifest["inputs"])
        with rec.mutate_index("class") as index:
            index.attrs["labels"] = ["QPSK", "BPSK"]
        assert not verify_inputs(rec, manifest["inputs"])
        with pytest.raises(ValueError, match="binding is stale"):
            dataset_manifest(rec, **options)
