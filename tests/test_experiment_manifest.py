"""Experiment manifests bind selections and reject stale split derivations."""

from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore, capture_inputs, verify_inputs
from sigmf_zarr.experiments import dataset_manifest
from sigmf_zarr.json import JSONObject, json_object


class _ManifestOptions(TypedDict):
    """Keyword arguments for `dataset_manifest`."""

    targets: dict[str, str]
    split: str
    partitions: list[str]
    transforms: JSONObject
    parameters: JSONObject


def test_manifest_binds_targets_selection_and_split_inputs(
    tmp_path: Path,
) -> None:
    """Record exact lookup tables and reject changed split-generation inputs.

    Args:
        tmp_path: Temporary writable store directory.
    """
    with SigMFZarrStore.create(tmp_path / "data.zarr") as store:
        rec = store.recordings.create(
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
        options: _ManifestOptions = {
            "targets": {"modulation": "class"},
            "split": "holdout",
            "partitions": ["test", "test"],
            "transforms": {"samples": "identity"},
            "parameters": {"seed": 42},
        }
        manifest = dataset_manifest(rec, **options)
        selection = json_object(manifest["selection"], name="selection")
        inputs = json_object(manifest["inputs"], name="inputs")
        indexes = json_object(inputs["indexes"], name="indexes")
        class_index = json_object(indexes["class"], name="class")
        assert selection["item_counts"] == {"test": 2}
        assert json_object(class_index["attributes"], name="attributes")[
            "labels"
        ] == ["BPSK", "QPSK"]
        assert set(indexes) == {
            "class",
            "holdout",
            "session",
        }
        assert verify_inputs(rec, inputs)
        with rec.mutate_index("class") as index:
            index.attrs["labels"] = ["QPSK", "BPSK"]
        assert not verify_inputs(rec, inputs)
        with pytest.raises(ValueError, match="binding is stale"):
            dataset_manifest(rec, **options)
