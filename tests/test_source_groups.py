"""Constituent source reuse remains isolated through transitive mixtures."""

import struct
from pathlib import Path

import numpy as np
import pytest

from examples.cspb_split import CSPB_SOURCES, create_split
from sigmf_zarr import SigMFZarrStore, import_cspb_dataset, verify_inputs
from sigmf_zarr.sources import source_groups


def test_transitive_source_isolation_and_fresh_validation(
    tmp_path: Path,
) -> None:
    """Group A, A+B, B+C, C and reject leakage after a metadata edit.

    Args:
        tmp_path: Temporary writable store directory.
    """
    path = tmp_path / "mixtures.zarr"
    with SigMFZarrStore.create(path) as store:
        rec = store.recordings.open(
            "cspb",
            batched=True,
            sample_shape=(4,),
            sample_axes=("time",),
            global_metadata={"cspb:truth_formats": ["psk-mixtures"]},
        )
        components = [[1], [1, 2], [2, 3], [3], [], [7]]
        rec.append_samples(
            np.zeros((6, 4), dtype=np.float32),
            item_metadata=[
                {
                    "global": {
                        "cspb:signals": [
                            {"cspb:source_signal_index": value}
                            for value in values
                        ]
                    }
                }
                for values in components
            ],
        )
        rec.add_index("signal_id", np.arange(6), axis="item", field="cspb:id")
        np.testing.assert_array_equal(
            source_groups(rec, CSPB_SOURCES, batch_size=2), [0, 0, 0, 0, 1, 2]
        )
        with pytest.raises(ValueError, match="multiple partitions"):
            rec.add_split(
                "bad",
                [0, 0, 1, 1, 1, 1],
                labels=["train", "test"],
                group_sources=CSPB_SOURCES,
            )
    counts = create_split(path, name="holdout", validation_fraction=0.5)
    assert counts["groups"] == 3
    with SigMFZarrStore.open(path, mode="r+") as store:
        rec = store.recordings["cspb"]
        split = rec.split("holdout")
        split.validate()
        binding = split.provenance["inputs"]
        assert verify_inputs(rec, binding)
        assignments = split.assignments[:]
        opposite = int(np.flatnonzero(assignments != assignments[0])[0])
        rec.set_item_metadata_entry(
            opposite,
            {"global": {"cspb:signals": [{"cspb:source_signal_index": 1}]}},
        )
        assert not verify_inputs(rec, binding)
        with pytest.raises(ValueError, match="multiple partitions"):
            split.validate()


@pytest.mark.parametrize("value", [None, {}, [{"other": 1}], [True]])
def test_unknown_components_do_not_use_fallback(tmp_path: Path, value) -> None:
    """Reject unknown source identities instead of using the item's file ID.

    Args:
        tmp_path: Temporary store directory.
        value: Malformed component metadata.
    """
    with SigMFZarrStore.create(tmp_path / "bad.zarr") as store:
        rec = store.recordings.open(
            "rec", batched=True, sample_shape=(2,), sample_axes=("time",)
        )
        rec.append_samples(
            np.zeros((1, 2), dtype=np.float32),
            item_metadata=[{"global": {"cspb:signals": value}}],
        )
        rec.add_index("signal_id", [1], axis="item", field="cspb:id")
        with pytest.raises(ValueError):
            source_groups(rec, CSPB_SOURCES)


@pytest.mark.parametrize("mixtures", [False, True])
def test_native_cspb_holdout(tmp_path: Path, mixtures: bool) -> None:
    """Consume native truth, including a mixture that connects every source.

    Args:
        tmp_path: Temporary source and store directory.
        mixtures: Whether to add transitive mixtures to the singleton files.
    """
    source = tmp_path / "tim"
    source.mkdir()
    ids = [1, 2, 3, 4] + ([10, 11, 12] if mixtures else [])
    for identity in ids:
        (source / f"signal_{identity}.tim").write_bytes(
            struct.pack("<ii", 2, 4) + np.zeros(8, dtype="<f4").tobytes()
        )
    rows = [f"Index_{i} {i} {i} .25 -.1 3 1 10.0\n" for i in range(1, 5)]
    if mixtures:
        rows.extend(
            f"Index_{i} {a} {b} .25 -.1 3 1 10.0 .2 .1 4 2 5.0\n"
            for i, a, b in [(10, 1, 2), (11, 2, 3), (12, 3, 4)]
        )
    truth = tmp_path / "truth.txt"
    truth.write_text("".join(rows), encoding="ascii")
    path = tmp_path / "native.zarr"
    with import_cspb_dataset(path, source, truth_paths=[truth]) as store:
        groups = source_groups(store.recordings["cspb"], CSPB_SOURCES)
        assert len(np.unique(groups)) == (1 if mixtures else 4)
    if mixtures:
        with pytest.raises(ValueError, match="independent source groups"):
            create_split(path, name="holdout")
    else:
        assert create_split(path, name="holdout")["groups"] == 4
