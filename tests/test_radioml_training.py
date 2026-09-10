"""End-to-end RadioML example checks using deterministic synthetic samples."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from sigmf_zarr import (
    SigMFZarrStore,
    import_radioml2016_dataset,
    verify_inputs,
)
from sigmf_zarr.store import ZarrFormat

torch = pytest.importorskip("torch")
from examples.radioml_train import (  # noqa: E402
    create_split,
    make_loader,
    make_model,
    run_epoch,
)


@pytest.fixture(params=[2, 3])
def imported(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Import a balanced, separable fixture through the RadioML importer.

    Args:
        tmp_path: Temporary store directory.
        request: Physical Zarr format parameter.

    Returns:
        Imported store with two classes, two source SNRs, and 40 items.
    """
    rng = np.random.default_rng(7)
    buckets = {}
    for class_id, label in enumerate(["BPSK", "QPSK"]):
        for snr in [-4, 8]:
            samples = rng.normal(0, 0.01, (10, 2, 16)).astype(np.float32)
            samples[:, class_id, :] += 1
            buckets[label, snr] = samples
    path = tmp_path / "radioml.zarr"
    import_radioml2016_dataset(
        path, buckets, zarr_format=cast(ZarrFormat, request.param)
    )
    return path


def test_split_stratification_and_independence(imported: Path) -> None:
    """Keep every bucket in both partitions with reproducible assignments.

    Args:
        imported: Synthetic imported store.
    """
    for name in ["first", "second"]:
        create_split(imported, recording="radioml2016", name=name, seed=42)
    source = SigMFZarrStore.open(imported, mode="r").recordings["radioml2016"]
    first = source.split("first")
    assignments = np.asarray(first.assignments[:])
    assert np.array_equal(assignments, source.split("second").assignments[:])
    assert np.bincount(assignments).tolist() == [32, 8]
    classes = np.asarray(source.index("mod_class_id")[:])
    snrs = np.asarray(source.index("snr_db")[:])
    for class_id in [0, 1]:
        for snr in [-4, 8]:
            selected = assignments[(classes == class_id) & (snrs == snr)]
            assert np.bincount(selected).tolist() == [8, 2]
    assert "group_index" not in first.provenance
    assert first.provenance["seed"] == 42
    assert "mod_class_id and snr_db" in first.provenance["description"]
    with pytest.raises(ValueError, match="already exists"):
        create_split(imported, recording="radioml2016", name="first", seed=1)
    assert np.array_equal(assignments, source.split("first").assignments[:])


def test_training_updates_and_weighted_evaluation(imported: Path) -> None:
    """Update weights only during training and count partial batches correctly.

    Args:
        imported: Synthetic imported store.
    """
    create_split(imported, recording="radioml2016", name="holdout")
    device = torch.device("cpu")
    training = make_loader(
        imported,
        recording="radioml2016",
        split="holdout",
        partition="train",
        batch_size=3,
        workers=0,
        seed=42,
        device=device,
    )
    validation = make_loader(
        imported,
        recording="radioml2016",
        split="holdout",
        partition="validation",
        batch_size=3,
        workers=0,
        seed=42,
        device=device,
    )
    assert not set(training.dataset.source_indices) & set(
        validation.dataset.source_indices
    )
    batch = next(iter(training))
    assert batch["targets"]["modulation"].dtype == torch.int64
    assert batch["targets"]["snr_db"].dtype == torch.int16
    assert batch["samples"].dtype == torch.float32
    assert batch["samples"].device.type == "cpu"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = make_model(2, sample_count=16)
        # Zero logits yield log(2) loss and predict class zero for every item.
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.zero_()
        before = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        evaluated = run_epoch(model, validation, device)
        assert not model.training
        assert evaluated.count == 8
        assert evaluated.correct == 4
        assert evaluated.report()["loss"] == pytest.approx(np.log(2))
        assert evaluated.by_snr == {-4: [2, 4], 8: [2, 4]}
        assert all(
            torch.equal(before[name], value)
            for name, value in model.state_dict().items()
        )
        trained = run_epoch(
            model,
            training,
            device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            max_batches=2,
        )
        assert model.training
        assert trained.count == 6
        assert np.isfinite(trained.report()["loss"])
        assert any(
            not torch.equal(before[name], value)
            for name, value in model.state_dict().items()
        )
    finally:
        torch.set_num_threads(threads)
        training.dataset.close()
        validation.dataset.close()


def test_cli_training_with_spawn(imported: Path) -> None:
    """Exercise the documented CLI, worker transforms, and full validation.

    Args:
        imported: Synthetic imported store.
    """
    create_split(imported, recording="radioml2016", name="holdout")
    manifest_path = imported.parent / "experiment.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.radioml_train",
            "train",
            str(imported),
            "--split",
            "holdout",
            "--device",
            "cpu",
            "--workers",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "3",
            "--max-train-batches",
            "1",
            "--manifest",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    configuration, epoch = [
        json.loads(line) for line in result.stdout.splitlines()
    ]
    assert configuration["model"] == "VT-CNN2"
    assert configuration["labels"] == ["BPSK", "QPSK"]
    assert configuration["split"] == "holdout"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["selection"]["item_counts"] == {
        "train": 32, "validation": 8
    }
    with SigMFZarrStore.open(imported, mode="r") as store:
        assert verify_inputs(
            store.recordings["radioml2016"], manifest["inputs"]
        )
    assert epoch["train"]["count"] == 3
    assert epoch["validation"]["count"] == 8
    assert set(epoch["validation"]["by_snr_db"]) == {"-4", "8"}
    assert (
        sum(row["count"] for row in epoch["validation"]["by_snr_db"].values())
        == 8
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0, float("nan")])
def test_invalid_fraction_does_not_create_split(
    imported: Path, fraction: float
) -> None:
    """Reject invalid fractions before modifying the source recording.

    Args:
        imported: Synthetic imported store.
        fraction: Invalid requested validation fraction.
    """
    with pytest.raises(ValueError, match="validation_fraction"):
        create_split(
            imported,
            recording="radioml2016",
            name="invalid",
            validation_fraction=fraction,
        )
    recording = SigMFZarrStore.open(imported, mode="r").recordings[
        "radioml2016"
    ]
    assert recording.find_indexes("sigmf-zarr:split") == ()


def test_singleton_bucket_is_rejected(tmp_path: Path) -> None:
    """Refuse a split that cannot represent a bucket in both partitions.

    Args:
        tmp_path: Temporary store directory.
    """
    path = tmp_path / "small.zarr"
    import_radioml2016_dataset(
        path, {("BPSK", 0): np.zeros((1, 2, 8), dtype=np.float32)}
    )
    with pytest.raises(ValueError, match="at least two"):
        create_split(path, recording="radioml2016", name="invalid")


def test_vtcnn2_reference_geometry() -> None:
    """Match the published 128-sample, 11-class model dimensions and count."""
    model = make_model(11)
    assert (
        sum(parameter.numel() for parameter in model.parameters()) == 2_830_427
    )
    dropout = [layer for layer in model if isinstance(layer, torch.nn.Dropout)]
    assert len(dropout) == 3
    assert all(layer.p == 0.5 for layer in dropout)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model.eval()
        values = torch.randn(2, 2, 128)
        with torch.no_grad():
            first = model[:4](values)
            second = model[4:7](first)
            logits = model(values)
            repeated = model(values)
        assert first.shape == (2, 256, 2, 130)
        assert second.shape == (2, 80, 1, 132)
        assert logits.shape == (2, 11)
        assert torch.isfinite(logits).all()
        assert torch.equal(logits, repeated)
    finally:
        torch.set_num_threads(threads)
