"""Train VT-CNN2 on RadioML using explicit SigMF-Zarr splits.

Run from the repository root with the PyTorch extra installed:
    .venv/bin/python -m examples.radioml_train --help
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.cli.command import positive_int
from sigmf_zarr.indexes import validate_categorical_values
from sigmf_zarr.pytorch import RecordingDataset, RecordingItem


@dataclass
class Metrics:
    """Item-weighted loss and classification counts for one epoch."""

    loss_sum: float = 0.0
    """Sum of per-item cross-entropy losses."""

    count: int = 0
    """Number of evaluated items."""

    correct: int = 0
    """Number of correctly classified items."""

    by_snr: dict[int, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    """Correct and total counts for each source SNR in dB."""

    def report(self) -> dict[str, object]:
        """Return JSON-compatible metrics without averaging batch averages.

        Returns:
            Mean loss, accuracy, count, and per-SNR validation statistics.

        Raises:
            ValueError: If no items were evaluated.
        """
        if not self.count:
            raise ValueError("Cannot report metrics for an empty partition")
        return {
            "loss": self.loss_sum / self.count,
            "accuracy": self.correct / self.count,
            "count": self.count,
            "by_snr_db": {
                str(snr): {
                    "correct": correct,
                    "count": count,
                    "accuracy": correct / count,
                }
                for snr, (correct, count) in sorted(self.by_snr.items())
            },
        }


def create_split(
    path: Path,
    *,
    recording: str,
    name: str,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> None:
    """Store a seeded item split stratified by modulation and source SNR.

    Every modulation/SNR bucket must contain at least two items. This split
    does not establish independence between source transmissions.

    Args:
        path: Existing imported RadioML store to modify.
        recording: Source recording name.
        name: New split index name. Existing indexes are never overwritten.
        seed: Nonnegative NumPy random seed.
        validation_fraction: Requested fraction within each bucket.

    Raises:
        ValueError: If a bucket is too small or target values are unsupported.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    source = SigMFZarrStore.open(path, mode="r+").recordings.open(
        recording, create=False, validation="structural"
    )
    class_index = source.index("mod_class_id")
    snr_index = source.index("snr_db")
    if any(
        index.attrs.get("axis") != "item" for index in (class_index, snr_index)
    ):
        raise ValueError("RadioML targets must be item-aligned")
    classes = np.asarray(class_index[:])
    snrs = np.asarray(snr_index[:])
    validate_categorical_values(
        classes, labels=class_index.attrs.get("labels")
    )
    if (
        snrs.dtype.kind not in "iu"
        or "validity" in snr_index.attrs
        or "validity" in class_index.attrs
    ):
        raise ValueError("This example requires dense integer RadioML targets")
    # Each unique (class, SNR) pair is a stratum. Group IDs let one sort gather
    # all strata without scanning the entire item array for each pair.
    _, groups = np.unique(
        np.column_stack((classes, snrs)), axis=0, return_inverse=True
    )
    counts = np.bincount(groups)
    if not len(counts) or np.any(counts < 2):
        raise ValueError("Each modulation/SNR bucket needs at least two items")
    assignments = np.zeros(len(classes), dtype=np.uint8)
    order = np.argsort(groups, kind="stable")
    for positions in np.split(order, np.cumsum(counts)[:-1]):
        # Small strata cannot realize every requested fraction exactly. Keep
        # at least one training and one validation item in every stratum.
        count = min(
            len(positions) - 1,
            max(1, int(len(positions) * validation_fraction)),
        )
        assignments[rng.permutation(positions)[:count]] = 1
    source.add_split(
        name,
        assignments,
        labels=["train", "validation"],
        split_type="holdout",
        method="random",
        seed=seed,
        generator="examples.radioml_train v1",
        description=(
            f"Item-random split stratified by mod_class_id and snr_db; "
            f"validation_fraction={validation_fraction}. "
            "No source-transmission isolation claim."
        ),
    )
    print(
        json.dumps(
            {
                "split": name,
                "train": int(np.sum(assignments == 0)),
                "validation": int(np.sum(assignments == 1)),
            }
        ),
        flush=True,
    )


def class_id(value: Tensor) -> Tensor:
    """Convert a stored class ID to the dtype required by cross-entropy loss.

    Args:
        value: Raw scalar index tensor.

    Returns:
        Scalar CPU int64 tensor with the same category ID.
    """
    return value.to(torch.int64)


def samples_float32(value: Tensor) -> Tensor:
    """Select float32 model inputs without normalization or axis conversion.

    Args:
        value: Stored I/Q tensor.

    Returns:
        CPU float32 tensor in stored axis order.
    """
    return value.to(torch.float32)


def make_loader(
    path: Path,
    *,
    recording: str,
    split: str,
    partition: str,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[RecordingItem]:
    """Build a loader with explicit targets and serializable CPU transforms.

    Args:
        path: Imported RadioML store.
        recording: Source recording name.
        split: Stored split index name.
        partition: Train or validation label.
        batch_size: Items per batch.
        workers: Loader subprocess count. Zero selects the main process.
        seed: Loader generator seed.
        device: Model device used to select pinned memory.

    Returns:
        Loader with spawn workers and source class IDs converted to int64.

    Raises:
        ValueError: If the partition is empty or sample layout is unsupported.
    """
    dataset = RecordingDataset(
        path,
        recording=recording,
        targets={"modulation": "mod_class_id", "snr_db": "snr_db"},
        split=(split, partition),
        sample_transform=samples_float32,
        item_target_transforms={"modulation": class_id},
    )
    if not len(dataset):
        raise ValueError(f"Partition {partition!r} is empty")
    sample = dataset[0]["samples"]
    if (
        dataset.sample_axes != ("iq", "time")
        or sample.ndim != 2
        or sample.shape[0] != 2
    ):
        raise ValueError(
            "Expected sample shape (2, time) with axes (iq, time)"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=partition == "train",
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )


def make_model(class_count: int, sample_count: int = 128) -> nn.Sequential:
    """Construct the RadioML VT-CNN2 architecture with raw-logit output.

    The reference is O'Shea, Corgan, and Clancy's RadioML VT-CNN2 example.
    See docs/pytorch.md for the source and training-protocol differences.

    Args:
        class_count: Number of source modulation labels.
        sample_count: Fixed number of time samples per item.

    Returns:
        Model mapping (batch, 2, time) I/Q tensors to class logits. Two
        padded convolutions increase the time extent by four samples.

    Raises:
        ValueError: If either dimension is nonpositive.
    """
    if class_count < 1 or sample_count < 1:
        raise ValueError("Class and sample counts must be positive")
    # Treat I/Q as a two-row image with one input channel. The second kernel
    # spans both rows. Each width-3 convolution adds two time samples because
    # its padding is two per side, giving 80 * (sample_count + 4) features.
    model = nn.Sequential(
        nn.Unflatten(1, (1, 2)),
        nn.Conv2d(1, 256, kernel_size=(1, 3), padding=(0, 2)),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(256, 80, kernel_size=(2, 3), padding=(0, 2)),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Flatten(),
        nn.Linear(80 * (sample_count + 4), 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, class_count),
    )
    for layer in model:
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            continue
        assert layer.bias is not None
        nn.init.zeros_(layer.bias)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader[RecordingItem],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> Metrics:
    """Train or evaluate one epoch and accumulate item-weighted metrics.

    Args:
        model: Classifier returning raw logits.
        loader: Explicitly selected training or validation partition.
        device: Model device. Loader tensors stay on CPU until this loop.
        optimizer: Training optimizer, or None for evaluation.
        max_batches: Optional training smoke-test limit. None reads all items.

    Returns:
        Loss and accuracy counts, including each observed source SNR.
    """
    model.train(optimizer is not None)
    metrics = Metrics()
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in islice(loader, max_batches):
            samples = batch["samples"].to(device, non_blocking=True)
            targets = batch["targets"]["modulation"].to(
                device, non_blocking=True
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(samples)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            correct = (logits.argmax(dim=1) == targets).detach().cpu()
            count = targets.numel()
            # CrossEntropyLoss returns a batch mean. Recover the loss sum so
            # a short final batch has the same per-item weight as full batches.
            metrics.loss_sum += loss.item() * count
            metrics.count += count
            metrics.correct += int(correct.sum())
            snrs = batch["targets"]["snr_db"]
            for snr in snrs.unique().tolist():
                selected = snrs == snr
                metrics.by_snr[snr][0] += int(correct[selected].sum())
                metrics.by_snr[snr][1] += int(selected.sum())
    return metrics


def train(args: argparse.Namespace) -> None:
    """Run explicit training and validation partitions and print JSON metrics.

    Args:
        args: Parsed training options.

    Raises:
        ValueError: If workers, device, or class labels are unsupported.
    """
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    loaders = [
        make_loader(
            args.store,
            recording=args.recording,
            split=args.split,
            partition=partition,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed,
            device=device,
        )
        for partition in ("train", "validation")
    ]
    training, validation = loaders
    dataset = training.dataset
    assert isinstance(dataset, RecordingDataset)
    labels = dataset.labels("item", "modulation")
    if labels is None:
        raise ValueError("The modulation index must declare its labels")
    sample_count = dataset[0]["samples"].shape[-1]
    model = make_model(len(labels), sample_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(
        json.dumps(
            {
                "model": "VT-CNN2",
                "parameters": sum(
                    value.numel() for value in model.parameters()
                ),
                "device": str(device),
                "labels": labels,
                "seed": args.seed,
                "split": args.split,
                "max_train_batches": args.max_train_batches,
            }
        ),
        flush=True,
    )
    try:
        for epoch in range(args.epochs):
            trained = run_epoch(
                model,
                training,
                device,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
            )
            evaluated = run_epoch(model, validation, device)
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train": trained.report(),
                        "validation": evaluated.report(),
                    }
                ),
                flush=True,
            )
    finally:
        for loader in loaders:
            assert isinstance(loader.dataset, RecordingDataset)
            loader.dataset.close()


def main() -> None:
    """Parse the example's split-creation or training command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="Create a new stratified split")
    training = commands.add_parser("train", help="Train and evaluate VT-CNN2")
    for command in (split, training):
        command.add_argument("store", type=Path)
        command.add_argument("--recording", default="radioml2016")
        command.add_argument(
            "--split", required=True, help="Named split index"
        )
        command.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    training.add_argument("--epochs", type=positive_int, default=5)
    training.add_argument("--batch-size", type=positive_int, default=256)
    training.add_argument("--workers", type=int, default=2)
    training.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    training.add_argument(
        "--max-train-batches",
        type=positive_int,
        help=(
            "Limit training batches per epoch for smoke tests. "
            "Validation remains complete."
        ),
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.command == "split":
        create_split(
            args.store,
            recording=args.recording,
            name=args.split,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )
    else:
        train(args)


if __name__ == "__main__":
    main()
