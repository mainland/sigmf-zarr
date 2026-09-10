"""Dense tensor loading, independent indexes, and process-owned handles."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sigmf_zarr import SigMFRecording, SigMFZarrStore
from sigmf_zarr.store import ZarrFormat

# Core installations must remain usable without the optional dependency.
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from sigmf_zarr.pytorch import (  # noqa: E402
    RecordingDataset,
    RecordingItem,
    _tensor_dtype,
)


@pytest.fixture(params=[2, 3])
def source(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[Path, SigMFRecording]:
    """Create chunked samples, independent target indexes, and named splits.

    Args:
        tmp_path: Temporary store directory.
        request: Physical Zarr format parameter.

    Returns:
        Store path and writable source recording.
    """
    path = tmp_path / "data.zarr"
    store = SigMFZarrStore.create(
        path, zarr_format=cast(ZarrFormat, request.param)
    )
    recording = store.recordings.open(
        "rec",
        batched=True,
        sample_shape=(2, 4),
        sample_axes=("iq", "time"),
        sample_dtype=">f4",
        sample_chunks=(2, 2, 4),
        sample_shards=(4, 2, 4) if request.param == 3 else None,
        global_metadata={"example:shared": "yes"},
    )
    recording.append_samples(np.arange(48, dtype=np.float32).reshape(6, 2, 4))
    recording.add_index(
        "class",
        np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8),
        axis="item",
        field="example:class",
        labels=["A", "B"],
    )
    recording.add_index(
        "snr", np.arange(6, dtype=">i2"), axis="item", field="example:snr"
    )
    recording.add_index(
        "session", [0, 0, 1, 1, 2, 2], axis="item", field="example:session"
    )
    recording.add_split(
        "split",
        [0, 0, 1, 1, 2, 2],
        labels=["train", "validation", "test"],
        group_index="session",
    )
    recording.set_item_metadata([{"global": {"example:class": "JSON"}}] * 6)
    return path, recording


def _dataset(path: Path, **kwargs: Any) -> RecordingDataset:
    """Open standard test targets with overridable explicit options.

    Args:
        path: Source store path.
        **kwargs: Constructor overrides.

    Returns:
        Configured dataset.
    """
    options = {
        "recording": "rec",
        "targets": {"class": "class", "snr": "snr"},
        "split": None,
    }
    options.update(kwargs)
    return RecordingDataset(path, **options)


def test_dense_items_and_selection(
    source: tuple[Path, SigMFRecording],
) -> None:
    """Preserve stored dtypes, axes, source positions, and raw index IDs.

    Args:
        source: Example store and recording.
    """
    path, recording = source
    with recording.mutate_index("session") as groups:
        groups[5] = 0
    dataset = _dataset(path, split=("split", ["test", "train"]))
    assert dataset._store is None
    assert dataset.sample_axes == ("iq", "time")
    assert dataset.source_indices.tolist() == [0, 1, 4, 5]
    dataset.source_indices[:] = 99
    assert len(dataset) == 4
    item = dataset[-1]
    assert item["item_index"] == 5
    assert item["samples"].dtype == torch.float32
    assert item["samples"].tolist() == [[40, 41, 42, 43], [44, 45, 46, 47]]
    assert item["targets"]["class"].dtype == torch.uint8
    assert item["targets"]["class"].item() == 1
    assert item["targets"]["snr"].dtype == torch.int16
    assert dataset.labels("item", "class") == ("A", "B")
    assert dataset.labels("item", "snr") is None
    assert dataset.metadata(0)["global"]["example:class"] == "JSON"
    with pytest.raises(ValueError, match="multiple partitions"):
        recording.validate_split("split")
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["item_index"].tolist() == [0, 1, 4, 5]
    assert batch["samples"].shape == (4, 2, 4)
    dataset.close()
    assert dataset._store is None
    assert dataset[0]["item_index"] == 0


def test_duplicate_transform_order(
    source: tuple[Path, SigMFRecording],
) -> None:
    """Run each occurrence independently and protect even reused transform
    buffers.

    Args:
        source: Example store and recording.
    """
    calls = []
    reused = torch.empty((2, 4))

    def samples(value: torch.Tensor) -> torch.Tensor:
        """Record sample transform order and deliberately reuse output
        storage.
        """
        calls.append(("sample", value.flatten()[0].item()))
        return reused.copy_(value).add_(1)

    def target(value: torch.Tensor) -> torch.Tensor:
        """Expand scalar targets before the joint transform.

        Args:
            value: Raw categorical target.

        Returns:
            One-element integer target vector.
        """
        calls.append(("target", value.item()))
        return value.to(torch.int64).reshape(1)

    def joint(item: RecordingItem) -> RecordingItem:
        """Record final transform order and apply joint changes.

        Args:
            item: Item after sample and target transforms.

        Returns:
            Jointly transformed item.
        """
        calls.append(("item", item["item_index"]))
        item["samples"].add_(len(calls))
        return item

    dataset = _dataset(
        source[0],
        sample_transform=samples,
        item_target_transforms={"class": target},
        item_transform=joint,
    )
    items = dataset.__getitems__([5, 5, 2])
    assert [item["item_index"] for item in items] == [5, 5, 2]
    assert calls == [
        ("sample", 40),
        ("target", 1),
        ("item", 5),
        ("sample", 40),
        ("target", 1),
        ("item", 5),
        ("sample", 16),
        ("target", 0),
        ("item", 2),
    ]
    assert [item["samples"][0, 0].item() for item in items] == [44, 47, 26]
    items[0]["samples"].zero_()
    items[0]["targets"]["class"].zero_()
    assert items[1]["samples"][0, 0].item() == 47
    assert items[1]["targets"]["class"].item() == 1
    assert _dataset(source[0])[5]["samples"][0, 0].item() == 40


@pytest.mark.parametrize("context", mp.get_all_start_methods())
def test_worker_reopen_after_parent_access(
    source: tuple[Path, SigMFRecording], context: str
) -> None:
    """Load in every local worker context after the parent has cached handles.

    Args:
        source: Example store and recording.
        context: Multiprocessing start method.
    """
    dataset = _dataset(source[0], split=("split", "test"))
    assert dataset[0]["item_index"] == 4
    assert dataset._pid == os.getpid()
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._store is None
    assert restored._arrays == {}
    assert restored[1]["item_index"] == 5
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        multiprocessing_context=context,
        persistent_workers=True,
        timeout=30,
    )
    for _ in range(2):
        assert [batch["item_index"].item() for batch in loader] == [4, 5]
    assert dataset[1]["item_index"] == 5


def test_remote_fork_guard(
    source: tuple[Path, SigMFRecording], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject inherited remote handles before attempting backend operations.

    Args:
        source: Example store and recording.
        monkeypatch: Process-context overrides.
    """
    dataset = _dataset(source[0])
    dataset[0]
    dataset._path = "s3://example/data.zarr"
    monkeypatch.setattr(mp, "parent_process", lambda: object())
    monkeypatch.setattr(mp, "get_start_method", lambda: "fork")
    with pytest.raises(RuntimeError, match="multiprocessing_context='spawn'"):
        dataset[0]


def test_lazy_json_and_index_validation(
    source: tuple[Path, SigMFRecording],
) -> None:
    """Defer unrequested JSON and target payload validation until explicit
    access.

    Args:
        source: Example store and recording.
    """
    path, recording = source
    recording._raw_group["item_metadata"][5] = "{bad json"
    dataset = _dataset(path)
    assert dataset[5]["targets"]["class"].item() == 1
    assert dataset.metadata(0)["global"]["example:class"] == "JSON"
    with pytest.raises(ValueError):
        dataset.metadata(5)
    dataset.close()
    with recording.mutate_index("class") as ids:
        ids[5] = 2
    with pytest.raises(ValueError, match="outside"):
        _dataset(path)[5]
    recording.add_index(
        "missing",
        [0.0, np.nan, 2, 3, 4, 5],
        axis="item",
        field="example:missing",
    )
    with pytest.raises(ValueError, match="nonfinite"):
        _dataset(path, targets={"value": "missing"})[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"targets": {"x": "absent"}},
        {"targets": {"": "class"}},
        {"split": ("split", "absent")},
        {"split": ("split", [])},
        {"split": "train"},
        {"item_target_transforms": {"absent": torch.clone}},
        {"storage_options": {"client": object()}},
    ],
)
def test_reject_invalid_configuration(
    source: tuple[Path, SigMFRecording], kwargs: dict[str, Any]
) -> None:
    """Reject unsupported selections before exposing items.

    Args:
        source: Example store and recording.
        kwargs: Malformed constructor options.
    """
    with pytest.raises((ValueError, TypeError, KeyError)):
        _dataset(source[0], **kwargs)


@pytest.mark.parametrize(
    "attributes",
    [
        {"validity": "mask"},
        {"labels": ["A", "A"]},
        {"labels": [1, 2]},
        {"labels": []},
    ],
)
def test_reject_target_descriptors(
    source: tuple[Path, SigMFRecording], attributes: dict[str, Any]
) -> None:
    """Reject nullable and malformed lookup descriptors without scanning
    values.

    Args:
        source: Example store and recording.
        attributes: Invalid target attributes.
    """
    path, recording = source
    recording._raw_group["indexes/class"].attrs.update(attributes)
    with pytest.raises(ValueError):
        _dataset(path)


@pytest.mark.parametrize(
    "transform",
    [
        lambda _: None,
        lambda item: {**item, "item_index": 99},
        lambda item: {**item, "targets": {}},
        lambda item: {**item, "samples": torch.empty(1, device="meta")},
    ],
)
def test_reject_invalid_transform_outputs(
    source: tuple[Path, SigMFRecording], transform: Any
) -> None:
    """Enforce item structure, identity, and CPU tensor output.

    Args:
        source: Example store and recording.
        transform: Malformed item transform.
    """
    with pytest.raises(TypeError):
        _dataset(source[0], item_transform=transform)[0]


def test_empty_targets_and_positions(
    source: tuple[Path, SigMFRecording],
) -> None:
    """Support unlabeled loading and standard sampler position semantics.

    Args:
        source: Example store and recording.
    """
    dataset = _dataset(source[0], targets={})
    assert dataset[0]["targets"] == {}
    assert dataset.__getitems__([]) == []
    for position in [6, -7]:
        with pytest.raises(IndexError):
            dataset[position]
    for position in [True, np.bool_(False), 1.2]:
        with pytest.raises(TypeError):
            dataset[position]
    with pytest.raises(ValueError):
        dataset.labels("signal", "class")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        dataset.labels("item", "class")
    ranks = [
        list(
            DistributedSampler(
                dataset, num_replicas=2, rank=rank, shuffle=False
            )
        )
        for rank in range(2)
    ]
    assert sorted(ranks[0] + ranks[1]) == list(range(6))
    items = dataset.__getitems__([5, 0, 5, 2])
    for position, item in zip([5, 0, 5, 2], items, strict=True):
        assert torch.equal(item["samples"], dataset[position]["samples"])


def test_unsupported_layout_and_dtype(tmp_path: Path) -> None:
    """Reject continuous recordings and NumPy types without tensor mappings.

    Args:
        tmp_path: Temporary store directory.
    """
    path = tmp_path / "continuous.zarr"
    SigMFZarrStore.create(path).recordings.open("rec", sample_shape=(4,))
    with pytest.raises(ValueError, match="batched"):
        _dataset(path, targets={})
    for dtype in ["U8", "O", "datetime64[ns]", "V8"]:
        with pytest.raises(ValueError, match="Unsupported tensor dtype"):
            _tensor_dtype(np.dtype(dtype))


def test_base_import_without_torch() -> None:
    """Keep core import independent of the optional PyTorch module."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sigmf_zarr; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
