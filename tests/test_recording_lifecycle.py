"""Tests for opening and creating recordings and their default chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr.core.array import ShardsLike

from sigmf_zarr.store import SigMFRecording, SigMFZarrStore


def test_recordings_open_and_create_are_strict(tmp_path: Path) -> None:
    """Recordings must be opened only when present and created explicitly.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    with pytest.raises(KeyError):
        store.recordings.open("missing")
    with pytest.raises(KeyError):
        store.recordings["missing"]
    assert store.list_recordings() == ()

    store.recordings.create(
        "rec", sample_shape=(4,), global_metadata={"test:label": "first"}
    )
    with pytest.raises(ValueError, match="already exists"):
        store.recordings.create("rec", sample_shape=(8,))
    original = store.recordings["rec"]
    assert original.sample_shape == (4,)
    assert original.global_metadata["test:label"] == "first"

    replaced = store.recordings.create(
        "rec", sample_shape=(8,), overwrite=True
    )
    assert replaced.sample_shape == (8,)
    assert "test:label" not in replaced.global_metadata
    with pytest.raises(ValueError, match="requires create=True"):
        SigMFRecording(store, "rec", overwrite=True)


@pytest.mark.parametrize(
    ("batched", "dtype", "sample_shape", "sample_axes", "chunks"),
    [
        (True, np.float32, (2, 128), None, (1024, 2, 128)),
        (True, np.int16, (3,), ("time",), (174762, 3)),
        (False, np.float32, (2, 16), None, (2, 131072)),
        (False, np.float64, (5,), None, (131072,)),
        (
            False,
            np.float32,
            (3, 2, 10),
            ("channel", "iq", "time"),
            (3, 2, 43690),
        ),
        (False, np.float32, (10, 2), ("time", "iq"), (131072, 2)),
    ],
)
def test_created_recordings_use_one_mebibyte_growth_chunks(
    tmp_path: Path,
    batched: bool,
    dtype: type[np.generic],
    sample_shape: tuple[int, ...],
    sample_axes: tuple[str, ...] | None,
    chunks: tuple[int, ...],
) -> None:
    """Default chunks span about 1 MiB along the growth axis.

    Chunks keep complete items or complete I/Q and channel planes, and they
    do not depend on the initial length of the growth axis.

    Args:
        tmp_path: Pytest temporary path fixture.
        batched: Whether the recording has an item axis.
        dtype: Sample dtype.
        sample_shape: Per-item or complete sample shape.
        sample_axes: Optional sample axes.
        chunks: Expected chunk shape.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec",
        batched=batched,
        sample_dtype=dtype,
        sample_shape=sample_shape,
        sample_axes=sample_axes,
    )
    assert recording.samples.chunks == chunks


@pytest.mark.parametrize(
    ("batched", "sample_shape", "sample_axes", "shards", "chunks"),
    [
        (True, (2, 128), None, (4096, 2, 128), (1024, 2, 128)),
        (True, (2, 128), None, (64, 2, 128), (64, 2, 128)),
        (True, (2, 3), None, (32, 2, 3), (32, 2, 3)),
        (False, (2, 10), None, (2, 300000), (2, 100000)),
    ],
)
def test_default_chunks_divide_explicit_shards(
    tmp_path: Path,
    batched: bool,
    sample_shape: tuple[int, ...],
    sample_axes: tuple[str, ...] | None,
    shards: tuple[int, ...],
    chunks: tuple[int, ...],
) -> None:
    """Default chunks must tile an explicit shard shape.

    Args:
        tmp_path: Pytest temporary path fixture.
        batched: Whether the recording has an item axis.
        sample_shape: Per-item or complete sample shape.
        sample_axes: Optional sample axes.
        shards: Explicit shard shape.
        chunks: Expected chunk shape.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec",
        batched=batched,
        sample_shape=sample_shape,
        sample_axes=sample_axes,
        sample_shards=shards,
    )
    assert recording.samples.chunks == chunks
    assert recording.samples.shards == shards


@pytest.mark.parametrize(
    "shards",
    [
        [64, 2, 128],
        {"shape": (64, 2, 128), "index_location": "end"},
        {"shape": [64, 2, 128], "index_location": "start"},
    ],
)
def test_default_chunks_divide_every_shard_form(
    tmp_path: Path, shards: ShardsLike
) -> None:
    """The divisor rule applies to list and mapping shard shapes.

    Args:
        tmp_path: Pytest temporary path fixture.
        shards: Explicit shard shape in a non-tuple form.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec", batched=True, sample_shape=(2, 128), sample_shards=shards
    )
    assert recording.samples.chunks == (64, 2, 128)
    assert recording.samples.shards == (64, 2, 128)


def test_explicit_sample_chunks_are_retained(tmp_path: Path) -> None:
    """A caller-supplied chunk shape overrides the default.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec", batched=True, sample_shape=(2, 8), sample_chunks=(16, 2, 8)
    )
    assert recording.samples.chunks == (16, 2, 8)
