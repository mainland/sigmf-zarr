"""Remote dense reads and serializable worker backend configuration."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.store import ZarrFormat

torch = pytest.importorskip("torch")
pytest.importorskip("aiohttp")
from torch.utils.data import DataLoader  # noqa: E402

from sigmf_zarr.pytorch import RecordingDataset  # noqa: E402
from tests.http_store import serve_directory  # noqa: E402


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "context",
    [
        method
        for method in ("spawn", "forkserver", "fork")
        if method in mp.get_all_start_methods()
    ],
)
def test_http_worker_reads(
    tmp_path: Path, zarr_format: int, context: str
) -> None:
    """Read remote chunks and shards, or reject fork before backend access.

    Args:
        tmp_path: Temporary HTTP server root.
        zarr_format: Physical Zarr format.
        context: Worker start method.
    """
    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=cast(ZarrFormat, zarr_format)
    )
    recording = store.recordings.open(
        "rec",
        batched=True,
        sample_shape=(4,),
        sample_axes=("time",),
        sample_chunks=(2, 4),
        sample_shards=(4, 4) if zarr_format == 3 else None,
    )
    values = np.arange(32, dtype=np.float32).reshape(8, 4)
    recording.append_samples(values)
    recording.add_index(
        "snr", np.arange(8, dtype=np.int16), axis="item", field="example:snr"
    )
    with serve_directory(tmp_path) as url:
        dataset = RecordingDataset(
            f"{url}/data.zarr",
            recording="rec",
            targets={"snr": "snr"},
            split=None,
        )
        assert torch.equal(dataset[0]["samples"], torch.from_numpy(values[0]))
        items = dataset.__getitems__([7, 7, 2])
        assert [item["targets"]["snr"].item() for item in items] == [7, 7, 2]
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=1,
            multiprocessing_context=context,
            timeout=30,
        )
        if context == "fork":
            with pytest.raises(RuntimeError, match="Remote stores require"):
                next(iter(loader))
        else:
            batches = list(loader)
            assert torch.equal(
                torch.cat([batch["samples"] for batch in batches]),
                torch.from_numpy(values),
            )
        dataset.close()
