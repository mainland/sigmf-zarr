"""Measure dense adapter startup and scalar versus batched item reads.

Run from the repository root with development and PyTorch extras installed:
    .venv/bin/python -m benchmarks.pytorch_loading
"""

from __future__ import annotations

import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.pytorch import RecordingDataset
from sigmf_zarr.store import ZarrFormat
from tests.http_store import serve_directory


def measure(path: str, backend: str, zarr_format: ZarrFormat) -> None:
    """Measure one warm-cache workload without transforms or collation.

    Args:
        path: Local path or HTTP URL.
        backend: Printed backend label.
        zarr_format: Physical Zarr format.
    """
    tracemalloc.start()
    started = time.perf_counter()
    dataset = RecordingDataset(
        path,
        recording="rec",
        targets={"class": "class", "snr": "snr"},
        split=None,
    )
    startup = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dataset[0]
    rng = np.random.default_rng(42)
    for order, positions in [
        ("sequential", list(range(256))),
        ("random", rng.integers(0, len(dataset), 256).tolist()),
    ]:
        timings: dict[str, list[float]] = {"scalar": [], "batch": []}
        for _ in range(3):
            started = time.perf_counter()
            scalar = [dataset[position] for position in positions]
            timings["scalar"].append(time.perf_counter() - started)
            started = time.perf_counter()
            batch = dataset.__getitems__(positions)
            timings["batch"].append(time.perf_counter() - started)
            assert [item["item_index"] for item in scalar] == [
                item["item_index"] for item in batch
            ]
        scalar_time = statistics.median(timings["scalar"])
        batch_time = statistics.median(timings["batch"])
        print(
            f"{backend}, {zarr_format}, {order}, {startup:.4f}, "
            f"{peak / 1024:.0f}, {scalar_time:.4f}, {batch_time:.4f}, "
            f"{scalar_time / batch_time:.1f}",
            flush=True,
        )
    dataset.close()


def main() -> None:
    """Compare local and loopback HTTP reads using reproducible RFML shapes."""
    print(
        "backend, format, order, startup_s, python_peak_KiB, "
        "scalar_s, batch_s, speedup"
    )
    with tempfile.TemporaryDirectory(
        prefix="sigmf-pytorch-benchmark-"
    ) as directory:
        root = Path(directory)
        for zarr_format in (2, 3):
            path = root / f"format{zarr_format}.zarr"
            store = SigMFZarrStore.create(path, zarr_format=zarr_format)
            recording = store.recordings.open(
                "rec",
                batched=True,
                sample_shape=(2, 128),
                sample_axes=("iq", "time"),
                sample_chunks=(256, 2, 128),
                sample_shards=(4096, 2, 128) if zarr_format == 3 else None,
            )
            recording.append_samples(
                np.random.default_rng(42)
                .normal(size=(8192, 2, 128))
                .astype(np.float32)
            )
            recording.add_index(
                "class",
                np.arange(8192, dtype=np.uint8) % 2,
                axis="item",
                field="example:class",
                labels=["A", "B"],
            )
            recording.add_index(
                "snr",
                np.zeros(8192, dtype=np.int16),
                axis="item",
                field="example:snr",
            )
            recording.set_item_metadata(
                [{"global": {"example:class": "JSON"}}] * 8192
            )
            measure(str(path), "local", zarr_format)
            with serve_directory(root) as url:
                measure(f"{url}/{path.name}", "http", zarr_format)


if __name__ == "__main__":
    main()
