"""A selected batch item becomes an independent standard SigMF recording."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.sigmf import export_sigmf, import_sigmf
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat

pytestmark = pytest.mark.filterwarnings(
    "ignore:Data source ends before the final annotation:UserWarning"
)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("axes", [("iq", "time"), ("time", "iq")])
def test_export_item_preserves_independent_captures(
    tmp_path: Path, zarr_format: ZarrFormat, axes: tuple[str, ...]
) -> None:
    """Resolve scopes, offsets, timestamps, and bytes without changing source.

    Args:
        tmp_path: Temporary directory fixture.
        zarr_format: Physical Zarr format.
        axes: Per-item sample axis order.
    """
    shape = (2, 4) if axes[0] == "iq" else (4, 2)
    samples = np.arange(16, dtype=np.float32).reshape(2, *shape)
    store_path = tmp_path / "source.zarr"
    with SigMFZarrStore.create(store_path, zarr_format=zarr_format) as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=shape, sample_axes=axes,
            sample_chunks=(1, 2, 2),
            global_metadata={
                "core:datatype": "cf32_le", "core:version": "1.0.0",
                "core:offset": 100,
            },
        )
        recording.append_samples(
            samples,
            capture={
                "core:frequency": 100, "core:datetime": "2025-01-01T00:00:00Z"
            },
            item_metadata=[None, {
                "global": {"core:offset": 200},
                "annotations": [
                    {"core:sample_start": 201, "core:sample_count": 2}
                ],
            }],
            item_captures=[
                [{"core:datetime": "2026-01-01T00:00:00.000000001Z"}],
                [
                    {"core:datetime": "2026-01-01T00:01:00.000000002Z"},
                    {"core:sample_start": 202, "core:frequency": 300},
                ],
            ],
        )
        recording.set_global_field("core:sha512", "0" * 128)
        store.update_integrity()
        before = recording.metadata()
        item_before = recording.get_item_metadata(1)
        for item in range(2):
            output = tmp_path / f"item-{item}.sigmf-meta"
            export_sigmf(store, "rec", output, item_index=item)
            metadata = json.loads(output.read_text())
            captures = metadata["captures"]
            assert captures[0]["core:sample_start"] == 100 * (item + 1)
            assert captures[0]["core:frequency"] == 100
            assert captures[0]["core:datetime"] == (
                "2026-01-01T00:00:00.000000001Z" if item == 0
                else "2026-01-01T00:01:00.000000002Z"
            )
            if item:
                assert captures[1] == {
                    "core:sample_start": 202, "core:frequency": 300,
                }
            expected = samples[item].T if axes[0] == "iq" else samples[item]
            data = output.with_suffix(".sigmf-data").read_bytes()
            assert data == np.ascontiguousarray(expected).tobytes()
            assert metadata["global"]["core:sha512"] == (
                hashlib.sha512(data).hexdigest()
            )
            assert metadata["global"]["core:version"] == "1.0.0"
            reimported = import_sigmf(
                tmp_path / f"roundtrip-{item}.zarr", output
            )
            assert reimported.captures == captures
        assert recording.metadata() == before
        assert recording.get_item_metadata(1) == item_before
        assert store.verify_integrity()


@pytest.mark.parametrize("item", [0, 1])
def test_shared_datetime_applies_only_to_declared_item(
    tmp_path: Path, item: int
) -> None:
    """Never invent a timestamp for a later independent item.

    Args:
        tmp_path: Temporary directory fixture.
        item: Item selected for export.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.append_samples(
            np.ones((2, 4)), capture={
                "core:datetime": "2026-01-01T00:00:00Z", "core:frequency": 100
            },
        )
        output = export_sigmf(store, "rec", tmp_path / "out", item_index=item)
        capture = json.loads(output.read_text())["captures"][0]
        assert ("core:datetime" in capture) == (item == 0)
        assert capture["core:sample_start"] == 0


@pytest.mark.parametrize("item", [-1, 1, True, 0.5])
def test_invalid_item_selection_leaves_no_output(
    tmp_path: Path, item: object
) -> None:
    """Reject invalid item positions before creating output files.

    Args:
        tmp_path: Temporary directory fixture.
        item: Invalid selector.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(4,), sample_axes=("time",)
        )
        recording.append_samples(np.ones((1, 4)))
        with pytest.raises(ValueError, match="item_index"):
            export_sigmf(store, "rec", tmp_path / "out", item_index=item)
        assert not (tmp_path / "out.sigmf-meta").exists()
        assert not (tmp_path / "out.sigmf-data").exists()


def test_export_does_not_broadcast_source_anchors(tmp_path: Path) -> None:
    """Source positions apply only where a capture establishes them.

    Args:
        tmp_path: Temporary directory.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.append_samples(
            np.ones((2, 4)), capture={"core:global_index": 1000},
            item_captures=[None, [{"core:sample_start": 2}]],
        )
        output = export_sigmf(store, "rec", tmp_path / "out", item_index=1)
        assert json.loads(output.read_text())["captures"] == [
            {"core:sample_start": 0}, {"core:sample_start": 2},
        ]
