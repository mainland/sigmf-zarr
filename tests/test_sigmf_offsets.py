"""Absolute sample coordinates survive standard SigMF interchange."""

import json
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.sigmf import export_sigmf, import_sigmf
from sigmf_zarr.store import SigMFZarrStore


@pytest.mark.parametrize(("start", "count"), [(1000, 4), (1003, 1), (1004, 0)])
def test_export_absolute_sample_coordinates(
    tmp_path: Path, start: int, count: int
) -> None:
    """Export captures and annotations in the source coordinate system.

    Args:
        tmp_path: Temporary directory fixture.
        start: Absolute annotation start.
        count: Annotation length.
    """
    source = tmp_path / "source.sigmf-meta"
    metadata = {
        "global": {
            "core:datatype": "rf32_le",
            "core:version": "1.2.6",
            "core:offset": 1000,
        },
        "captures": [{"core:sample_start": 1000, "core:sample_count": 4}],
        "annotations": [
            {"core:sample_start": start, "core:sample_count": count}
        ],
    }
    source.write_text(json.dumps(metadata))
    data = np.arange(4, dtype="<f4").tobytes()
    source.with_suffix(".sigmf-data").write_bytes(data)
    # sigmf-python compares annotation starts with the data length without
    # subtracting core:offset, so it warns on both import and export.
    with pytest.warns(UserWarning, match="ends before the final annotation"):
        import_sigmf(tmp_path / "store.zarr", source)
    with pytest.warns(UserWarning, match="ends before the final annotation"):
        output = export_sigmf(
            SigMFZarrStore.open(tmp_path / "store.zarr"),
            "source",
            tmp_path / "output.sigmf-meta",
        )
    result = json.loads(output.read_text())
    assert result["captures"] == metadata["captures"]
    assert result["annotations"] == metadata["annotations"]
    assert output.with_suffix(".sigmf-data").read_bytes() == data


@pytest.mark.parametrize(("start", "count"), [(999, 1), (1004, 1), (1005, 0)])
def test_export_rejects_absolute_spans_outside_dataset(
    tmp_path: Path, start: int, count: int
) -> None:
    """Reject spans outside both bounds before publishing output.

    Args:
        tmp_path: Temporary directory fixture.
        start: Invalid absolute annotation start.
        count: Annotation length.
    """
    with SigMFZarrStore.create(tmp_path / "store.zarr") as store:
        store.recordings.create(
            "rec",
            sample_shape=(4,),
            sample_axes=("time",),
            global_metadata={"core:datatype": "rf32_le", "core:offset": 1000},
            annotations=[
                {"core:sample_start": start, "core:sample_count": count}
            ],
        )
        target = tmp_path / "output.sigmf-meta"
        with pytest.raises(ValueError, match="annotation 0"):
            export_sigmf(store, "rec", target)
        assert not target.exists()
        assert not target.with_suffix(".sigmf-data").exists()


def test_export_rejects_schema_violations_before_writing(
    tmp_path: Path,
) -> None:
    """Schema violations must raise ValueError before any output exists.

    Args:
        tmp_path: Temporary directory fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    store.recordings.create(
        "rec",
        sample_shape=(100,),
        global_metadata={"core:datatype": "rf32_le", "core:version": "1.2.0"},
        annotations=[
            {"core:sample_start": 50, "core:sample_count": 1},
            {"core:sample_start": 10, "core:sample_count": 1},
        ],
    )
    with pytest.raises(ValueError, match="SigMF schema"):
        export_sigmf(store, "rec", tmp_path / "out")
    assert not list(tmp_path.glob("out.*"))
