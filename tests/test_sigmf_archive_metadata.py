"""Archive conversion preserves metadata and rejects omitted members."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.sigmf import export_sigmf_archive, import_sigmf_archive


def _archive(tmp_path: Path, *, extra: str | None = None) -> Path:
    """Build a recording and collection without dependency normalization.

    Args:
        tmp_path: Temporary directory fixture.
        extra: Optional unsupported archive content.

    Returns:
        Source archive path.
    """
    root = tmp_path / "source"
    root.mkdir()
    meta = root / "rec.sigmf-meta"
    meta.write_text(json.dumps({
        "global": {"core:datatype": "rf32_le", "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0}], "annotations": [],
    }))
    meta.with_suffix(".sigmf-data").write_bytes(
        np.arange(4, dtype="<f4").tobytes()
    )
    stream = {
        "name": "rec", "hash": hashlib.sha512(meta.read_bytes()).hexdigest()
    }
    if extra == "stream":
        stream["test:role"] = "reference"
    collection = root / "paired.sigmf-collection"
    collection.write_text(json.dumps({"collection": {
        "core:version": "1.0.0", "core:streams": [stream],
        "test:nested": {"receivers": [1, 2]},
    }}))
    if extra == "attachment":
        (root / "calibration.txt").write_text("calibration data")
    if extra == "collection":
        (root / "second.sigmf-collection").write_bytes(collection.read_bytes())
    archive = tmp_path / "source.sigmf"
    with tarfile.open(archive, "w") as output:
        output.add(root, arcname="source")
    return archive


def test_collection_preserves_version_and_extension_metadata(
    tmp_path: Path,
) -> None:
    """Retain collection provenance while regenerating metadata-file hashes.

    Args:
        tmp_path: Temporary directory fixture.
    """
    archive = _archive(tmp_path)
    store = import_sigmf_archive(tmp_path / "store.zarr", archive)
    collection = store.collections.open("paired")
    before = dict(collection.metadata)
    assert before["core:version"] == "1.0.0"
    output = export_sigmf_archive(
        store, tmp_path / "output.sigmf", collection_name="paired"
    )
    with tarfile.open(output) as result:
        document = json.load(
            result.extractfile("output/paired.sigmf-collection")
        )
        metadata = result.extractfile("output/rec.sigmf-meta").read()
    info = document["collection"]
    assert info["core:version"] == "1.0.0"
    assert info["test:nested"] == {"receivers": [1, 2]}
    assert info["core:streams"] == [{
        "name": "rec", "hash": hashlib.sha512(metadata).hexdigest()
    }]
    assert collection.metadata == before


@pytest.mark.parametrize("extra", ["attachment", "collection", "stream"])
def test_archive_rejects_unrepresented_content(
    tmp_path: Path, extra: str
) -> None:
    """Reject content loss before creating the destination store.

    Args:
        tmp_path: Temporary directory fixture.
        extra: Unsupported archive content.
    """
    archive = _archive(tmp_path, extra=extra)
    destination = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match="auxiliary|Multiple|core:streams"):
        import_sigmf_archive(destination, archive)
    assert not destination.exists()
