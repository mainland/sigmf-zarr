"""Input identities distinguish stale results from structurally valid
indexes.
"""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.integrity import calculate_array_sha512
from sigmf_zarr.provenance import (
    capture_inputs,
    file_identity,
    file_manifest_identity,
    import_metadata,
    verify_inputs,
)


@pytest.mark.parametrize(
    "changed", ["samples", "metadata", "values", "labels"]
)
def test_input_binding_detects_changes(tmp_path: Path, changed: str) -> None:
    """Detect input edits while allowing unrelated derived outputs.

    Args:
        tmp_path: Temporary store directory.
        changed: Bound input to mutate.
    """
    store = SigMFZarrStore.create(tmp_path / "data.zarr")
    rec = store.recordings.open(
        "rec", batched=True, sample_shape=(4,), sample_axes=("time",)
    )
    rec.append_samples(np.zeros((2, 4), dtype=np.float32))
    rec.add_index(
        "class", [0, 1], axis="item", field="test:class", labels=["A", "B"]
    )
    binding = capture_inputs(rec, indexes=["class"])
    rec.add_index(
        "derived",
        [1, 1],
        axis="item",
        field="test:derived",
        attributes={"inputs": binding},
    )
    assert verify_inputs(rec, binding)
    if changed == "samples":
        with rec.mutate_samples() as samples:
            samples[0, 0] = 1
    elif changed == "metadata":
        rec.set_global_field("test:source", "changed")
    else:
        with rec.mutate_index("class") as index:
            if changed == "values":
                index[0] = 1
            else:
                index.attrs["labels"] = ["B", "A"]
    assert not verify_inputs(rec, binding)


def test_source_manifest_identity(tmp_path: Path) -> None:
    """Include content, relative names, and truth roles in source identities.

    Args:
        tmp_path: Temporary input directory.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"samples")
    b.write_bytes(b"truth")
    identity = file_manifest_identity([b, a], root=tmp_path, role="samples")
    assert identity == file_manifest_identity(
        [a, b, a], root=tmp_path, role="samples"
    )
    metadata = import_metadata(
        {},
        sources=[file_identity(b, role="truth")],
        operation="test",
        parameters={},
    )
    assert metadata["core:extensions"] == [
        {"name": "sigmf-zarr-provenance", "version": "0.1.0", "optional": True}
    ]
    b.write_bytes(b"changed truth")
    assert identity != file_manifest_identity(
        [a, b], root=tmp_path, role="samples"
    )
    with pytest.raises(ValueError, match="generated"):
        import_metadata(metadata, sources=[], operation="test", parameters={})
    with pytest.raises(ValueError, match="Conflicting"):
        import_metadata(
            {"core:extensions": [
                {"name": "native", "version": "0.1.0", "optional": True},
                {"name": "native", "version": "0.2.0", "optional": True},
            ]},
            sources=[], operation="test", parameters={},
            source_namespace="native",
        )


def test_logical_array_hash_is_portable() -> None:
    """Hash the same numeric values independently of source byte order."""
    little = np.arange(8, dtype="<f4").reshape(2, 4)
    assert calculate_array_sha512(little) == calculate_array_sha512(
        little.astype(">f4")
    )
