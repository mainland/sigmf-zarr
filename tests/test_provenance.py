"""Input identities detect stale results.

Structural index validation alone does not detect stale results.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore, ZarrFormat
from sigmf_zarr.integrity import calculate_array_sha512
from sigmf_zarr.json import JSONObject
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
    rec = store.recordings.create(
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
    )
    with rec.mutate_index("derived") as derived:
        derived.attrs["inputs"] = binding
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


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_input_binding_rejects_invalid_index(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Return false, not raise, when an append invalidates a bound index.

    Args:
        tmp_path: Temporary store directory.
        zarr_format: Physical Zarr format.
    """
    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=zarr_format
    )
    rec = store.recordings.create(
        "rec", batched=True, sample_shape=(2, 4), sample_axes=("iq", "time")
    )
    rec.append_samples(np.zeros((4, 2, 4), dtype=np.float32))
    rec.add_index("snr", [1, 2, 3, 4], axis="item", field="test:snr")
    binding = capture_inputs(rec, indexes=["snr"])
    assert verify_inputs(rec, binding)
    rec.append_samples(np.ones((1, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="invalid"):
        rec.index("snr")
    assert not verify_inputs(rec, binding)
    with pytest.raises(ValueError, match="version"):
        verify_inputs(rec, {**binding, "version": 2})
    with pytest.raises(ValueError, match="indexes"):
        verify_inputs(rec, {**binding, "indexes": []})


def test_verify_inputs_rejects_malformed_bindings(tmp_path: Path) -> None:
    """Bindings that capture_inputs() cannot return must raise ValueError.

    Args:
        tmp_path: Temporary store directory.
    """
    store = SigMFZarrStore.create(tmp_path / "data.zarr")
    rec = store.recordings.create(
        "rec", batched=True, sample_shape=(4,), sample_axes=("time",)
    )
    rec.append_samples(np.zeros((2, 4), dtype=np.float32))
    rec.add_index("class", [0, 1], axis="item", field="test:c")
    binding = capture_inputs(rec, indexes=["class"])
    entry = cast(dict[str, Any], binding["indexes"])["class"]
    for malformed, message in [
        ({"x": 1}, "version"),
        ({**binding, "extra": 1}, "exactly the fields"),
        ({**binding, "sample_sha512": "0" * 127}, "sample_sha512"),
        ({**binding, "source_metadata_sha512": "A" * 128}, "source_metadata"),
        ({**binding, "indexes": {"class": {"sha512": "0" * 128}}}, "class"),
        (
            {**binding, "indexes": {"class": {**entry, "sha512": 1}}},
            "class",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            verify_inputs(rec, cast(JSONObject, malformed))


def test_manifest_normalizes_paths(tmp_path: Path) -> None:
    """Dot components must not duplicate entries or escape the root.

    Args:
        tmp_path: Temporary input directory.
    """
    (tmp_path / "sub").mkdir()
    a = tmp_path / "a.bin"
    a.write_bytes(b"samples")
    identity = file_manifest_identity([a], root=tmp_path, role="samples")
    assert identity["count"] == 1
    assert (
        file_manifest_identity(
            [a, tmp_path / "sub" / ".." / "a.bin", tmp_path / "." / "a.bin"],
            root=tmp_path / "sub" / "..",
            role="samples",
        )
        == identity
    )
    sub = tmp_path / "sub"
    for path in [sub / ".." / "a.bin", sub / ".." / "sub" / ".." / "a.bin"]:
        with pytest.raises(ValueError, match="not below root"):
            file_manifest_identity([path], root=sub, role="x")
    with pytest.raises(ValueError, match="names the root"):
        file_manifest_identity([sub / "." / "x" / ".."], root=sub, role="x")


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
            {
                "core:extensions": [
                    {"name": "native", "version": "0.1.0", "optional": True},
                    {"name": "native", "version": "0.2.0", "optional": True},
                ]
            },
            sources=[],
            operation="test",
            parameters={},
            source_namespace="native",
        )


def test_logical_array_hash_is_portable() -> None:
    """Hash the same numeric values independently of source byte order."""
    little = np.arange(8, dtype="<f4").reshape(2, 4)
    assert calculate_array_sha512(little) == calculate_array_sha512(
        little.astype(">f4")
    )
