"""Tests for store-wide hashes over child digests and fresh root state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from zarr.core.array import Array

import sigmf_zarr.integrity as integrity
from sigmf_zarr.integrity import INTEGRITY_ATTR, calculate_array_sha512
from sigmf_zarr.json import json_object
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat

RECORDING_SAMPLE_SHA512 = (
    "ee52fef19b8c7cd33ae5f5c38021c0b95a39de9fb3513b4e2a9b75253ef896a0"
    "f0bf119f5089c26cbfc0b1b4ce546013b523ce42db34c4909529a1462ce2795d"
)
"""Sample digest of the example recording."""

RECORDING_METADATA_SHA512 = (
    "8a75313cf34fcce25c9380faec4fae2129c8705a5f3adbab7bc4adc58170dc95"
    "6de1b8f2bc54a3023125ab1b65746d30428d508480c205c15d1142a398879f61"
)
"""Metadata digest of the example recording."""

COLLECTION_METADATA_SHA512 = (
    "12c6c4130d3a9e74e1558cd89378816307325ec0db6fdd3f4a7017f0b5959795"
    "0ef78377f37d23b5845c92fd4ecb24cbe543a998fd9e94904a563ef84bb3f9ea"
)
"""Metadata digest of the example collection."""

STORE_METADATA_SHA512 = (
    "9c3cbd24ccb341d2f5981844c69e2f4b637dfd4c8ba5e9d87ec6c470ac8ff939"
    "76e1fd365f43a29504698beeabb200ae2e208616c1691230f49019b8c87525d6"
)
"""Store-wide digest of the example store."""

STRING_INDEX_SHA512 = (
    "616a5ddf4b97db3d381d52de4ed20e91637babef65e986fbf70480de4627738b"
    "c422b615bdd7fa7f70ce399a08f959efad9c2cc73d65c242f74295415cc198a7"
)
"""Digest of the stored string index `["a", "bc", "\u00e9"]`."""

ITEM_METADATA_SHA512 = (
    "f3f40e836ba5a69b49674cd2235b108934403b70de2ed6744768250b77bacce8"
    "087d8955105eaf035d8ab540e8a98ed035eb0dd20bd263954d20a79dd119a971"
)
"""Digest of the stored `item_metadata` of the string example."""

STRING_RECORDING_METADATA_SHA512 = (
    "beb330c61a0625f45b58c1b998ee3671f5362b576fb3b6c56ac8b118b9fd3978"
    "b5c98c89a4bac610937f405235f8bd998c9154c5f58beaefeaafcadeb97f500a"
)
"""Metadata digest of the string example recording."""

STRING_STORE_METADATA_SHA512 = (
    "ce293f8570c7f64206e3346664c16e7909b70b84f0a6d22b1abc154c9c3fa719"
    "b7fc6b6874f89769e3e78dd7256cfb817bc1d41eb938caab3b7aa2da1b1d3e7f"
)
"""Store-wide digest of the string example store."""


def canonical_json(value: object) -> bytes:
    """Encode JSON by the documented canonical rule without the library.

    Args:
        value: JSON-compatible value.

    Returns:
        Canonical JSON bytes.
    """
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def string_array_sha512(values: list[str]) -> str:
    """Hash a one-dimensional string array from the documented rule.

    Args:
        values: Array values in order.

    Returns:
        Lowercase SHA-512 hexadecimal digest.
    """
    slab = hashlib.sha512()
    for value in values:
        encoded = canonical_json(value)
        slab.update(len(encoded).to_bytes(8, "big") + encoded)
    descriptor = {
        "dtype": "string",
        "shape": [len(values)],
        "slab_axis": 0,
        "slab_length": 2**24 // 16,
        "version": 1,
    }
    return hashlib.sha512(
        b"sigmf-zarr-array-v1\0"
        + canonical_json(descriptor)
        + b"\0"
        + slab.digest()
    ).hexdigest()


def example_store(path: Path, *, zarr_format: int = 3) -> SigMFZarrStore:
    """Create a small store with a recording, collection, and store index.

    Args:
        path: Store path.
        zarr_format: Physical Zarr format.

    Returns:
        Store without integrity hashes.
    """
    store = SigMFZarrStore.create(
        path, zarr_format=cast(ZarrFormat, zarr_format)
    )
    recording = store.recordings.create(
        "rec",
        sample_shape=(4,),
        global_metadata={"core:datatype": "rf32_le"},
    )
    recording.set_samples(np.arange(4, dtype=np.float32))
    store.collections.create(
        "group", recording_ids=["rec"], metadata={"test:name": "pair"}
    )
    store.add_index("ids", np.arange(3, dtype=np.int16))
    return store


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_store_hashes_match_known_answers(
    tmp_path: Path, zarr_format: int
) -> None:
    """Every native hash must match fixed vectors in both formats.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format.
    """
    store = example_store(tmp_path / "store.zarr", zarr_format=zarr_format)
    store.update_integrity()
    recording = store.recordings["rec"]
    assert recording.sample_sha512 == RECORDING_SAMPLE_SHA512
    assert recording.sample_sha512 == calculate_array_sha512(
        np.arange(4, dtype="<f4")
    )
    assert recording.metadata_sha512 == RECORDING_METADATA_SHA512
    assert (
        store.collections["group"].metadata_sha512
        == COLLECTION_METADATA_SHA512
    )
    assert store.metadata_sha512 == STORE_METADATA_SHA512


def string_store(path: Path, *, zarr_format: int) -> SigMFZarrStore:
    """Create a store with a string index and per-item metadata.

    Args:
        path: Store path.
        zarr_format: Physical Zarr format.

    Returns:
        Store without integrity hashes.
    """
    store = SigMFZarrStore.create(
        path, zarr_format=cast(ZarrFormat, zarr_format)
    )
    recording = store.recordings.create(
        "batch",
        batched=True,
        sample_shape=(4,),
        global_metadata={"core:datatype": "rf32_le"},
    )
    recording.append_samples(np.arange(8, dtype=np.float32).reshape(2, 4))
    recording.set_item_metadata([{"global": {"test:name": "\u00e9"}}, None])
    store.add_index("names", np.array(["a", "bc", "\u00e9"], dtype=object))
    return store


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_stored_string_hashes_match_known_answers(
    tmp_path: Path, zarr_format: int
) -> None:
    """Stored strings must hash by the documented rule in both formats.

    Zarr reads these arrays as NumPy `StringDType`, whose dtype string
    differs between NumPy releases, so the vectors must not depend on it.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format.
    """
    store = string_store(tmp_path / "store.zarr", zarr_format=zarr_format)
    store.update_integrity()
    names = store.index("names")
    assert names.dtype.kind == "T"
    assert calculate_array_sha512(names) == STRING_INDEX_SHA512
    assert string_array_sha512(["a", "bc", "\u00e9"]) == STRING_INDEX_SHA512
    recording = store.recordings["batch"]
    item_metadata = recording.item_metadata_array
    assert item_metadata.dtype.kind == "T"
    assert calculate_array_sha512(item_metadata) == ITEM_METADATA_SHA512
    assert (
        string_array_sha512(['{"global":{"test:name":"\\u00e9"}}', "{}"])
        == ITEM_METADATA_SHA512
    )
    assert recording.metadata_sha512 == STRING_RECORDING_METADATA_SHA512
    assert store.metadata_sha512 == STRING_STORE_METADATA_SHA512


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_nested_integrity_attributes_are_hashed(
    tmp_path: Path, zarr_format: int
) -> None:
    """Only the hashed group's own `integrity` attribute is excluded.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=cast(ZarrFormat, zarr_format)
    )
    recording = store.recordings.create(
        "rec", sample_shape=(2, 4), sample_axes=("channel", "time")
    )
    recording.set_channel_metadata(0, {"integrity": "calibrated"})
    store.update_integrity()
    assert store.verify_integrity()
    # Another tool can change the attribute without invalidating hashes.
    recording._channel_group(0).attrs[INTEGRITY_ATTR] = "tampered"
    assert not recording.verify_integrity()
    assert not store.verify_integrity()

    store.update_integrity()
    store._recordings_group.attrs[INTEGRITY_ATTR] = "container"
    assert not store.verify_metadata_integrity()


def test_collection_hash_matches_documented_payload(tmp_path: Path) -> None:
    """A nested `integrity` attribute must hash as ordinary content.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = example_store(tmp_path / "store.zarr")
    collection = store.collections["group"]
    group = collection._raw_group
    group.create_group("extra").attrs[INTEGRITY_ATTR] = {"note": "kept"}
    collection.update_integrity()
    payload = {
        "arrays": {},
        "attrs": {"metadata": {"test:name": "pair"}, "recording_ids": ["rec"]},
        "groups": {
            "extra": {
                "arrays": {},
                "attrs": {INTEGRITY_ATTR: {"note": "kept"}},
                "groups": {},
            }
        },
    }
    expected = hashlib.sha512(
        b"sigmf-zarr-metadata-v1\0" + canonical_json(payload)
    ).hexdigest()
    assert collection.metadata_sha512 == expected


def test_store_hash_requires_current_child_digests(tmp_path: Path) -> None:
    """The store-wide hash cannot bind digests that a child lacks.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = example_store(tmp_path / "store.zarr")
    with pytest.raises(ValueError, match="Recording 'rec'"):
        store.update_metadata_integrity()
    store.recordings["rec"].update_integrity()
    with pytest.raises(ValueError, match="Collection 'group'"):
        store.update_metadata_integrity()
    store.collections["group"].update_integrity()
    store.update_metadata_integrity()
    assert store.verify_integrity()


def test_store_hash_rejects_recordings_with_invalid_indexes(
    tmp_path: Path,
) -> None:
    """An invalid recording index removes the digest the store would bind.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "batch", batched=True, sample_shape=(4,)
    )
    recording.append_samples(np.ones((3, 4), dtype=np.float32))
    recording.add_index(
        "snr", np.zeros(3, dtype=np.int16), axis="item", field="test:snr"
    )
    store.update_integrity()
    recording.append_samples(np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="no current integrity"):
        store.update_metadata_integrity()
    with pytest.raises(ValueError, match="invalid"):
        store.update_integrity()


def test_store_hash_does_not_reread_child_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store-wide hashing reads only store-wide arrays.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = example_store(tmp_path / "store.zarr")
    store.update_integrity()
    hashed: list[str] = []
    original = integrity.calculate_array_sha512

    def record(
        array: Array[Any] | ReadOnlyArray | npt.NDArray[Any], **kwargs: int
    ) -> str:
        """Record each hashed array path.

        Args:
            array: Array being hashed.
            **kwargs: Hash options.

        Returns:
            Original digest.
        """
        hashed.append(str(getattr(array, "path", "")))
        return original(array, **kwargs)

    monkeypatch.setattr(integrity, "calculate_array_sha512", record)
    assert store.verify_metadata_integrity()
    assert hashed == ["indexes/ids"]


def test_tampered_child_digest_fails_store_verification(
    tmp_path: Path,
) -> None:
    """The store-wide hash binds the digests each child declares.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = example_store(tmp_path / "store.zarr")
    store.update_integrity()
    group = store._recordings_group["rec"]
    declared = json_object(group.attrs[INTEGRITY_ATTR], name="integrity")
    declared["metadata_sha512"] = "0" * 128
    group.attrs[INTEGRITY_ATTR] = declared
    assert not store.verify_metadata_integrity()
    assert not store.verify_integrity()


def test_mutation_through_stale_handle_removes_store_hash(
    tmp_path: Path,
) -> None:
    """Invalidation must observe a hash written through another handle.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "store.zarr"
    first = SigMFZarrStore.create(path)
    SigMFZarrStore.open(path, mode="r+").update_integrity()
    assert first.metadata_sha512 is not None
    first.add_index("ids", np.arange(4))
    assert SigMFZarrStore.open(path).metadata_sha512 is None


def test_store_hash_updates_only_when_complete(tmp_path: Path) -> None:
    """Conditional updates leave the store hash absent when incomplete.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = example_store(tmp_path / "store.zarr")
    store.update_integrity()
    store.recordings.create("unhashed", sample_shape=(2,))
    store._update_metadata_integrity_when_complete()
    assert store.metadata_sha512 is None
    store.update_integrity()
    # An exception inside the mutation leaves the store index invalid.
    with pytest.raises(RuntimeError), store.mutate_index("ids"):
        raise RuntimeError
    with pytest.raises(ValueError, match="invalid"):
        store.update_metadata_integrity()
    store._update_metadata_integrity_when_complete()
    assert store.metadata_sha512 is None
