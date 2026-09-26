"""Tests for SigMF-Zarr-native logical integrity hashes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr.core.array import Array
from zarr.core.group import Group

from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.store import SigMFZarrStore


def _create_hashed_store(tmp_path, *, zarr_format: int) -> SigMFZarrStore:
    """Create one small store with all internal hashes populated.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.

    Returns:
        Created and hashed store.
    """
    store = SigMFZarrStore.create(
        tmp_path / f"store-v{zarr_format}.zarr",
        overwrite=True,
        zarr_format=zarr_format,
    )
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(2, 4),
        sample_axes=("iq", "time"),
        global_metadata={"core:datatype": "cf32_le"},
    )
    recording.set_samples(np.arange(8, dtype=np.float32).reshape(2, 4))
    recording.add_index(
        "quality",
        np.arange(4, dtype=np.int16),
        axis="time",
        field="test:quality",
    )
    store.collections.open(
        "all",
        create=True,
        metadata={"core:description": "all recordings"},
        recording_ids=("rec",),
    )
    store.update_integrity()
    return store


def test_internal_hashes_are_independent_of_zarr_format(tmp_path) -> None:
    """Logical hashes should match across Zarr formats 2 and 3.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    stores = [
        _create_hashed_store(tmp_path, zarr_format=zarr_format)
        for zarr_format in (2, 3)
    ]
    recordings = [store.recordings.open("rec") for store in stores]
    collections = [store.collections.open("all") for store in stores]

    assert recordings[0].sample_sha512 == recordings[1].sample_sha512
    assert recordings[0].metadata_sha512 == recordings[1].metadata_sha512
    assert collections[0].metadata_sha512 == collections[1].metadata_sha512
    assert stores[0].metadata_sha512 == stores[1].metadata_sha512
    assert all(store.verify_integrity() for store in stores)


def test_managed_mutations_invalidate_affected_internal_hashes(
    tmp_path,
) -> None:
    """Managed mutations should never leave affected hashes looking current.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    recording = store.recordings.open("rec")

    recording.set_global_field("core:description", "changed")

    assert recording.sample_sha512 is not None
    assert recording.metadata_sha512 is None
    assert store.metadata_sha512 is None

    store.update_integrity()
    recording.append_samples(np.ones((2, 1), dtype=np.float32))

    assert recording.sample_sha512 is None
    assert recording.metadata_sha512 is None
    assert store.metadata_sha512 is None


def test_samples_are_writable_only_in_mutation_context(tmp_path) -> None:
    """Public sample views should reject writes outside their context.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    recording = store.recordings.open("rec")

    try:
        recording.samples[0, 0] = np.float32(99.0)
    except TypeError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("Expected direct sample mutation to fail")

    try:
        recording.group["samples"][0, 0] = np.float32(99.0)
    except TypeError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("Expected group traversal to remain read-only")

    assert recording.verify_integrity() is True

    with recording.mutate_samples() as samples:
        samples[0, 0] = np.float32(99.0)

    assert recording.sample_sha512 is None
    assert recording.metadata_sha512 is None
    assert store.verify_integrity() is False

    store.update_integrity()
    try:
        recording.group.attrs["global"] = {}
    except TypeError:
        pass
    else:
        raise AssertionError("Expected recording attrs to be read-only")

    recording.set_global_field("core:description", "managed change")

    assert recording.metadata_sha512 is None
    assert store.verify_integrity() is False


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_read_only_views_wrap_configuration_and_group_traversal(
    tmp_path: Path, zarr_format: int
) -> None:
    """Read helpers must not expose writable descendant handles.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store = _create_hashed_store(tmp_path, zarr_format=zarr_format)
    recording = store.recordings.open("rec")
    arrays = [
        recording.samples.with_config({"order": "F"}),
        *recording.group.array_values(),
    ]
    groups = list(recording.group.group_values())
    for group in groups:
        assert isinstance(group, ReadOnlyGroup)
        assert group.read_only is True
        arrays.extend(group.array_values())
    for _, member in recording.group.members(max_depth=None):
        assert isinstance(member, ReadOnlyArray | ReadOnlyGroup)
        if isinstance(member, ReadOnlyArray):
            arrays.append(member)
    for array in arrays:
        assert isinstance(array, ReadOnlyArray)
        assert array.read_only is True
        with pytest.raises(TypeError, match="read-only"):
            array[...] = 99
        with pytest.raises(AttributeError):
            _ = array.store

    np.testing.assert_array_equal(
        recording.samples.with_config({"order": "F"})[:],
        np.arange(8, dtype=np.float32).reshape(2, 4),
    )
    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_read_only_views_reject_unreviewed_zarr_members(
    tmp_path: Path,
    zarr_format: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New Zarr members and convenience creators must remain inaccessible.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = _create_hashed_store(tmp_path, zarr_format=zarr_format)
    recording = store.recordings.open("rec")
    monkeypatch.setattr(Array, "future_writer", "unsafe", raising=False)
    monkeypatch.setattr(Group, "future_writer", "unsafe", raising=False)
    for name in (
        "empty", "empty_like", "full", "full_like", "ones", "ones_like",
        "zeros", "zeros_like", "create_array", "require_array", "store",
        "future_writer",
    ):
        with pytest.raises(AttributeError):
            getattr(recording.group, name)
    with pytest.raises(AttributeError):
        _ = recording.samples.future_writer

    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_read_only_metadata_values_are_detached(
    tmp_path: Path, zarr_format: int
) -> None:
    """Editing a returned metadata object must not alter live metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store = _create_hashed_store(tmp_path, zarr_format=zarr_format)
    recording = store.recordings.open("rec")
    with recording.mutate_samples() as array:
        array.attrs["test:nested"] = {"values": [1, 2]}
    store.update_integrity()

    recording.group.attrs["global"]["core:datatype"] = "ri16_le"
    recording.group.metadata.attributes["global"]["core:datatype"] = "ri16_le"
    recording.samples.attrs["test:nested"]["values"][0] = 100
    recording.samples.metadata.attributes["test:nested"]["values"][0] = 100

    assert recording.global_metadata["core:datatype"] == "cf32_le"
    assert recording.samples.attrs["test:nested"] == {"values": [1, 2]}
    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_nested_recording_indexes_require_valid_alignment(
    tmp_path: Path, zarr_format: int
) -> None:
    """Nested indexes must participate in invalidation and integrity checks.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store = _create_hashed_store(tmp_path, zarr_format=zarr_format)
    recording = store.recordings.open("rec")
    recording.add_index(
        "nested/quality",
        np.arange(4),
        axis="time",
        field="test:quality",
    )
    assert "nested/quality" in recording.info()
    store.update_integrity()
    with (
        pytest.raises(RuntimeError, match="stop"),
        recording.mutate_index("nested/quality"),
    ):
        raise RuntimeError("stop")
    assert recording.verify_integrity() is False
    with pytest.raises(ValueError, match="nested/quality.*did not complete"):
        recording.update_integrity()
    with recording.mutate_index("nested/quality"):
        pass
    store.update_integrity()

    recording.append_samples(np.ones((2, 1), dtype=np.float32))
    assert (
        recording.indexes["nested/quality"].attrs["sigmf-zarr:valid"] is False
    )
    with pytest.raises(ValueError, match="axis 'time' changed"):
        recording.index("nested/quality")
    for name in ("quality", "nested/quality"):
        with recording.mutate_index(name) as array:
            array.resize((5,))
            array[:] = np.arange(5)
    store.update_integrity()
    assert store.verify_integrity()


def test_indexes_are_writable_only_in_mutation_context(tmp_path) -> None:
    """Index mutation contexts should invalidate metadata hashes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    recording = store.recordings.open("rec")

    try:
        recording.index("quality")[0] = np.int16(100)
    except TypeError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("Expected direct index mutation to fail")

    with recording.mutate_index("quality") as index:
        index[0] = np.int16(100)

    assert recording.sample_sha512 is not None
    assert recording.metadata_sha512 is None
    assert store.verify_integrity() is False
    assert recording.index("quality")[0] == np.int16(100)


def test_failed_index_mutation_leaves_index_invalid(tmp_path) -> None:
    """A failed mutation should not leave an index looking usable.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If a failed mutation leaves the index valid.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    recording = store.recordings.open("rec")

    try:
        with recording.mutate_index("quality") as index:
            index[0] = np.int16(100)
            raise RuntimeError("stop")
    except RuntimeError:
        pass

    try:
        recording.index("quality")
    except ValueError as exc:
        assert "mutation did not complete" in str(exc)
    else:
        raise AssertionError("Expected failed mutation to invalidate index")


def test_axis_resize_invalidates_and_repair_restores_index(tmp_path) -> None:
    """Resizing an indexed axis should require explicit index repair.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If a stale index remains readable.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    recording = store.recordings.open("rec")
    recording.append_samples(np.ones((2, 1), dtype=np.float32))

    try:
        recording.index("quality")
    except ValueError as exc:
        assert "axis 'time' changed" in str(exc)
    else:
        raise AssertionError("Expected resized axis to invalidate its index")

    try:
        recording.update_integrity()
    except ValueError as exc:
        assert "Cannot update recording integrity" in str(exc)
    else:
        raise AssertionError("Expected hashing to reject an invalid index")

    with recording.mutate_index("quality") as index:
        index.resize((5,))
        index[:] = np.arange(5, dtype=np.int16)

    np.testing.assert_array_equal(
        recording.index("quality")[:],
        np.arange(5, dtype=np.int16),
    )


def test_store_index_uses_read_only_view_and_mutation_context(
    tmp_path,
) -> None:
    """Store-wide indexes should use the same protected mutation boundary.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    store.add_index("fold", np.array([0, 1], dtype=np.int8))
    store.update_integrity()

    try:
        store.index("fold")[0] = np.int8(1)
    except TypeError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("Expected direct store index mutation to fail")

    with store.mutate_index("fold") as index:
        index[0] = np.int8(1)

    assert store.metadata_sha512 is None
    assert store.index("fold")[0] == np.int8(1)


def test_unsupported_integrity_version_does_not_verify(tmp_path) -> None:
    """Verification should reject hashes with an unknown contract version.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    recording = store.recordings.open("rec")
    integrity = dict(recording.integrity)
    integrity["version"] = 2
    recording._raw_group.attrs["integrity"] = integrity

    assert recording.verify_integrity() is False
    assert store.verify_integrity() is False


def test_extension_channel_and_collection_views_are_managed(tmp_path) -> None:
    """All public metadata groups should enforce integrity invalidation.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = _create_hashed_store(tmp_path, zarr_format=3)
    recording = store.recordings.open("rec")
    collection = store.collections.open("all")
    channel_recording = store.recordings.open(
        "channels",
        create=True,
        sample_shape=(2, 2, 4),
        sample_axes=("channel", "iq", "time"),
    )
    store.update_integrity()

    for view in (recording.extensions, recording.channels, collection.group):
        try:
            view.attrs["unmanaged"] = True
        except TypeError:
            pass
        else:
            raise AssertionError("Expected metadata attrs to be read-only")

    for view in (recording.samples, recording.extensions, collection.group):
        try:
            _ = view.store
        except AttributeError:
            pass
        else:
            raise AssertionError("Expected backing stores to remain private")

    with recording.mutate_extensions() as extensions:
        extensions.create_group("example").attrs["value"] = 1

    assert recording.metadata_sha512 is None
    assert store.metadata_sha512 is None

    store.update_integrity()
    channel_recording.set_channel_metadata(0, {"test:receiver": "a"})

    assert channel_recording.metadata_sha512 is None
    assert store.metadata_sha512 is None

    store.update_integrity()
    collection.set_metadata({"core:description": "changed"})

    assert collection.metadata_sha512 is None
    assert store.metadata_sha512 is None


def test_item_metadata_is_writable_only_in_mutation_context(tmp_path) -> None:
    """Per-item metadata should use a validated managed mutation path.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(
        tmp_path / "batched.zarr", overwrite=True
    )
    recording = store.recordings.open(
        "batch",
        create=True,
        batched=True,
        sample_shape=(2, 4),
        sample_axes=("iq", "time"),
    )
    recording.append_samples(np.zeros((2, 2, 4), dtype=np.float32))
    recording.set_item_metadata([{}, {}])
    store.update_integrity()

    try:
        recording.item_metadata_array[0] = '{"global": {}}'
    except TypeError:
        pass
    else:
        raise AssertionError("Expected item metadata to be read-only")

    with recording.mutate_item_metadata() as item_metadata:
        item_metadata[0] = '{"global":{"test:value":1}}'

    assert recording.get_item_metadata(0) == {
        "global": {"test:value": 1}
    }
    assert recording.metadata_sha512 is None
    assert store.metadata_sha512 is None


def test_item_metadata_mutation_rejects_invalid_json(tmp_path) -> None:
    """Bulk mutation should validate encoded entries chunk by chunk.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If malformed JSON survives context validation.
    """
    store = SigMFZarrStore.create(
        tmp_path / "batched.zarr", overwrite=True
    )
    recording = store.recordings.open(
        "batch",
        create=True,
        batched=True,
        sample_shape=(2, 4),
        sample_axes=("iq", "time"),
    )
    recording.append_samples(np.zeros((2, 2, 4), dtype=np.float32))
    recording.set_item_metadata([{}, {}])
    store.update_integrity()

    try:
        with recording.mutate_item_metadata() as item_metadata:
            item_metadata[0] = "not JSON"
    except ValueError as exc:
        assert "not valid JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid item metadata JSON to fail")

    assert recording.metadata_sha512 is None
    assert store.metadata_sha512 is None
