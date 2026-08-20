"""Tests for store-level helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from numcodecs import CRC32C, Blosc, VLenUTF8, Zstd
from zarr.codecs import BloscCodec, VLenUTF8Codec
from zarr.core.array import Array

from sigmf_zarr.json import JSONValue, json_object
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat


def test_default_sample_chunks_preserves_existing_default() -> None:
    """Default chunking should stay sample-major without a byte target."""
    chunks = SigMFZarrStore.default_sample_chunks(
        "float32",
        (2, 128),
        5000,
        batched=True,
    )

    assert chunks == (1024, 2, 128)


def test_default_sample_chunks_targets_chunk_bytes_for_batches() -> None:
    """Batch chunking should aim for the requested byte size."""
    chunks = SigMFZarrStore.default_sample_chunks(
        "float32",
        (2, 128),
        5000,
        batched=True,
        target_chunk_bytes=1_048_576,
    )

    assert chunks == (1024, 2, 128)


def test_default_sample_chunks_targets_chunk_bytes_for_unbatched() -> None:
    """Unbatched chunking should keep the original sample shape."""
    chunks = SigMFZarrStore.default_sample_chunks(
        "float32",
        (4096, 512),
        1,
        batched=False,
        target_chunk_bytes=1_048_576,
    )

    assert chunks == (4096, 512)


def test_default_sample_chunks_rejects_non_positive_target() -> None:
    """Chunk byte targets must be positive."""
    with pytest.raises(
        ValueError, match="target_chunk_bytes must be positive"
    ):
        SigMFZarrStore.default_sample_chunks(
            "float32",
            (2, 128),
            1,
            target_chunk_bytes=0,
        )


def test_zarr_extension_declaration_is_optional() -> None:
    """SigMF-Zarr metadata should be advisory to classic SigMF readers."""
    declaration = SigMFRecording._zarr_extension_declaration()

    assert declaration["name"] == "sigmf-zarr"
    assert declaration["optional"] is True


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "name", ["", ".", "..", "a/b", "a\\b", "/a", "a/", "a\x00b"]
)
@pytest.mark.parametrize("kind", ["recording", "collection"])
def test_resource_names_reject_paths_before_writing(
    tmp_path: Path, zarr_format: ZarrFormat, name: str, kind: str
) -> None:
    """Rejected identifiers must not create intermediate resource groups.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        name: Empty identifier, path, or invalid group name.
        kind: Resource interface to exercise.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    store.update_metadata_integrity()
    original_hash = store.metadata_sha512
    if kind == "recording":
        with pytest.raises(
            ValueError, match="single nonempty Zarr group name"
        ):
            store.recordings.create(name, sample_shape=(1,), overwrite=True)
    else:
        with pytest.raises(
            ValueError, match="single nonempty Zarr group name"
        ):
            store.collections.create(name, overwrite=True)
    assert store.list_recordings() == ()
    assert store.list_collections() == ()
    assert store.metadata_sha512 == original_hash
    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("alias", ["/rec", "rec/", "./rec"])
def test_resource_aliases_cannot_bypass_collection_references(
    tmp_path: Path, zarr_format: ZarrFormat, alias: str
) -> None:
    """Normalized paths must not act as alternate recording identifiers.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        alias: Path that Zarr normalizes to the existing recording name.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    store.recordings.create("rec", sample_shape=(1,))
    store.collections.create("rec", recording_ids=["rec"])
    store.update_integrity()
    assert alias not in store.recordings
    assert alias not in store.collections
    with pytest.raises(KeyError):
        store.recordings.remove(alias)
    assert store.collections["rec"].recording_ids == ("rec",)
    assert store.list_recordings() == ("rec",)
    assert store.verify_integrity()


def test_schema_version_is_initial_draft() -> None:
    """Schema version should remain the initial pre-draft format."""
    assert SigMFZarrStore.SCHEMA_VERSION == 1


def test_create_defaults_to_zarr_format_3(tmp_path: Path) -> None:
    """New stores should use Zarr format 3 unless requested otherwise.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)

    assert store.zarr_format == 3
    assert (store_path / "zarr.json").is_file()


def test_zarr_format_2_round_trip(tmp_path: Path) -> None:
    """Format-2 stores should preserve the full logical schema.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=2)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
        sample_axes=("iq", "time"),
    )
    samples = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    recording.append_samples(samples)
    recording.add_index(
        "label",
        ["BPSK", "QPSK", "OOK"],
        axis="item",
        field="example:label",
    )
    recording.set_item_metadata(
        [
            {"global": {"example:item": 0}},
            None,
            {"global": {"example:item": 2}},
        ]
    )

    assert store.zarr_format == 2
    assert (store_path / ".zgroup").is_file()
    assert isinstance(recording.samples.metadata.compressor, Zstd)
    assert any(
        isinstance(codec, CRC32C)
        for codec in recording.samples.metadata.filters or ()
    )
    assert any(
        isinstance(codec, VLenUTF8)
        for codec in recording.index("label").metadata.filters or ()
    )
    assert any(
        isinstance(codec, VLenUTF8)
        for codec in recording.item_metadata_array.metadata.filters or ()
    )
    assert any(
        isinstance(codec, CRC32C)
        for codec in recording.item_metadata_array.metadata.filters or ()
    )

    reopened = SigMFZarrStore.open(store_path)
    reopened_recording = reopened.recordings["rec"]
    assert reopened.zarr_format == 2
    assert reopened_recording.sample_checksum == "crc32c"
    np.testing.assert_array_equal(reopened_recording.samples[:], samples)
    reopened_global = json_object(
        reopened_recording.resolved_item_metadata(2)["global"], name="global"
    )
    assert reopened_global["example:item"] == 2


def test_zarr_format_2_rejects_sharding(tmp_path: Path) -> None:
    """Format-2 stores should reject format-3-only shard settings.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr",
        zarr_format=2,
    )

    with pytest.raises(ValueError, match="requires Zarr format 3"):
        store.recordings.create(
            "rec",
            sample_shape=(2, 4),
            sample_shards=(2, 4),
        )

    assert "rec" not in store.recordings


def test_zarr_format_2_translates_sample_compression(tmp_path: Path) -> None:
    """Format-3 compressor options should map to format-2 numcodecs.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr",
        zarr_format=2,
    )
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 4),
        sample_compressor=BloscCodec(cname="zstd", clevel=7),
    )

    compressor = recording.samples.metadata.compressor
    assert isinstance(compressor, Blosc)
    assert compressor.cname == "zstd"
    assert compressor.clevel == 7


def test_zarr_format_2_can_disable_sample_checksums(tmp_path: Path) -> None:
    """Format-2 sample arrays should allow checksum filtering to be disabled.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr",
        zarr_format=2,
    )
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 4),
        sample_checksum=None,
    )

    assert recording.sample_checksum is None
    assert not any(
        isinstance(codec, CRC32C)
        for codec in recording.samples.metadata.filters or ()
    )


def test_zarr_format_2_detects_corrupted_sample_chunk(tmp_path: Path) -> None:
    """Format-2 CRC32C filters should reject corrupted sample bytes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=2)
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 4),
        sample_compressor=None,
    )
    recording.set_samples(np.ones((2, 4), dtype=np.float32))
    chunk_path = store_path / "recordings" / "rec" / "samples" / "0.0"
    encoded = bytearray(chunk_path.read_bytes())
    encoded[0] ^= 0xFF
    chunk_path.write_bytes(encoded)

    with pytest.raises(RuntimeError, match="crc32c checksum do not match"):
        _ = recording.samples[:]


def test_recording_batched_is_derived_from_sample_rank(tmp_path: Path) -> None:
    """Recording.batched should be derived from sample rank.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "batched",
        batched=True,
        sample_shape=(2, 128),
    )

    assert recording.batched is True
    assert recording.sample_shape == (2, 128)
    assert recording.sample_axes == ("iq", "time")
    assert recording.runtime_axes == ("item", "iq", "time")
    assert recording.axis_index("time") == 2
    assert recording.sample_count == 128
    assert "extra_metadata" not in dict(recording.group.attrs)
    assert recording.global_metadata["sigmf-zarr:sample-axes"] == [
        "iq",
        "time",
    ]

    reopened = SigMFZarrStore.open(
        tmp_path / "store.zarr",
        mode="r+",
    ).recordings.open("batched")

    assert reopened.batched is True
    assert reopened.sample_shape == (2, 128)
    assert reopened.sample_axes == ("iq", "time")


def test_recording_unbatched_is_derived_from_sample_rank(
    tmp_path: Path,
) -> None:
    """Unbatched recordings should derive from matching sample rank.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "unbatched",
        batched=False,
        sample_shape=(16, 2),
    )

    assert recording.batched is False
    assert recording.sample_shape == (16, 2)
    assert recording.sample_axes == ("time", "iq")
    assert recording.runtime_axes == ("time", "iq")
    assert recording.axis_index("time") == 0
    assert recording.sample_count == 16


def test_recording_can_use_channel_iq_time_axes(tmp_path: Path) -> None:
    """Recordings should support channel/IQ/time axis order.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "array",
        batched=True,
        sample_shape=(4, 2, 128),
        sample_axes=("channel", "iq", "time"),
    )

    assert recording.sample_axes == ("channel", "iq", "time")
    assert recording.runtime_axes == ("item", "channel", "iq", "time")
    assert recording.axis_index("channel") == 1
    assert recording.axis_index("iq") == 2
    assert recording.axis_index("time") == 3
    assert recording.sample_count == 128


def test_recording_stores_channel_metadata(tmp_path: Path) -> None:
    """Channel metadata should align with the explicit channel axis.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.create(
        "array",
        batched=True,
        sample_shape=(2, 2, 8),
        sample_axes=("channel", "iq", "time"),
        channel_metadata=[
            {"antenna:element": 0, "antenna:polarization": "H"},
            {"antenna:element": 1, "antenna:polarization": "V"},
        ],
    )

    assert recording.num_channels == 2
    assert recording.global_metadata["sigmf-zarr:num-channels"] == 2
    assert recording.channel_metadata(0) == {
        "antenna:element": 0,
        "antenna:polarization": "H",
    }
    recording.set_channel_metadata(
        1,
        {"antenna:element": 1, "antenna:gain": 12.5},
    )

    reopened = SigMFZarrStore.open(store_path).recordings.open("array")
    assert reopened.channel_metadata(1) == {
        "antenna:element": 1,
        "antenna:gain": 12.5,
    }


def test_recording_rejects_mismatched_channel_metadata(tmp_path: Path) -> None:
    """Creation should require one metadata object per channel.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)

    with pytest.raises(ValueError, match="Expected metadata for 2 channels"):
        store.recordings.create(
            "array",
            sample_shape=(2, 2, 8),
            sample_axes=("channel", "iq", "time"),
            channel_metadata=[{"antenna:element": 0}],
        )


def test_recording_samples_use_crc32c_by_default(tmp_path: Path) -> None:
    """Sample chunks should use CRC32C integrity checks by default.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 8),
    )

    assert recording.sample_checksum == "crc32c"
    assert recording.global_metadata["sigmf-zarr:checksum"] == "crc32c"
    assert SigMFRecording._codecs_have_crc32c(
        recording.samples.metadata.codecs
    )


def test_recording_can_disable_sample_checksums(tmp_path: Path) -> None:
    """Recordings should support explicitly disabling sample checksums.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 8),
        sample_checksum=None,
    )

    assert recording.sample_checksum is None
    assert not SigMFRecording._codecs_have_crc32c(
        recording.samples.metadata.codecs
    )


def test_recording_detects_corrupted_sample_chunk(tmp_path: Path) -> None:
    """CRC32C should reject corrupted encoded sample bytes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 4),
    )
    recording.set_samples(np.ones((2, 4), dtype=np.float32))
    chunk_path = (
        store_path / "recordings" / "rec" / "samples" / "c" / "0" / "0"
    )
    encoded = bytearray(chunk_path.read_bytes())
    encoded[0] ^= 0xFF
    chunk_path.write_bytes(encoded)

    with pytest.raises(ValueError, match="checksum do not match"):
        _ = recording.samples[:]


def test_sharded_recording_checksums_logical_chunks(tmp_path: Path) -> None:
    """CRC32C should remain inside a sharded sample codec pipeline.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
        sample_chunks=(1, 2, 4),
        sample_shards=(2, 2, 4),
    )

    assert SigMFRecording._codecs_have_crc32c(
        recording.samples.metadata.codecs
    )


def test_set_samples_resizes_unbatched_time_axis(tmp_path: Path) -> None:
    """set_samples should resize only the explicit time axis.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(2, 4),
    )
    samples = np.arange(12, dtype=np.float32).reshape(2, 6)

    recording.set_samples(samples)

    assert recording.samples.shape == (2, 6)
    assert recording.sample_shape == (2, 6)
    assert recording.sample_count == 6
    assert recording.global_metadata["sigmf-zarr:sample-shape"] == [2, 6]
    np.testing.assert_array_equal(recording.samples[:], samples)

    reopened = SigMFZarrStore.open(store_path, mode="r+").recordings["rec"]
    assert reopened.sample_shape == (2, 6)
    assert reopened.sample_count == 6
    np.testing.assert_array_equal(reopened.samples[:], samples)


def test_recording_sha512_lifecycle(tmp_path: Path) -> None:
    """Managed mutations should invalidate whole-recording SHA-512 metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
    )
    samples = np.arange(4, dtype=np.float32)
    recording.set_samples(samples)
    expected = hashlib.sha512(samples.astype("<f4").tobytes()).hexdigest()

    initial_sha512 = recording.sha512
    assert initial_sha512 is None
    assert recording.calculate_sha512() == expected
    assert recording.verify_sha512() is False
    assert recording.update_sha512() == expected
    updated_sha512 = recording.sha512
    assert updated_sha512 == expected
    assert recording.verify_sha512() is True

    with pytest.raises(TypeError, match="read-only"):
        recording.samples[0] = np.float32(99.0)
    unchanged_sha512 = recording.sha512
    assert unchanged_sha512 == expected

    with recording.mutate_samples() as writable_samples:
        writable_samples[0] = np.float32(99.0)
    assert recording.sha512 is None

    recording.update_sha512()
    recording.append_samples(np.array([5.0], dtype=np.float32))
    assert recording.sha512 is None

    recording.update_sha512()
    recording.set_samples(samples)
    assert recording.sha512 is None


def test_datatype_change_invalidates_recording_sha512(tmp_path: Path) -> None:
    """Changing the output byte encoding should invalidate `core:sha512`.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
    )
    recording.update_sha512()

    recording.set_global_field("core:datatype", "rf32_be")

    assert recording.sha512 is None


def test_set_samples_rejects_non_time_axis_resize(tmp_path: Path) -> None:
    """set_samples should reject non-time axis changes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(4, 2, 8),
        sample_axes=("channel", "iq", "time"),
    )

    with pytest.raises(ValueError, match="Expected 'channel' axis length 4"):
        recording.set_samples(np.zeros((5, 2, 8), dtype=np.float32))


def test_set_samples_rejects_rank_changes(tmp_path: Path) -> None:
    """set_samples should reject arrays with different rank.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(2, 8),
    )

    with pytest.raises(ValueError, match="Expected samples with ndim 2"):
        recording.set_samples(np.zeros((2, 8, 1), dtype=np.float32))


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("int16", np.full(4, 1e6)),
        ("int16", np.array([0.5, 1.0, 2.0, 3.0])),
        ("int16", np.array([np.nan, 1.0, 2.0, 3.0])),
        ("float32", np.full(4, 1 + 2j)),
        ("float32", np.full(4, 2**24 + 1, dtype=np.int64)),
        ("uint8", np.array([-1, 0, 1, 2])),
        ("uint16", np.array([-1, 0, 1, 2], dtype=np.int16)),
        ("uint32", np.array([-1, 0, 1, 2], dtype=np.int16)),
        ("int16", np.array([0, 1, 32768, 65535], dtype=np.uint16)),
        ("uint64", np.array([-1, 0, 1, 2], dtype=np.int64)),
        ("int64", np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64)),
        ("float32", np.array([True, False, True, False], dtype=object)),
    ],
)
def test_lossy_sample_conversion_is_rejected(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    dtype: str,
    values: npt.NDArray[np.generic],
) -> None:
    """Appends and replacements must not wrap, round, or drop components.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        dtype: Stored sample dtype.
        values: Four samples that the stored dtype cannot represent.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    batch = store.recordings.create(
        "batch", batched=True, sample_shape=(4,), sample_dtype=dtype
    )
    continuous = store.recordings.create(
        "continuous", sample_shape=(4,), sample_dtype=dtype
    )
    original = continuous.samples[:]

    with pytest.raises(ValueError, match=r"numeric|without changing"):
        batch.append_samples(np.asarray(values).reshape(1, 4))
    with pytest.raises(ValueError, match=r"numeric|without changing"):
        continuous.set_samples(values)
    with pytest.raises(ValueError, match=r"numeric|without changing"):
        continuous.append_samples(values)

    assert len(batch) == 0
    np.testing.assert_array_equal(continuous.samples[:], original)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("float32", np.arange(4, dtype=np.int64)),
        ("float32", np.array([0.5, -1.25, 3.0, np.nan])),
        ("int16", np.array([-3.0, 0.0, 2.0, 32767.0])),
        ("uint16", np.array([0, 1, 2, 32767], dtype=np.int16)),
        ("int16", np.array([0, 1, 2, 32767], dtype=np.uint16)),
        ("uint64", np.array([0, 1, 2, 2**63 - 1], dtype=np.int64)),
        ("int64", np.array([0, 1, 2, 2**63 - 1], dtype=np.uint64)),
        ("complex64", np.array([1, 2, 3, 4], dtype=np.int8)),
        ("float64", np.array([True, False, True, False])),
    ],
)
def test_exact_sample_conversion_is_accepted(
    tmp_path: Path, dtype: str, values: npt.NDArray[np.generic]
) -> None:
    """Conversions that preserve every value remain permitted.

    Args:
        tmp_path: Pytest temporary path fixture.
        dtype: Stored sample dtype.
        values: Four samples representable in the stored dtype.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec", batched=True, sample_shape=(4,), sample_dtype=dtype
    )
    recording.append_samples(np.asarray(values).reshape(1, 4))
    np.testing.assert_array_equal(
        recording.samples[0], np.asarray(values).astype(dtype)
    )


def test_append_samples_adds_batched_capture(tmp_path: Path) -> None:
    """append_samples should append captures for batched recordings.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    samples = np.arange(16, dtype=np.float32).reshape(2, 2, 4)

    recording.append_samples(
        samples,
        capture={"core:frequency": 915_000_000.0},
    )

    assert recording.samples.shape == (2, 2, 4)
    np.testing.assert_array_equal(recording.samples[:], samples)
    assert recording.captures == [
        {
            "core:frequency": 915_000_000.0,
            "core:sample_start": 0,
        }
    ]


def test_recording_stores_and_resolves_item_metadata(tmp_path: Path) -> None:
    """Per-item bundles should supplement shared recording metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
        global_metadata={"core:sample_rate": 1_000_000.0},
        captures=[{"core:sample_start": 0}],
    )
    recording.append_samples(np.zeros((2, 2, 4), dtype=np.float32))
    recording.set_item_metadata(
        [
            None,
            {
                "global": {"core:frequency": 915_000_000.0},
                "captures": [{"core:datetime": "2026-08-19T00:00:00Z"}],
                "annotations": [{"core:label": "example"}],
            },
        ]
    )

    assert recording.has_item_metadata is True
    assert "sigmf-zarr:has-item-metadata" not in recording.global_metadata
    assert recording.get_item_metadata(0) == {}
    assert recording.get_item_metadata(-1) == {
        "global": {"core:frequency": 915_000_000.0},
        "captures": [{"core:datetime": "2026-08-19T00:00:00Z"}],
        "annotations": [{"core:label": "example"}],
    }
    resolved = recording.resolved_item_metadata(1)
    resolved_global = json_object(resolved["global"], name="global")
    assert resolved_global["core:sample_rate"] == 1_000_000.0
    assert resolved_global["core:frequency"] == 915_000_000.0
    assert resolved["captures"] == [
        {"core:sample_start": 0},
        {"core:datetime": "2026-08-19T00:00:00Z"},
    ]
    assert resolved["annotations"] == [{"core:label": "example"}]

    reopened = SigMFZarrStore.open(store_path).recordings.open("rec")
    assert reopened.get_item_metadata(1) == recording.get_item_metadata(1)
    assert SigMFRecording._codecs_have_crc32c(
        reopened.item_metadata_array.metadata.codecs
    )


def test_item_metadata_entry_and_slice_setters(tmp_path: Path) -> None:
    """Validated setters should update only the selected item bundles.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(np.zeros((4, 2, 4), dtype=np.float32))

    recording.set_item_metadata_entry(
        1,
        {"global": {"example:value": 1}},
    )
    recording.set_item_metadata_slice(
        slice(2, 4),
        [
            {"global": {"example:value": 2}},
            {"global": {"example:value": 3}},
        ],
    )

    assert recording.get_item_metadata(0) == {}
    assert recording.get_item_metadata(1) == {"global": {"example:value": 1}}
    assert recording.get_item_metadata(2) == {"global": {"example:value": 2}}
    assert recording.get_item_metadata(-1) == {"global": {"example:value": 3}}


def test_item_metadata_setters_validate_before_mutation(
    tmp_path: Path,
) -> None:
    """Rejected setter values should not change content or integrity.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((2, 2, 4), dtype=np.float32),
        item_metadata=[None, None],
    )
    store.update_integrity()
    integrity = recording.integrity

    with pytest.raises(ValueError, match="cannot override storage fields"):
        recording.set_item_metadata_entry(
            0,
            {"global": {"sigmf-zarr:dtype": "<i2"}},
        )

    assert recording.get_item_metadata(0) == {}
    assert recording.integrity == integrity


def test_item_metadata_slice_setter_validates_selection(
    tmp_path: Path,
) -> None:
    """Slice updates should require contiguous, correctly sized values.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(np.zeros((4, 2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="step of 1"):
        recording.set_item_metadata_slice(slice(None, None, 2), [{}, {}])

    with pytest.raises(
        ValueError, match="Expected metadata for 2 selected items"
    ):
        recording.set_item_metadata_slice(slice(1, 3), [{}])


def test_item_metadata_validation_reads_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full validation should not call the scalar metadata accessor.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((4097, 2, 4), dtype=np.float32),
        item_metadata=[None] * 4097,
    )

    def fail_scalar_access(
        unused_recording: SigMFRecording,
        unused_item_index: int,
    ) -> object:
        del unused_recording, unused_item_index
        raise AssertionError("scalar item metadata access is too slow")

    monkeypatch.setattr(
        SigMFRecording,
        "get_item_metadata",
        fail_scalar_access,
    )

    recording._validate_item_metadata_array()


def test_append_samples_keeps_item_metadata_aligned(tmp_path: Path) -> None:
    """Batched appends should extend item metadata with explicit defaults.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(np.zeros((1, 2, 4), dtype=np.float32))
    recording.append_samples(
        np.ones((2, 2, 4), dtype=np.float32),
        item_metadata=[
            {"global": {"core:frequency": 100.0}},
            None,
        ],
    )
    recording.append_samples(np.ones((1, 2, 4), dtype=np.float32))

    assert recording.item_metadata_array.shape == (4,)
    assert recording.get_item_metadata(0) == {}
    assert recording.get_item_metadata(1) == {
        "global": {"core:frequency": 100.0}
    }
    assert recording.get_item_metadata(2) == {}
    assert recording.get_item_metadata(3) == {}


def test_item_metadata_rejects_storage_overrides(tmp_path: Path) -> None:
    """Item metadata should not override SigMF-Zarr storage fields.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(np.zeros((1, 2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="cannot override storage fields"):
        recording.set_item_metadata([{"global": {"sigmf-zarr:dtype": "<i2"}}])


def test_clear_item_metadata_removes_array(tmp_path: Path) -> None:
    """Clearing item metadata should restore the absent-array state.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((1, 2, 4), dtype=np.float32),
        item_metadata=[None],
    )

    recording.clear_item_metadata()

    assert recording.has_item_metadata is False
    assert "sigmf-zarr:has-item-metadata" not in recording.global_metadata
    assert recording.get_item_metadata(0) == {}


def test_item_metadata_detects_corrupted_zarr_format_3_chunk(
    tmp_path: Path,
) -> None:
    """CRC32C should reject corrupted format-3 item metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((1, 2, 4), dtype=np.float32),
        item_metadata=[{"global": {"example:value": 1}}],
    )
    chunk_path = (
        store_path / "recordings" / "rec" / "item_metadata" / "c" / "0"
    )
    encoded = bytearray(chunk_path.read_bytes())
    encoded[-1] ^= 0xFF
    chunk_path.write_bytes(encoded)

    with pytest.raises(ValueError, match="checksum do not match"):
        recording.get_item_metadata(0)


def test_item_metadata_detects_corrupted_zarr_format_2_chunk(
    tmp_path: Path,
) -> None:
    """CRC32C should reject corrupted format-2 item metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=2)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((1, 2, 4), dtype=np.float32),
        item_metadata=[{"global": {"example:value": 1}}],
    )
    item_metadata = recording.item_metadata_array
    compressor = item_metadata.metadata.compressor
    assert compressor is not None
    chunk_path = store_path / "recordings" / "rec" / "item_metadata" / "0"
    decoded = bytearray(compressor.decode(chunk_path.read_bytes()))
    decoded[0] ^= 0xFF
    chunk_path.write_bytes(bytes(compressor.encode(decoded)))

    with pytest.raises(RuntimeError, match="crc32c checksum do not match"):
        recording.get_item_metadata(0)


def test_append_samples_extends_unbatched_time_axis_with_capture(
    tmp_path: Path,
) -> None:
    """append_samples should extend unbatched recordings along time.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(2, 4),
    )
    appended = np.arange(6, dtype=np.float32).reshape(2, 3)

    recording.append_samples(
        appended,
        capture={"core:datetime": "2026-06-01T00:00:00Z"},
    )

    assert recording.samples.shape == (2, 7)
    assert recording.sample_shape == (2, 7)
    assert recording.sample_count == 7
    assert recording.global_metadata["sigmf-zarr:sample-shape"] == [2, 7]
    np.testing.assert_array_equal(recording.samples[:, 4:7], appended)
    assert recording.captures == [
        {
            "core:datetime": "2026-06-01T00:00:00Z",
            "core:sample_start": 4,
        }
    ]

    reopened = SigMFZarrStore.open(store_path, mode="r+").recordings["rec"]
    assert reopened.sample_shape == (2, 7)
    assert reopened.sample_count == 7


def test_append_samples_rejects_unbatched_non_time_axis_change(
    tmp_path: Path,
) -> None:
    """append_samples should reject unbatched non-time axis changes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(4, 2, 8),
        sample_axes=("channel", "iq", "time"),
    )

    with pytest.raises(ValueError, match="Expected 'channel' axis length 4"):
        recording.append_samples(np.zeros((5, 2, 3), dtype=np.float32))


def test_recording_add_index_records_axis_metadata(tmp_path: Path) -> None:
    """Recording indexes should describe the metadata they materialize.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 8),
    )
    recording.append_samples(np.zeros((3, 2, 8), dtype=np.float32))

    index = recording.add_index(
        "snr_db",
        np.array([0, 2, 4], dtype=np.int16),
        axis="item",
        field="radioml:snr",
        unit="dB",
    )

    np.testing.assert_array_equal(
        index[:],
        np.array([0, 2, 4], dtype=np.int16),
    )
    assert dict(index.attrs) == {
        "axis": "item",
        "field": "radioml:snr",
        "kind": "metadata",
        "unit": "dB",
    }
    np.testing.assert_array_equal(recording.index("snr_db")[:], index[:])


def test_recording_add_index_stores_label_metadata(tmp_path: Path) -> None:
    """Recording indexes should support label lookup metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(np.zeros((2, 2, 4), dtype=np.float32))

    index = recording.add_index(
        "mod_class_id",
        np.array([0, 1], dtype=np.int16),
        axis="item",
        field="radioml:mod_class",
        labels=["AM-DSB", "BPSK"],
    )

    assert dict(index.attrs) == {
        "axis": "item",
        "field": "radioml:mod_class",
        "kind": "metadata",
        "labels": ["AM-DSB", "BPSK"],
    }


def test_recording_add_index_rejects_axis_length_mismatch(
    tmp_path: Path,
) -> None:
    """Recording indexes should match the indexed axis length.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(2, 8),
    )

    with pytest.raises(
        ValueError, match="does not match axis 'time' length 8"
    ):
        recording.add_index(
            "snr_db",
            np.array([0, 2], dtype=np.int16),
            axis="time",
            field="radioml:snr",
        )


def test_decode_json_index_rejects_non_object_entries(tmp_path: Path) -> None:
    """Decoded index entries must be finite JSON objects.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    store.add_index("good", np.array(['{"a": 1}', "{}"]))
    assert store.decode_json_index("good") == [{"a": 1}, {}]
    for name, entries in (
        ("list", ['{"a": 1}', "[1, 2]"]),
        ("nan", ['{"a": NaN}']),
        ("text", ["not json"]),
    ):
        store.add_index(name, np.array(entries))
        with pytest.raises(ValueError, match=f"index '{name}' entry"):
            store.decode_json_index(name)


def test_recording_open_rejects_invalid_sample_axes(tmp_path: Path) -> None:
    """Opening should reject invalid sample-axis metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(2, 16),
    )
    metadata = dict(recording.global_metadata)
    metadata["sigmf-zarr:sample-axes"] = ["item", "time"]
    recording._raw_group.attrs["global"] = metadata

    with pytest.raises(ValueError, match="reserved axis 'item'"):
        store.recordings.open("rec")


def test_item_metadata_presence_is_derived_from_array(tmp_path: Path) -> None:
    """Item metadata presence should not require a redundant global flag.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((1, 2, 4), dtype=np.float32),
        item_metadata=[None],
    )
    assert "sigmf-zarr:has-item-metadata" not in recording.global_metadata
    assert store.recordings.open("rec").has_item_metadata is True


def test_recording_open_rejects_mismatched_channel_groups(
    tmp_path: Path,
) -> None:
    """Opening should reject channel groups that do not match the axis.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 2, 4),
        sample_axes=("channel", "iq", "time"),
    )
    del recording._channels_group["1"]

    with pytest.raises(ValueError, match="do not match expected channels"):
        store.recordings.open("rec")


def test_recording_open_rejects_mismatched_checksum_metadata(
    tmp_path: Path,
) -> None:
    """Opening should reject checksum declarations without matching codecs.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 4),
    )
    metadata = dict(recording.global_metadata)
    metadata["sigmf-zarr:checksum"] = None
    recording._raw_group.attrs["global"] = metadata

    with pytest.raises(
        ValueError, match="does not match sample codec checksum"
    ):
        store.recordings.open("rec")


def test_recording_creation_replaces_supplied_zarr_metadata(
    tmp_path: Path,
) -> None:
    """Creation should replace caller-supplied SigMF-Zarr metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=True,
        sample_shape=(4,),
        global_metadata={
            "core:extensions": [
                {
                    "name": "sigmf-zarr",
                    "version": "stale",
                    "optional": False,
                },
                {
                    "name": "other-extension",
                    "version": "1.0.0",
                    "optional": True,
                },
            ],
            "sigmf-zarr:dtype": "<i2",
            "sigmf-zarr:sample-shape": [999],
            "sigmf-zarr:sample-axes": ["time"],
        },
    )

    assert recording.batched is True
    assert recording.global_metadata["sigmf-zarr:dtype"] == "<f4"
    assert recording.global_metadata["sigmf-zarr:sample-shape"] == [4]
    assert recording.global_metadata["sigmf-zarr:sample-axes"] == ["time"]
    assert recording.global_metadata["core:extensions"] == [
        {
            "name": "other-extension",
            "version": "1.0.0",
            "optional": True,
        },
        {
            "name": "sigmf-zarr",
            "version": "0.1.0",
            "optional": True,
        },
    ]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_extension_updates_preserve_the_managed_declaration(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Extension edits retain the storage declaration and other namespaces.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    recording = store.recordings.create("rec", sample_shape=(4,))
    own = SigMFRecording._zarr_extension_declaration()
    other = {"name": "other", "version": "1.0.0", "optional": True}
    stale = {"name": "sigmf-zarr", "version": "stale", "optional": False}
    for declarations in ([], [other], [stale, other, stale]):
        recording.set_global_field("core:extensions", declarations)
        expected = [other, own] if other in declarations else [own]
        assert recording.global_metadata["core:extensions"] == expected
        assert (
            store.recordings["rec"].global_metadata
            == recording.global_metadata
        )
        store.update_integrity()
        assert store.verify_integrity()
    original_hash = store.metadata_sha512
    with pytest.raises(ValueError, match="core:extensions"):
        recording.set_global_field("core:extensions", ["invalid"])
    assert store.metadata_sha512 == original_hash
    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "declarations",
    [
        [],
        [{"name": "sigmf-zarr", "version": "wrong", "optional": True}],
        [{"name": "sigmf-zarr", "version": "0.1.0", "optional": False}],
        [{"name": "sigmf-zarr", "version": "0.1.0", "optional": 1}],
        [SigMFRecording._zarr_extension_declaration()] * 2,
    ],
)
def test_open_rejects_invalid_storage_extension_declarations(
    tmp_path: Path, zarr_format: ZarrFormat, declarations: list[JSONValue]
) -> None:
    """A matching integrity hash does not establish schema conformance.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        declarations: Missing, contradictory, or duplicated declarations.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    recording = store.recordings.create("rec", sample_shape=(4,))
    metadata = dict(recording.global_metadata)
    metadata["core:extensions"] = declarations
    recording._raw_group.attrs["global"] = metadata
    recording.update_integrity()
    store.update_metadata_integrity()
    with pytest.raises(ValueError, match="core:extensions must declare"):
        store.recordings.open("rec")


def test_set_global_field_rejects_managed_storage_fields(
    tmp_path: Path,
) -> None:
    """Storage fields must stay consistent with the sample array.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create("rec", sample_shape=(2, 8))
    original = recording.global_metadata
    for key, value in (
        ("sigmf-zarr:sample-shape", [3]),
        ("sigmf-zarr:sample-axes", ["time", "iq"]),
        ("sigmf-zarr:dtype", "<i2"),
    ):
        with pytest.raises(ValueError, match="managed by the recording"):
            recording.set_global_field(key, value)
    assert recording.global_metadata == original
    assert store.recordings["rec"].sample_shape == (2, 8)


def test_recording_open_rejects_invalid_sample_shape_rank(
    tmp_path: Path,
) -> None:
    """Opening should reject sample shapes that disagree with axes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.create(
        "rec",
        batched=False,
        sample_shape=(16, 2),
    )
    metadata = dict(recording.global_metadata)
    metadata["sigmf-zarr:sample-shape"] = [16, 2, 1]
    recording._raw_group.attrs["global"] = metadata

    with pytest.raises(ValueError, match="same length as sample_shape"):
        store.recordings.open("rec")


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("dtype", ["U", "O"])
def test_recording_extensions_store_variable_length_utf8(
    tmp_path: Path, zarr_format: ZarrFormat, dtype: str
) -> None:
    """Unicode and object strings must retain UTF-8 and variable lengths.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        dtype: NumPy Unicode or object input representation.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    recording = store.recordings.create("rec", sample_shape=(4,))
    values = np.array(["AM", "caf\u00e9"], dtype=dtype)  # codespell:ignore
    recording.add_extension_array("labels", values)
    labels = recording.extensions["labels"]
    assert isinstance(labels, ReadOnlyArray)
    np.testing.assert_array_equal(labels[:], values)
    with recording.mutate_extensions() as group:
        writable_labels = group["labels"]
        assert isinstance(writable_labels, Array)
        writable_labels[0] = "A much longer label"
    updated_labels = recording.extensions["labels"]
    assert isinstance(updated_labels, ReadOnlyArray)
    np.testing.assert_array_equal(
        updated_labels[:],
        ["A much longer label", "caf\u00e9"],  # codespell:ignore
    )


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_store_string_index_uses_variable_length_utf8(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Store-wide string indexes should use variable-length UTF-8 storage.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )

    index = store.add_index("labels", np.array(["AM-DSB", "BPSK"]))

    assert index.shape == (2,)
    assert index.chunks == (2,)
    assert list(index[:]) == ["AM-DSB", "BPSK"]
    if zarr_format == 3:
        assert isinstance(index.serializer, VLenUTF8Codec)
    else:
        assert any(
            codec.codec_id == "vlen-utf8"
            for codec in index.metadata.filters or ()
        )
