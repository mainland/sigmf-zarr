"""Tests for store-level helpers."""

from __future__ import annotations

import hashlib

import numpy as np
from numcodecs import CRC32C, Blosc, VLenUTF8, Zstd
from zarr.codecs import BloscCodec, VLenUTF8Codec

from sigmf_zarr.store import SigMFRecording, SigMFZarrStore


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
    """Unbatched chunking should divide the final sample axis."""
    chunks = SigMFZarrStore.default_sample_chunks(
        "float32",
        (4096, 512),
        1,
        batched=False,
        target_chunk_bytes=1_048_576,
    )

    assert chunks == (4096, 64)


def test_default_sample_chunks_rejects_non_positive_target() -> None:
    """Chunk byte targets must be positive.

    Raises:
        AssertionError: If non-positive target sizing does not raise.
    """
    try:
        SigMFZarrStore.default_sample_chunks(
            "float32",
            (2, 128),
            1,
            target_chunk_bytes=0,
        )
    except ValueError as exc:
        assert "target_chunk_bytes must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive target")


def test_zarr_extension_declaration_is_optional() -> None:
    """SigMF-Zarr metadata should be advisory to classic SigMF readers."""
    declaration = SigMFRecording._zarr_extension_declaration()

    assert declaration["name"] == "sigmf-zarr"
    assert declaration["optional"] is True


def test_schema_version_is_initial_draft() -> None:
    """Schema version should remain the initial pre-draft format."""
    assert SigMFZarrStore.SCHEMA_VERSION == 1


def test_create_defaults_to_zarr_format_3(tmp_path) -> None:
    """New stores should use Zarr format 3 unless requested otherwise.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)

    assert store.zarr_format == 3
    assert (store_path / "zarr.json").is_file()


def test_zarr_format_2_round_trip(tmp_path) -> None:
    """Format-2 stores should preserve the full logical schema.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=2)
    recording = store.recordings.open(
        "rec",
        create=True,
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
    assert reopened_recording.resolved_item_metadata(2)["global"][
        "example:item"
    ] == 2


def test_zarr_format_2_rejects_sharding(tmp_path) -> None:
    """Format-2 stores should reject format-3-only shard settings.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If sample sharding is accepted for format 2.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr",
        zarr_format=2,
    )

    try:
        store.recordings.open(
            "rec",
            create=True,
            sample_shape=(2, 4),
            sample_shards=(2, 4),
        )
    except ValueError as exc:
        assert "requires Zarr format 3" in str(exc)
    else:
        raise AssertionError("Expected format-2 sharding to fail")

    assert "rec" not in store.recordings


def test_zarr_format_2_translates_sample_compression(tmp_path) -> None:
    """Format-3 compressor options should map to format-2 numcodecs.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr",
        zarr_format=2,
    )
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 4),
        sample_compressor=BloscCodec(cname="zstd", clevel=7),
    )

    compressor = recording.samples.metadata.compressor
    assert isinstance(compressor, Blosc)
    assert compressor.cname == "zstd"
    assert compressor.clevel == 7


def test_zarr_format_2_translates_lz4_sample_compression(tmp_path) -> None:
    """Format-3 LZ4 options should map to a format-2 Blosc codec.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr",
        zarr_format=2,
    )
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 4),
        sample_compressor=BloscCodec(cname="lz4", clevel=4),
    )

    compressor = recording.samples.metadata.compressor
    assert isinstance(compressor, Blosc)
    assert compressor.cname == "lz4"
    assert compressor.clevel == 4


def test_zarr_format_2_can_disable_sample_checksums(tmp_path) -> None:
    """Format-2 sample arrays should allow checksum filtering to be disabled.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr",
        zarr_format=2,
    )
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 4),
        sample_checksum=None,
    )

    assert recording.sample_checksum is None
    assert not any(
        isinstance(codec, CRC32C)
        for codec in recording.samples.metadata.filters or ()
    )


def test_zarr_format_2_detects_corrupted_sample_chunk(tmp_path) -> None:
    """Format-2 CRC32C filters should reject corrupted sample bytes.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If corrupted format-2 samples can be read.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=2)
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 4),
        sample_compressor=None,
    )
    recording.set_samples(np.ones((2, 4), dtype=np.float32))
    chunk_path = store_path / "recordings" / "rec" / "samples" / "0.0"
    encoded = bytearray(chunk_path.read_bytes())
    encoded[0] ^= 0xFF
    chunk_path.write_bytes(encoded)

    try:
        _ = recording.samples[:]
    except RuntimeError as exc:
        assert "crc32c checksum do not match" in str(exc)
    else:
        raise AssertionError("Expected corrupted format-2 chunk to fail")


def test_recording_batched_is_derived_from_sample_rank(tmp_path) -> None:
    """Recording.batched should be derived from sample rank.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "batched",
        create=True,
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
        mode="a",
    ).recordings.open("batched")

    assert reopened.batched is True
    assert reopened.sample_shape == (2, 128)
    assert reopened.sample_axes == ("iq", "time")


def test_recording_unbatched_is_derived_from_sample_rank(tmp_path) -> None:
    """Unbatched recordings should derive from matching sample rank.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "unbatched",
        create=True,
        batched=False,
        sample_shape=(16, 2),
    )

    assert recording.batched is False
    assert recording.sample_shape == (16, 2)
    assert recording.sample_axes == ("time", "iq")
    assert recording.runtime_axes == ("time", "iq")
    assert recording.axis_index("time") == 0
    assert recording.sample_count == 16


def test_recording_can_use_channel_iq_time_axes(tmp_path) -> None:
    """Recordings should support channel/IQ/time axis order.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "array",
        create=True,
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


def test_recording_stores_channel_metadata(tmp_path) -> None:
    """Channel metadata should align with the explicit channel axis.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.open(
        "array",
        create=True,
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


def test_recording_rejects_mismatched_channel_metadata(tmp_path) -> None:
    """Creation should require one metadata object per channel.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If mismatched channel metadata is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)

    try:
        store.recordings.open(
            "array",
            create=True,
            sample_shape=(2, 2, 8),
            sample_axes=("channel", "iq", "time"),
            channel_metadata=[{"antenna:element": 0}],
        )
    except ValueError as exc:
        assert "Expected metadata for 2 channels" in str(exc)
    else:
        raise AssertionError("Expected mismatched channel metadata to fail")


def test_recording_samples_use_crc32c_by_default(tmp_path) -> None:
    """Sample chunks should use CRC32C integrity checks by default.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 8),
    )

    assert recording.sample_checksum == "crc32c"
    assert recording.global_metadata["sigmf-zarr:checksum"] == "crc32c"
    assert SigMFRecording._codecs_have_crc32c(
        recording.samples.metadata.codecs
    )


def test_recording_can_disable_sample_checksums(tmp_path) -> None:
    """Recordings should support explicitly disabling sample checksums.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 8),
        sample_checksum=None,
    )

    assert recording.sample_checksum is None
    assert not SigMFRecording._codecs_have_crc32c(
        recording.samples.metadata.codecs
    )


def test_recording_detects_corrupted_sample_chunk(tmp_path) -> None:
    """CRC32C should reject corrupted encoded sample bytes.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If corrupted samples can be read without error.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 4),
    )
    recording.set_samples(np.ones((2, 4), dtype=np.float32))
    chunk_path = (
        store_path / "recordings" / "rec" / "samples" / "c" / "0" / "0"
    )
    encoded = bytearray(chunk_path.read_bytes())
    encoded[0] ^= 0xFF
    chunk_path.write_bytes(encoded)

    try:
        _ = recording.samples[:]
    except ValueError as exc:
        assert "checksum do not match" in str(exc)
    else:
        raise AssertionError("Expected corrupted sample chunk to fail")


def test_sharded_recording_checksums_logical_chunks(tmp_path) -> None:
    """CRC32C should remain inside a sharded sample codec pipeline.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=True,
        sample_shape=(2, 4),
        sample_chunks=(1, 2, 4),
        sample_shards=(2, 2, 4),
    )

    assert SigMFRecording._codecs_have_crc32c(
        recording.samples.metadata.codecs
    )


def test_set_samples_resizes_unbatched_time_axis(tmp_path) -> None:
    """set_samples should resize only the explicit time axis.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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

    reopened = SigMFZarrStore.open(store_path, mode="a").recordings.open("rec")
    assert reopened.sample_shape == (2, 6)
    assert reopened.sample_count == 6
    np.testing.assert_array_equal(reopened.samples[:], samples)


def test_recording_sha512_lifecycle(tmp_path) -> None:
    """Managed mutations should invalidate whole-recording SHA-512 metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
    )
    samples = np.arange(4, dtype=np.float32)
    recording.set_samples(samples)
    expected = hashlib.sha512(samples.astype("<f4").tobytes()).hexdigest()

    assert recording.sha512 is None
    assert recording.calculate_sha512() == expected
    assert recording.verify_sha512() is False
    assert recording.update_sha512() == expected
    assert recording.sha512 == expected
    assert recording.verify_sha512() is True

    try:
        recording.samples[0] = np.float32(99.0)
    except TypeError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("Expected direct sample mutation to fail")
    assert recording.sha512 == expected

    with recording.mutate_samples() as writable_samples:
        writable_samples[0] = np.float32(99.0)
    assert recording.sha512 is None

    recording.update_sha512()
    recording.append_samples(np.array([5.0], dtype=np.float32))
    assert recording.sha512 is None

    recording.update_sha512()
    recording.set_samples(samples)
    assert recording.sha512 is None


def test_datatype_change_invalidates_recording_sha512(tmp_path) -> None:
    """Changing the output byte encoding should invalidate `core:sha512`.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
    )
    recording.update_sha512()

    recording.set_global_field("core:datatype", "rf32_be")

    assert recording.sha512 is None


def test_set_samples_rejects_non_time_axis_resize(tmp_path) -> None:
    """set_samples should reject non-time axis changes.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If a non-time axis resize is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(4, 2, 8),
        sample_axes=("channel", "iq", "time"),
    )

    try:
        recording.set_samples(np.zeros((5, 2, 8), dtype=np.float32))
    except ValueError as exc:
        assert "Expected 'channel' axis length 4" in str(exc)
    else:
        raise AssertionError("Expected non-time axis resize to fail")


def test_set_samples_rejects_rank_changes(tmp_path) -> None:
    """set_samples should reject arrays with different rank.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If a rank change is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(2, 8),
    )

    try:
        recording.set_samples(np.zeros((2, 8, 1), dtype=np.float32))
    except ValueError as exc:
        assert "Expected samples with ndim 2" in str(exc)
    else:
        raise AssertionError("Expected rank change to fail")


def test_append_samples_adds_batched_capture(tmp_path) -> None:
    """append_samples should append captures for batched recordings.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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


def test_recording_stores_and_resolves_item_metadata(tmp_path) -> None:
    """Per-item bundles should supplement shared recording metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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
    assert resolved["global"]["core:sample_rate"] == 1_000_000.0
    assert resolved["global"]["core:frequency"] == 915_000_000.0
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


def test_item_metadata_entry_and_slice_setters(tmp_path) -> None:
    """Validated setters should update only the selected item bundles.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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
    assert recording.get_item_metadata(1) == {
        "global": {"example:value": 1}
    }
    assert recording.get_item_metadata(2) == {
        "global": {"example:value": 2}
    }
    assert recording.get_item_metadata(-1) == {
        "global": {"example:value": 3}
    }


def test_item_metadata_setters_validate_before_mutation(tmp_path) -> None:
    """Rejected setter values should not change content or integrity.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If invalid metadata changes the recording.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((2, 2, 4), dtype=np.float32),
        item_metadata=[None, None],
    )
    store.update_integrity()
    integrity = recording.integrity

    try:
        recording.set_item_metadata_entry(
            0,
            {"global": {"sigmf-zarr:dtype": "<i2"}},
        )
    except ValueError as exc:
        assert "cannot override storage fields" in str(exc)
    else:
        raise AssertionError("Expected invalid item metadata to fail")

    assert recording.get_item_metadata(0) == {}
    assert recording.integrity == integrity


def test_item_metadata_slice_setter_validates_selection(tmp_path) -> None:
    """Slice updates should require contiguous, correctly sized values.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If an unsupported slice is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(np.zeros((4, 2, 4), dtype=np.float32))

    try:
        recording.set_item_metadata_slice(slice(None, None, 2), [{}, {}])
    except ValueError as exc:
        assert "step of 1" in str(exc)
    else:
        raise AssertionError("Expected a strided metadata slice to fail")

    try:
        recording.set_item_metadata_slice(slice(1, 3), [{}])
    except ValueError as exc:
        assert "Expected metadata for 2 selected items" in str(exc)
    else:
        raise AssertionError("Expected a short metadata value list to fail")


def test_item_metadata_validation_reads_chunks(tmp_path, monkeypatch) -> None:
    """Full validation should not call the scalar metadata accessor.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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


def test_append_samples_keeps_item_metadata_aligned(tmp_path) -> None:
    """Batched appends should extend item metadata with explicit defaults.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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


def test_item_metadata_rejects_storage_overrides(tmp_path) -> None:
    """Item metadata should not override SigMF-Zarr storage fields.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If a storage override is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(np.zeros((1, 2, 4), dtype=np.float32))

    try:
        recording.set_item_metadata(
            [{"global": {"sigmf-zarr:dtype": "<i2"}}]
        )
    except ValueError as exc:
        assert "cannot override storage fields" in str(exc)
    else:
        raise AssertionError("Expected item storage override to fail")


def test_clear_item_metadata_removes_array(tmp_path) -> None:
    """Clearing item metadata should restore the absent-array state.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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
    tmp_path,
) -> None:
    """CRC32C should reject corrupted format-3 item metadata.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If corrupted item metadata can be read.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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

    try:
        recording.get_item_metadata(0)
    except ValueError as exc:
        assert "checksum do not match" in str(exc)
    else:
        raise AssertionError("Expected corrupted item metadata to fail")


def test_item_metadata_detects_corrupted_zarr_format_2_chunk(
    tmp_path,
) -> None:
    """CRC32C should reject corrupted format-2 item metadata.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If corrupted item metadata can be read.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=2)
    recording = store.recordings.open(
        "rec",
        create=True,
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

    try:
        recording.get_item_metadata(0)
    except RuntimeError as exc:
        assert "crc32c checksum do not match" in str(exc)
    else:
        raise AssertionError("Expected corrupted item metadata to fail")


def test_append_samples_extends_unbatched_time_axis_with_capture(
    tmp_path,
) -> None:
    """append_samples should extend unbatched recordings along time.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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

    reopened = SigMFZarrStore.open(store_path, mode="a").recordings.open("rec")
    assert reopened.sample_shape == (2, 7)
    assert reopened.sample_count == 7


def test_append_samples_rejects_unbatched_non_time_axis_change(
    tmp_path,
) -> None:
    """append_samples should reject unbatched non-time axis changes.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If a non-time axis resize is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(4, 2, 8),
        sample_axes=("channel", "iq", "time"),
    )

    try:
        recording.append_samples(np.zeros((5, 2, 3), dtype=np.float32))
    except ValueError as exc:
        assert "Expected 'channel' axis length 4" in str(exc)
    else:
        raise AssertionError("Expected non-time axis resize to fail")


def test_recording_add_index_records_axis_metadata(tmp_path) -> None:
    """Recording indexes should describe the metadata they materialize.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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


def test_recording_add_index_stores_label_metadata(tmp_path) -> None:
    """Recording indexes should support label lookup metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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
    tmp_path,
) -> None:
    """Recording indexes should match the indexed axis length.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If mismatched index length is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(2, 8),
    )

    try:
        recording.add_index(
            "snr_db",
            np.array([0, 2], dtype=np.int16),
            axis="time",
            field="radioml:snr",
        )
    except ValueError as exc:
        assert "does not match axis 'time' length 8" in str(exc)
    else:
        raise AssertionError("Expected mismatched index length to fail")


def test_recording_open_rejects_invalid_sample_axes(tmp_path) -> None:
    """Opening should reject invalid sample-axis metadata.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If invalid sample-axis metadata is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(2, 16),
    )
    metadata = dict(recording.global_metadata)
    metadata["sigmf-zarr:sample-axes"] = ["item", "time"]
    recording._raw_group.attrs["global"] = metadata

    try:
        store.recordings.open("rec")
    except ValueError as exc:
        assert "reserved axis 'item'" in str(exc)
    else:
        raise AssertionError("Expected invalid sample axes to fail open")


def test_item_metadata_presence_is_derived_from_array(tmp_path) -> None:
    """Item metadata presence should not require a redundant global flag.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=True,
        sample_shape=(2, 4),
    )
    recording.append_samples(
        np.zeros((1, 2, 4), dtype=np.float32),
        item_metadata=[None],
    )
    assert "sigmf-zarr:has-item-metadata" not in recording.global_metadata
    assert store.recordings.open("rec").has_item_metadata is True


def test_recording_open_rejects_mismatched_channel_groups(tmp_path) -> None:
    """Opening should reject channel groups that do not match the axis.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If inconsistent channel groups are accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 2, 4),
        sample_axes=("channel", "iq", "time"),
    )
    del recording._channels_group["1"]

    try:
        store.recordings.open("rec")
    except ValueError as exc:
        assert "do not match expected channels" in str(exc)
    else:
        raise AssertionError("Expected mismatched channel groups to fail")


def test_recording_open_rejects_mismatched_checksum_metadata(
    tmp_path,
) -> None:
    """Opening should reject checksum declarations without matching codecs.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If inconsistent checksum metadata is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(2, 4),
    )
    metadata = dict(recording.global_metadata)
    metadata["sigmf-zarr:checksum"] = None
    recording._raw_group.attrs["global"] = metadata

    try:
        store.recordings.open("rec")
    except ValueError as exc:
        assert "does not match sample codec checksum" in str(exc)
    else:
        raise AssertionError("Expected mismatched checksum metadata to fail")


def test_recording_creation_replaces_supplied_zarr_metadata(
    tmp_path,
) -> None:
    """Creation should replace caller-supplied SigMF-Zarr metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
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


def test_recording_open_rejects_invalid_sample_shape_rank(tmp_path) -> None:
    """Opening should reject sample shapes that disagree with axes.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If invalid sample rank is accepted.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr", overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=(16, 2),
    )
    metadata = dict(recording.global_metadata)
    metadata["sigmf-zarr:sample-shape"] = [16, 2, 1]
    recording._raw_group.attrs["global"] = metadata

    try:
        store.recordings.open("rec")
    except ValueError as exc:
        assert "same length as sample_shape" in str(exc)
    else:
        raise AssertionError("Expected invalid sample shape to fail open")


def test_create_array_uses_utf8_serializer_for_unicode_data() -> None:
    """Unicode arrays should be normalized for UTF-8-backed storage."""

    class FakeStore:
        def __init__(self) -> None:
            """Initialize a fake store."""
            self.created: dict[str, object] | None = None

        @staticmethod
        def default_index_chunks(num_values: int) -> tuple[int]:
            """Choose fake index chunks.

            Args:
                num_values: Number of index values.

            Returns:
                One-dimensional chunk shape.
            """
            return (min(4096, max(1, num_values)),)

        def create_array(
            self,
            group: object,
            name: str,
            data: object,
            *,
            overwrite: bool,
            chunks: tuple[int, ...] | None = None,
            shards: object = None,
            compressors: object = "auto",
            compressor: object = "auto",
            serializer: object = "auto",
        ) -> dict[str, object]:
            """Create a fake array.

            Args:
                group: Parent group placeholder.
                name: Array name.
                data: Array data.
                overwrite: Whether replacement is allowed.
                chunks: Optional chunk shape.
                shards: Optional shard shape.
                compressors: Optional compressors configuration.
                compressor: Optional single-compressor configuration.
                serializer: Optional serializer configuration.

            Returns:
                Created array details.
            """
            del group, overwrite, shards, compressors, compressor
            value_array = np.asarray(data)
            if value_array.dtype.kind in {"U", "O"} and serializer == "auto":
                value_array = value_array.astype(str).astype(object)
                serializer = VLenUTF8Codec()
            self.created = {
                "name": name,
                "data": value_array,
                "chunks": chunks
                or self.default_index_chunks(len(value_array)),
                "serializer": serializer,
            }
            return self.created

    recording = object.__new__(SigMFRecording)
    recording._store = FakeStore()

    created = recording._create_array(
        object(),
        "labels",
        np.array(["AM-DSB", "BPSK"]),
        overwrite=False,
    )

    assert isinstance(created["serializer"], VLenUTF8Codec)
    assert np.asarray(created["data"]).dtype.kind == "O"
    np.testing.assert_array_equal(
        np.asarray(created["data"], dtype=object),
        np.array(["AM-DSB", "BPSK"], dtype=object),
    )


def test_create_array_normalizes_object_arrays_to_strings() -> None:
    """Object arrays of strings should be normalized before creation."""

    class FakeStore:
        def __init__(self) -> None:
            """Initialize a fake store."""
            self.created: dict[str, object] | None = None

        @staticmethod
        def default_index_chunks(num_values: int) -> tuple[int]:
            """Choose fake index chunks.

            Args:
                num_values: Number of index values.

            Returns:
                One-dimensional chunk shape.
            """
            return (min(4096, max(1, num_values)),)

        def create_array(
            self,
            group: object,
            name: str,
            data: object,
            *,
            overwrite: bool,
            chunks: tuple[int, ...] | None = None,
            shards: object = None,
            compressors: object = "auto",
            compressor: object = "auto",
            serializer: object = "auto",
        ) -> dict[str, object]:
            """Create a fake array.

            Args:
                group: Parent group placeholder.
                name: Array name.
                data: Array data.
                overwrite: Whether replacement is allowed.
                chunks: Optional chunk shape.
                shards: Optional shard shape.
                compressors: Optional compressors configuration.
                compressor: Optional single-compressor configuration.
                serializer: Optional serializer configuration.

            Returns:
                Created array details.
            """
            del group, overwrite, shards, compressors, compressor
            value_array = np.asarray(data)
            if value_array.dtype.kind in {"U", "O"} and serializer == "auto":
                value_array = value_array.astype(str).astype(object)
                serializer = VLenUTF8Codec()
            self.created = {
                "name": name,
                "data": value_array,
                "chunks": chunks
                or self.default_index_chunks(len(value_array)),
                "serializer": serializer,
            }
            return self.created

    recording = object.__new__(SigMFRecording)
    recording._store = FakeStore()

    created = recording._create_array(
        object(),
        "labels",
        np.array(["AM-DSB", "BPSK"], dtype=object),
        overwrite=False,
    )

    assert isinstance(created["serializer"], VLenUTF8Codec)
    assert np.asarray(created["data"]).dtype.kind == "O"
    np.testing.assert_array_equal(
        np.asarray(created["data"], dtype=object),
        np.array(["AM-DSB", "BPSK"], dtype=object),
    )


def test_add_index_reuses_shared_array_creation_for_strings() -> None:
    """String indexes should use the shared string-normalization path."""

    class FakeGroup:
        def __contains__(self, name: object) -> bool:
            """Return whether a fake member exists.

            Args:
                name: Candidate member name.

            Returns:
                Always false for this fake.
            """
            del name
            return False

        def create_array(
            self,
            name: str,
            **kwargs: object,
        ) -> dict[str, object]:
            """Create a fake array.

            Args:
                name: Array name.
                **kwargs: Array creation options.

            Returns:
                Created array details.
            """
            return {"name": name, **kwargs}

        def require_group(self, name: str) -> FakeGroup:
            """Return this fake group.

            Args:
                name: Required group name.

            Returns:
                This fake group.
            """
            del name
            return self

    store = object.__new__(SigMFZarrStore)
    store._group = type(
        "RootGroup",
        (),
        {
            "metadata": type("Metadata", (), {"zarr_format": 3})(),
            "__getitem__": lambda self, key: FakeGroup(),
        },
    )()

    created = store.add_index("labels", np.array(["AM-DSB", "BPSK"]))

    assert isinstance(created["serializer"], VLenUTF8Codec)
    assert created["shape"] == (2,)
    assert np.asarray(created[slice(None)]).dtype.kind == "O"
    assert created["chunks"] == (2,)
