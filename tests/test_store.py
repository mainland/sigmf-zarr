"""Tests for store-level helpers."""

from __future__ import annotations

from numcodecs import CRC32C, Blosc
from zarr.codecs import BloscCodec

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
