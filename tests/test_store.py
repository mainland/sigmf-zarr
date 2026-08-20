"""Tests for store-level helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from numcodecs import CRC32C, Blosc
from zarr.codecs import BloscCodec

from sigmf_zarr.json import JSONValue
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
    assert alias not in store.recordings
    assert alias not in store.collections
    with pytest.raises(KeyError):
        store.recordings.remove(alias)
    assert store.collections["rec"].recording_ids == ("rec",)
    assert store.list_recordings() == ("rec",)


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
    with pytest.raises(ValueError, match="core:extensions"):
        recording.set_global_field("core:extensions", ["invalid"])


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
    """Opening rejects missing or contradictory storage declarations.

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
