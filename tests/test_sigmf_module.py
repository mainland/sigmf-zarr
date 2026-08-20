"""Tests for the reusable SigMF import/export helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sigmf import SHA512_KEY
from sigmf.sigmffile import SigMFFile

import sigmf_zarr.sigmf as sigmf_module
from tests.testdata import TEST_FLOAT32_DATA, TEST_METADATA


def metadata_without_hash() -> dict[str, object]:
    """Return shared SigMF metadata without the fixed test hash.

    Returns:
        Test metadata with the fixed hash removed.
    """
    global_metadata = dict(TEST_METADATA[SigMFFile.GLOBAL_KEY])
    global_metadata.pop(SHA512_KEY, None)
    return {
        SigMFFile.GLOBAL_KEY: global_metadata,
        SigMFFile.CAPTURE_KEY: list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
        SigMFFile.ANNOTATION_KEY: list(
            TEST_METADATA[SigMFFile.ANNOTATION_KEY]
        ),
    }


def multichannel_metadata(
    *,
    datatype: str,
    num_channels: int,
    sample_count: int,
) -> dict[str, object]:
    """Return SigMF metadata for a multi-channel test recording.

    Args:
        datatype: SigMF datatype string.
        num_channels: Number of interleaved standard SigMF channels.
        sample_count: Number of time samples per channel.

    Returns:
        Standard SigMF metadata object.
    """
    return {
        SigMFFile.GLOBAL_KEY: {
            "core:datatype": datatype,
            "core:num_channels": num_channels,
            "core:version": "1.2.0",
        },
        SigMFFile.CAPTURE_KEY: [{"core:sample_start": 0}],
        SigMFFile.ANNOTATION_KEY: [
            {
                "core:sample_start": 0,
                "core:sample_count": sample_count,
            }
        ],
    }


def write_standard_sigmf(
    meta_path: str | Path,
    samples: np.ndarray,
    metadata: dict[str, object],
) -> None:
    """Create one small standard SigMF recording on disk.

    Args:
        meta_path: Target metadata path.
        samples: Sample array to write.
        metadata: SigMF metadata object.
    """
    meta_file = Path(meta_path)
    data_file = meta_file.with_suffix(".sigmf-data")
    data_file.write_bytes(np.ascontiguousarray(samples).tobytes())

    sigmf_file = SigMFFile(metadata=metadata)
    sigmf_file.set_data_file(data_file=data_file)
    sigmf_file.tofile(str(meta_file), overwrite=True)


@dataclass
class FakeRecording:
    """Simple fake recording used by the module tests."""

    name: str
    batched: bool = False
    global_metadata: dict[str, object] = field(default_factory=dict)
    captures: list[dict[str, object]] = field(default_factory=list)
    annotations: list[dict[str, object]] = field(default_factory=list)
    samples: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    sample_axes: tuple[str, ...] = ("time",)
    sample_sha512: str | None = None
    metadata_sha512: str | None = None
    integrity: dict[str, object] = field(default_factory=dict)

    def set_samples(self, samples: np.ndarray) -> None:
        """Set fake recording samples.

        Args:
            samples: Sample array to store.
        """
        self.samples = np.asarray(samples)

    def set_global_field(self, key: str, value: object) -> None:
        """Set one fake global metadata field.

        Args:
            key: Metadata key to set.
            value: Metadata value to store.
        """
        self.global_metadata[key] = value

    def update_integrity(self) -> dict[str, object]:
        """Pretend to update internal integrity metadata.

        Returns:
            Empty fake integrity object.
        """
        return {}

    def axis_index(self, axis_name: str) -> int:
        """Return the fake runtime axis index.

        Args:
            axis_name: Axis name to locate.

        Returns:
            Runtime axis index.
        """
        return self.sample_axes.index(axis_name)

    @property
    def sample_count(self) -> int:
        """Return the fake time-axis size.

        Returns:
            Number of samples along the fake time axis.
        """
        return int(self.samples.shape[self.axis_index("time")])

    @property
    def num_channels(self) -> int:
        """Return the fake recording's channel count.

        Returns:
            Explicit channel-axis length, or one when it is absent.
        """
        if "channel" not in self.sample_axes:
            return 1
        return int(self.samples.shape[self.axis_index("channel")])


class FakeRecordings:
    """Simple recordings view with create/open behavior."""

    def __init__(
        self, recordings: dict[str, FakeRecording] | None = None
    ) -> None:
        """Initialize a fake recordings view.

        Args:
            recordings: Optional initial recordings.
        """
        self._recordings = {} if recordings is None else recordings

    def open(self, recording_name: str, **kwargs: object) -> FakeRecording:
        """Open or create a fake recording.

        Args:
            recording_name: Recording name.
            **kwargs: Open/create options.

        Returns:
            Fake recording.
        """
        if kwargs.get("create"):
            recording = FakeRecording(
                name=recording_name,
                batched=bool(kwargs.get("batched", False)),
                global_metadata=dict(kwargs.get("global_metadata", {})),
                captures=list(kwargs.get("captures", [])),
                annotations=list(kwargs.get("annotations", [])),
                sample_axes=tuple(kwargs.get("sample_axes", ("time",))),
            )
            self._recordings[recording_name] = recording
            return recording
        return self._recordings[recording_name]


@dataclass
class FakeCollection:
    """Simple fake collection used by archive helpers."""

    name: str
    metadata: dict[str, object]
    recording_ids: tuple[str, ...]
    metadata_sha512: str | None = None
    integrity: dict[str, object] = field(default_factory=dict)

    def update_integrity(self) -> dict[str, object]:
        """Pretend to update internal integrity metadata.

        Returns:
            Empty fake integrity object.
        """
        return {}


class FakeCollections:
    """Simple collections view with create/open behavior."""

    def __init__(
        self, collections: dict[str, FakeCollection] | None = None
    ) -> None:
        """Initialize a fake collections view.

        Args:
            collections: Optional initial collections.
        """
        self._collections = {} if collections is None else collections

    def open(self, collection_name: str, **kwargs: object) -> FakeCollection:
        """Open or create a fake collection.

        Args:
            collection_name: Collection name.
            **kwargs: Open/create options.

        Returns:
            Fake collection.
        """
        if kwargs.get("create"):
            collection = FakeCollection(
                name=collection_name,
                metadata=dict(kwargs.get("metadata", {})),
                recording_ids=tuple(kwargs.get("recording_ids", ())),
            )
            self._collections[collection_name] = collection
            return collection
        return self._collections[collection_name]


@dataclass
class FakeStore:
    """Simple fake store exposing the subset used by the module."""

    recordings: FakeRecordings = field(default_factory=FakeRecordings)
    collections: FakeCollections = field(default_factory=FakeCollections)

    def update_metadata_integrity(self) -> dict[str, object]:
        """Pretend to update store metadata integrity.

        Returns:
            Empty fake integrity object.
        """
        return {}

    def list_recordings(self) -> tuple[str, ...]:
        """Return fake recording names.

        Returns:
            Sorted fake recording names.
        """
        return tuple(sorted(self.recordings._recordings))


class ChunkTrackingSamples:
    """Small Zarr-like sample array that records read slices."""

    shape: tuple[int, ...]
    """Sample array shape."""

    chunks: tuple[int, ...]
    """Sample array chunk shape."""

    reads: list[slice]
    """Sample-axis slices requested by the export path."""

    def __init__(
        self,
        samples: np.ndarray,
        *,
        chunks: tuple[int, ...],
    ) -> None:
        """Initialize a chunk-tracking sample array.

        Args:
            samples: Sample data to expose.
            chunks: Chunk shape to report.
        """
        self._samples = samples
        self.shape = tuple(int(dim) for dim in samples.shape)
        self.chunks = chunks
        self.reads = []

    def __getitem__(self, key: object) -> np.ndarray:
        """Return a sample slice and record the requested sample span.

        Args:
            key: Index or slice requested by the export path.

        Returns:
            Sliced sample array.
        """
        if isinstance(key, slice):
            self.reads.append(key)
        elif isinstance(key, tuple):
            for item in key:
                if isinstance(item, slice) and item != slice(None):
                    self.reads.append(item)
                    break
        return self._samples[key]


def test_coerce_sigmf_samples_rejects_channel_remainder() -> None:
    """Import sample coercion should require whole channel frames.

    Raises:
        AssertionError: If uneven channel data is accepted.
    """
    try:
        sigmf_module.coerce_sigmf_samples(
            np.arange(15, dtype=np.float32),
            num_channels=2,
        )
    except ValueError as exc:
        assert "not divisible by core:num_channels 2" in str(exc)
    else:
        raise AssertionError("Expected uneven channel data to fail")


def test_calculate_and_verify_sha512_stream_sample_chunks() -> None:
    """Whole-recording hashing should use exported bytes and chunked reads."""
    samples = ChunkTrackingSamples(TEST_FLOAT32_DATA.copy(), chunks=(5,))
    recording = FakeRecording(
        name="rec",
        global_metadata=metadata_without_hash()[SigMFFile.GLOBAL_KEY],
        samples=samples,
    )
    expected = hashlib.sha512(TEST_FLOAT32_DATA.tobytes()).hexdigest()

    assert sigmf_module.calculate_sha512(recording) == expected
    assert sigmf_module.verify_sha512(recording) is False

    recording.global_metadata[SHA512_KEY] = expected
    assert sigmf_module.verify_sha512(recording) is True
    assert samples.reads == [
        slice(0, 5),
        slice(5, 10),
        slice(10, 15),
        slice(15, 16),
        slice(0, 5),
        slice(5, 10),
        slice(10, 15),
        slice(15, 16),
    ]


def test_strip_zarr_metadata_preserves_unrelated_extensions() -> None:
    """Stripping should remove only SigMF-Zarr export-internal metadata."""
    global_metadata = {
        "core:datatype": "rf32_le",
        "core:extensions": [
            {
                "name": "sigmf-zarr",
                "version": "0.1.0",
                "optional": False,
            },
            {
                "name": "other-extension",
                "version": "1.0.0",
                "optional": False,
            },
        ],
        "sigmf-zarr:dtype": "<f4",
        "sigmf-zarr:sample-shape": [16],
        "sigmf-zarr:sample-axes": ["time"],
        "other-extension:field": "kept",
    }

    stripped = sigmf_module.strip_zarr_metadata(global_metadata)

    assert stripped == {
        "core:datatype": "rf32_le",
        "core:extensions": [
            {
                "name": "other-extension",
                "version": "1.0.0",
                "optional": False,
            }
        ],
        "other-extension:field": "kept",
    }
    assert "sigmf-zarr:dtype" in global_metadata


def test_sigmf_dtype_to_zarr_for_real_samples() -> None:
    """Real SigMF datatypes should map to scalar Zarr samples."""
    dtype, sample_shape = sigmf_module.sigmf_dtype_to_zarr("rf32_le")

    assert dtype == np.dtype("<f4")
    assert sample_shape == ()


def test_sigmf_dtype_to_zarr_for_complex_samples() -> None:
    """Complex SigMF datatypes should map to an I/Q axis."""
    dtype, sample_shape = sigmf_module.sigmf_dtype_to_zarr("cf32_le")

    assert dtype == np.dtype("<f4")
    assert sample_shape == (2,)


def test_sigmf_dtype_to_zarr_for_complex_float64_samples() -> None:
    """Complex float64 datatypes should retain 64-bit I/Q components."""
    little_dtype, little_shape = sigmf_module.sigmf_dtype_to_zarr("cf64_le")
    big_dtype, big_shape = sigmf_module.sigmf_dtype_to_zarr("cf64_be")

    assert little_dtype == np.dtype("<f8")
    assert big_dtype == np.dtype(">f8")
    assert little_shape == (2,)
    assert big_shape == (2,)
