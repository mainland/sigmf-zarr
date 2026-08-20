"""Tests for the reusable SigMF import/export helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sigmf import DATATYPE_KEY, SAMPLE_COUNT_KEY, SAMPLE_START_KEY, SHA512_KEY
from sigmf.sigmffile import SigMFFile, fromfile

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


def test_import_sigmf_reads_standard_recording(tmp_path, monkeypatch) -> None:
    """Import one standard SigMF recording into a testable store API.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    samples = TEST_FLOAT32_DATA.copy()
    meta_path = tmp_path / "source.sigmf-meta"
    store = FakeStore()
    write_standard_sigmf(meta_path, samples, TEST_METADATA)

    monkeypatch.setattr(
        sigmf_module.SigMFZarrStore,
        "create",
        lambda *args, **kwargs: store,
    )

    recording = sigmf_module.import_sigmf(
        tmp_path / "store.zarr",
        meta_path,
        recording_name="rec",
    )

    assert recording.name == "rec"
    np.testing.assert_allclose(
        recording.samples,
        TEST_FLOAT32_DATA,
    )
    assert recording.sample_axes == ("time",)
    assert recording.global_metadata == TEST_METADATA[SigMFFile.GLOBAL_KEY]
    assert recording.captures == TEST_METADATA[SigMFFile.CAPTURE_KEY]
    assert recording.annotations == TEST_METADATA[SigMFFile.ANNOTATION_KEY]


def test_import_sigmf_can_create_zarr_format_2(tmp_path) -> None:
    """Standard SigMF import should support a format-2 target store.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    meta_path = tmp_path / "source.sigmf-meta"
    store_path = tmp_path / "store.zarr"
    write_standard_sigmf(meta_path, TEST_FLOAT32_DATA, TEST_METADATA)

    recording = sigmf_module.import_sigmf(
        store_path,
        meta_path,
        zarr_format=2,
    )

    assert sigmf_module.SigMFZarrStore.open(store_path).zarr_format == 2
    np.testing.assert_allclose(recording.samples[:], TEST_FLOAT32_DATA)


def test_import_sigmf_auto_detects_existing_zarr_format_2(tmp_path) -> None:
    """Standard SigMF import should preserve an existing format-2 store.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    meta_path = tmp_path / "source.sigmf-meta"
    store_path = tmp_path / "store.zarr"
    write_standard_sigmf(meta_path, TEST_FLOAT32_DATA, TEST_METADATA)
    sigmf_module.SigMFZarrStore.create(store_path, zarr_format=2)

    recording = sigmf_module.import_sigmf(store_path, meta_path)

    assert sigmf_module.SigMFZarrStore.open(store_path).zarr_format == 2
    np.testing.assert_allclose(recording.samples[:], TEST_FLOAT32_DATA)


def test_import_sigmf_stores_complex_samples_with_iq_axis(
    tmp_path,
    monkeypatch,
) -> None:
    """Import should store complex SigMF data as leading I/Q samples.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    samples = TEST_FLOAT32_DATA.astype(np.complex64) * (1 + 2j)
    metadata = metadata_without_hash()
    metadata[SigMFFile.GLOBAL_KEY][DATATYPE_KEY] = "cf32_le"
    meta_path = tmp_path / "source.sigmf-meta"
    store = FakeStore()
    write_standard_sigmf(meta_path, samples, metadata)

    monkeypatch.setattr(
        sigmf_module.SigMFZarrStore,
        "create",
        lambda *args, **kwargs: store,
    )

    recording = sigmf_module.import_sigmf(
        tmp_path / "store.zarr",
        meta_path,
        recording_name="rec",
    )

    assert recording.sample_axes == ("iq", "time")
    assert recording.samples.shape == (2, 16)
    np.testing.assert_allclose(recording.samples[0], samples.real)
    np.testing.assert_allclose(recording.samples[1], samples.imag)


def test_cf64_import_export_preserves_dataset_bytes(tmp_path) -> None:
    """Both cf64 byte orders should round-trip without precision loss.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    values = np.array(
        [
            1.0 + 2.0j,
            -3.25 + 4.5j,
            np.pi - np.e * 1j,
            np.nextafter(1.0, 2.0) + np.nextafter(-1.0, -2.0) * 1j,
        ],
        dtype=np.complex128,
    )

    for datatype, dtype in (("cf64_le", "<c16"), ("cf64_be", ">c16")):
        for zarr_format in (2, 3):
            case_path = tmp_path / f"{datatype}-zarr{zarr_format}"
            case_path.mkdir()
            source_meta = case_path / "source.sigmf-meta"
            source_data = source_meta.with_suffix(".sigmf-data")
            store_path = case_path / "store.zarr"
            metadata = multichannel_metadata(
                datatype=datatype,
                num_channels=1,
                sample_count=len(values),
            )
            write_standard_sigmf(
                source_meta,
                values.astype(dtype),
                metadata,
            )
            expected_bytes = source_data.read_bytes()

            recording = sigmf_module.import_sigmf(
                store_path,
                source_meta,
                zarr_format=zarr_format,
            )

            assert np.dtype(recording.samples.dtype).itemsize == 8
            assert recording.sha512 == hashlib.sha512(
                expected_bytes
            ).hexdigest()
            assert recording.verify_sha512() is True
            assert recording.verify_integrity() is True
            assert (
                sigmf_module.SigMFZarrStore.open(
                    store_path
                ).verify_integrity()
                is True
            )
            np.testing.assert_array_equal(recording.samples[0], values.real)
            np.testing.assert_array_equal(recording.samples[1], values.imag)

            exported_meta = sigmf_module.export_sigmf(
                sigmf_module.SigMFZarrStore.open(store_path),
                "source",
                case_path / "exported.sigmf-meta",
            )
            exported_data = exported_meta.with_suffix(".sigmf-data")

            assert exported_data.read_bytes() == expected_bytes
            np.testing.assert_array_equal(
                np.fromfile(exported_data, dtype=dtype),
                values,
            )


def test_import_sigmf_stores_real_multichannel_samples(
    tmp_path,
    monkeypatch,
) -> None:
    """Import should store interleaved real channels as a channel axis.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    samples = np.arange(16, dtype=np.float32).reshape(8, 2)
    metadata = multichannel_metadata(
        datatype="rf32_le",
        num_channels=2,
        sample_count=8,
    )
    meta_path = tmp_path / "source.sigmf-meta"
    store = FakeStore()
    write_standard_sigmf(meta_path, samples, metadata)

    monkeypatch.setattr(
        sigmf_module.SigMFZarrStore,
        "create",
        lambda *args, **kwargs: store,
    )

    recording = sigmf_module.import_sigmf(
        tmp_path / "store.zarr",
        meta_path,
        recording_name="rec",
    )

    assert recording.sample_axes == ("channel", "time")
    assert recording.samples.shape == (2, 8)
    np.testing.assert_allclose(recording.samples[0], samples[:, 0])
    np.testing.assert_allclose(recording.samples[1], samples[:, 1])


def test_import_sigmf_stores_complex_multichannel_samples(
    tmp_path,
    monkeypatch,
) -> None:
    """Import should store interleaved complex channels with IQ axes.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    real = np.arange(16, dtype=np.float32).reshape(8, 2)
    imag = real + 100
    samples = real + 1j * imag
    metadata = multichannel_metadata(
        datatype="cf32_le",
        num_channels=2,
        sample_count=8,
    )
    meta_path = tmp_path / "source.sigmf-meta"
    store = FakeStore()
    write_standard_sigmf(meta_path, samples.astype(np.complex64), metadata)

    monkeypatch.setattr(
        sigmf_module.SigMFZarrStore,
        "create",
        lambda *args, **kwargs: store,
    )

    recording = sigmf_module.import_sigmf(
        tmp_path / "store.zarr",
        meta_path,
        recording_name="rec",
    )

    assert recording.sample_axes == ("channel", "iq", "time")
    assert recording.samples.shape == (2, 2, 8)
    np.testing.assert_allclose(recording.samples[0, 0], real[:, 0])
    np.testing.assert_allclose(recording.samples[0, 1], imag[:, 0])
    np.testing.assert_allclose(recording.samples[1, 0], real[:, 1])
    np.testing.assert_allclose(recording.samples[1, 1], imag[:, 1])

    meta_path = sigmf_module.export_sigmf(
        store,
        "rec",
        tmp_path / "exported",
    )
    exported = fromfile(str(meta_path))

    np.testing.assert_allclose(exported.read_samples(), samples)
    assert exported.get_global_info()["core:num_channels"] == 2


def test_sigmf_num_channels_rejects_invalid_metadata() -> None:
    """Import metadata parsing should reject invalid num_channels.

    Raises:
        AssertionError: If invalid channel metadata is accepted.
    """
    try:
        sigmf_module._sigmf_num_channels({"core:num_channels": True})
    except ValueError as exc:
        assert "core:num_channels" in str(exc)
        assert "must be an integer" in str(exc)
    else:
        raise AssertionError("Expected invalid num_channels to fail")


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


def test_export_sigmf_writes_standard_recording(tmp_path) -> None:
    """Export one recording from the reusable module to SigMF files.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=dict(TEST_METADATA[SigMFFile.GLOBAL_KEY]),
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    exported = fromfile(str(meta_path))

    assert meta_path.name == "exported.sigmf-meta"
    assert meta_path.with_suffix(".sigmf-data").exists()
    np.testing.assert_allclose(
        exported.read_samples(),
        TEST_FLOAT32_DATA,
    )
    assert exported.get_global_info() == TEST_METADATA[SigMFFile.GLOBAL_KEY]


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


def test_export_sigmf_adds_sha512_when_missing(tmp_path) -> None:
    """Standard export should hash the exact bytes it writes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=metadata_without_hash()[
                        SigMFFile.GLOBAL_KEY
                    ],
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(
        store, "rec", tmp_path / "exported"
    )
    data_path = meta_path.with_suffix(".sigmf-data")
    exported = json.loads(meta_path.read_text())

    assert exported[SigMFFile.GLOBAL_KEY][SHA512_KEY] == (
        hashlib.sha512(data_path.read_bytes()).hexdigest()
    )


def test_export_sigmf_rejects_stale_sha512(tmp_path) -> None:
    """Standard export should reject stale whole-recording integrity data.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If export accepts a stale digest.
    """
    global_metadata = metadata_without_hash()[SigMFFile.GLOBAL_KEY]
    global_metadata[SHA512_KEY] = "0" * 128
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )
    meta_path = tmp_path / "exported.sigmf-meta"
    data_path = tmp_path / "exported.sigmf-data"
    meta_path.write_text("previous metadata", encoding="utf-8")
    data_path.write_bytes(b"previous data")

    try:
        sigmf_module.export_sigmf(
            store,
            "rec",
            tmp_path / "exported",
            overwrite=True,
        )
    except ValueError as exc:
        assert "core:sha512 does not match" in str(exc)
    else:
        raise AssertionError("Expected export with stale SHA-512 to fail")

    assert meta_path.read_text(encoding="utf-8") == "previous metadata"
    assert data_path.read_bytes() == b"previous data"


def test_export_sigmf_rejects_stale_internal_sample_hash(tmp_path) -> None:
    """Export should verify a stored internal sample hash when present.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If export accepts a stale internal sample hash.
    """
    store = sigmf_module.SigMFZarrStore.create(
        tmp_path / "store.zarr",
        overwrite=True,
    )
    recording = store.recordings.open(
        "rec",
        create=True,
        batched=False,
        sample_shape=TEST_FLOAT32_DATA.shape,
        sample_axes=("time",),
        global_metadata=metadata_without_hash()[SigMFFile.GLOBAL_KEY],
        captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
        annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
    )
    recording.set_samples(TEST_FLOAT32_DATA)
    store.update_integrity()
    recording._samples_array[0] = np.float32(99.0)

    try:
        sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    except ValueError as exc:
        assert "internal sample SHA-512" in str(exc)
    else:
        raise AssertionError("Expected stale internal sample hash to fail")


def test_export_sigmf_rejects_stale_capture_count(tmp_path) -> None:
    """Export should reject captures that no longer cover the sample data.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=dict(TEST_METADATA[SigMFFile.GLOBAL_KEY]),
                    captures=[
                        {
                            SAMPLE_START_KEY: 0,
                            SAMPLE_COUNT_KEY: 8,
                        }
                    ],
                    annotations=[],
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    try:
        sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    except ValueError as exc:
        assert "Capture metadata describes 8 samples" in str(exc)
        assert "force=True" in str(exc)
    else:
        raise AssertionError("Expected stale capture metadata to fail export")


def test_export_sigmf_force_allows_stale_capture_count(tmp_path) -> None:
    """Forced export should bypass the stale capture guard.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=dict(TEST_METADATA[SigMFFile.GLOBAL_KEY]),
                    captures=[
                        {
                            SAMPLE_START_KEY: 0,
                            SAMPLE_COUNT_KEY: 8,
                        }
                    ],
                    annotations=[],
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(
        store,
        "rec",
        tmp_path / "exported",
        force=True,
    )

    assert meta_path.exists()
    assert meta_path.with_suffix(".sigmf-data").exists()


def test_export_sigmf_rejects_out_of_bounds_annotation(tmp_path) -> None:
    """Export should reject annotations beyond the sample data.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=dict(TEST_METADATA[SigMFFile.GLOBAL_KEY]),
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=[
                        {
                            SAMPLE_START_KEY: 12,
                            SAMPLE_COUNT_KEY: 8,
                        }
                    ],
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    try:
        sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    except ValueError as exc:
        assert "annotation 0 ends at sample 20" in str(exc)
    else:
        raise AssertionError(
            "Expected out-of-bounds annotation to fail export"
        )


def test_export_sigmf_writes_sample_data_by_chunk(tmp_path) -> None:
    """Export should read sample data in sample-axis chunks.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    samples = ChunkTrackingSamples(TEST_FLOAT32_DATA.copy(), chunks=(5,))
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=dict(TEST_METADATA[SigMFFile.GLOBAL_KEY]),
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
                    samples=samples,
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    exported = fromfile(str(meta_path))

    assert samples.reads == [
        slice(0, 5),
        slice(5, 10),
        slice(10, 15),
        slice(15, 16),
    ]
    np.testing.assert_allclose(exported.read_samples(), TEST_FLOAT32_DATA)


def test_export_sigmf_writes_iq_leading_samples_by_time_chunk(
    tmp_path,
) -> None:
    """Export should slice IQ-leading arrays along their time axis.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    real = TEST_FLOAT32_DATA.copy()
    imag = TEST_FLOAT32_DATA[::-1].copy()
    samples = ChunkTrackingSamples(
        np.stack((real, imag), axis=0),
        chunks=(2, 5),
    )
    global_metadata = dict(TEST_METADATA[SigMFFile.GLOBAL_KEY])
    global_metadata.pop(SHA512_KEY, None)
    global_metadata[DATATYPE_KEY] = "cf32_le"
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
                    samples=samples,
                    sample_axes=("iq", "time"),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    exported = fromfile(str(meta_path))

    assert samples.reads == [
        slice(0, 5),
        slice(5, 10),
        slice(10, 15),
        slice(15, 16),
    ]
    np.testing.assert_allclose(exported.read_samples(), real + 1j * imag)


def test_export_sigmf_sets_num_channels_from_channel_axis(
    tmp_path,
) -> None:
    """Export should fill num_channels for multi-channel recordings.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    global_metadata = {
        "core:datatype": "cf32_le",
        "core:version": "1.2.0",
    }
    samples = np.zeros((4, 2, 8), dtype=np.float32)
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    samples=samples,
                    sample_axes=("channel", "iq", "time"),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert metadata["global"]["core:num_channels"] == 4


def test_export_sigmf_preserves_matching_num_channels(tmp_path) -> None:
    """Export should keep matching explicit num_channels metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    global_metadata = {
        "core:datatype": "cf32_le",
        "core:num_channels": 4,
        "core:version": "1.2.0",
    }
    samples = np.zeros((4, 2, 8), dtype=np.float32)
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    samples=samples,
                    sample_axes=("channel", "iq", "time"),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert metadata["global"]["core:num_channels"] == 4


def test_export_sigmf_rejects_mismatched_num_channels(tmp_path) -> None:
    """Export should reject num_channels that disagrees with axes.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If mismatched channel metadata is accepted.
    """
    global_metadata = {
        "core:datatype": "cf32_le",
        "core:num_channels": 2,
        "core:version": "1.2.0",
    }
    samples = np.zeros((4, 2, 8), dtype=np.float32)
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    samples=samples,
                    sample_axes=("channel", "iq", "time"),
                )
            }
        )
    )

    try:
        sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    except ValueError as exc:
        assert "core:num_channels" in str(exc)
        assert "channel axis has length 4" in str(exc)
    else:
        raise AssertionError("Expected mismatched num_channels to fail")


def test_export_sigmf_rejects_invalid_num_channels(tmp_path) -> None:
    """Export should reject non-positive num_channels metadata.

    Args:
        tmp_path: Pytest temporary path fixture.

    Raises:
        AssertionError: If invalid channel metadata is accepted.
    """
    global_metadata = {
        "core:datatype": "rf32_le",
        "core:num_channels": 0,
        "core:version": "1.2.0",
    }
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    try:
        sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    except ValueError as exc:
        assert "core:num_channels" in str(exc)
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid num_channels to fail")


def test_prepare_export_global_omits_implicit_single_channel() -> None:
    """Metadata preparation should not add implicit single-channel metadata."""
    global_metadata = {
        "core:datatype": "rf32_le",
        "core:version": "1.2.0",
    }
    recording = FakeRecording(
        name="rec",
        global_metadata=global_metadata,
        samples=TEST_FLOAT32_DATA.copy(),
    )

    metadata = sigmf_module._prepare_export_global_metadata(recording)

    assert "core:num_channels" not in metadata


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


def test_export_sigmf_strips_zarr_metadata(tmp_path) -> None:
    """Classic SigMF export should not include SigMF-Zarr storage fields.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    global_metadata = dict(TEST_METADATA[SigMFFile.GLOBAL_KEY])
    global_metadata.update(
        {
            "core:extensions": [
                {
                    "name": "sigmf-zarr",
                    "version": "0.1.0",
                    "optional": False,
                },
                {
                    "name": "other-extension",
                    "version": "1.0.0",
                    "optional": True,
                },
            ],
            "sigmf-zarr:dtype": "<f4",
            "sigmf-zarr:sample-shape": [16],
            "sigmf-zarr:sample-axes": ["time"],
            "other-extension:field": "kept",
        }
    )
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    exported_global = metadata["global"]

    assert "sigmf-zarr:dtype" not in exported_global
    assert "sigmf-zarr:sample-shape" not in exported_global
    assert "sigmf-zarr:sample-axes" not in exported_global
    assert exported_global["core:extensions"] == [
        {
            "name": "other-extension",
            "version": "1.0.0",
            "optional": True,
        }
    ]
    assert exported_global["other-extension:field"] == "kept"


def test_export_sigmf_removes_empty_extensions_after_stripping(
    tmp_path,
) -> None:
    """The classic export should omit empty extension declarations.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    global_metadata = dict(TEST_METADATA[SigMFFile.GLOBAL_KEY])
    global_metadata.update(
        {
            "core:extensions": [
                {
                    "name": "sigmf-zarr",
                    "version": "0.1.0",
                    "optional": True,
                }
            ],
            "sigmf-zarr:dtype": "<f4",
        }
    )
    store = FakeStore(
        recordings=FakeRecordings(
            {
                "rec": FakeRecording(
                    name="rec",
                    global_metadata=global_metadata,
                    captures=list(TEST_METADATA[SigMFFile.CAPTURE_KEY]),
                    annotations=list(TEST_METADATA[SigMFFile.ANNOTATION_KEY]),
                    samples=TEST_FLOAT32_DATA.copy(),
                )
            }
        )
    )

    meta_path = sigmf_module.export_sigmf(store, "rec", tmp_path / "exported")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert "core:extensions" not in metadata["global"]


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
