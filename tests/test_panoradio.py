"""Integration and validation tests for Panoradio dataset imports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sigmf_zarr import import_panoradio_dataset
from sigmf_zarr.cli import ImportCommand
from sigmf_zarr.cli.import_panoradio import ImportPanoradioCommand
from sigmf_zarr.cli.sigmf import SigMFCommand
from sigmf_zarr.panoradio import _load_panoradio_samples
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat


def _write_source(
    directory: Path,
    *,
    dtype: str = "<c8",
) -> tuple[Path, Path, npt.NDArray[np.complexfloating]]:
    """Create samples and shuffled tags without external dataset files.

    Args:
        directory: Directory for the NumPy and CSV files.
        dtype: Complex NumPy dtype, including source byte order.

    Returns:
        Source path, tags path, and source samples in original item order.
    """
    components = np.arange(20, dtype=np.float64).reshape(5, 4)
    samples = (
        components + 2.0**-35 + 1j * (-components - 2.0**-36)
    ).astype(dtype)
    source = directory / "Panoradio.small.npy"
    np.save(source, samples)
    tags = directory / "tags.csv"
    tags.write_text(
        "idx, mode, snr\n"
        "3, AM, -32768\n"
        "1, psk31, 0\n"
        "4, USB, 32767\n"
        "0, USB, 12\n"
        "2, AM, -8\n",
        encoding="ascii",
    )
    return source, tags, samples


def _assert_recording(
    recording: SigMFRecording,
    samples: npt.NDArray[np.complexfloating],
) -> None:
    """Check exact I/Q values and aligned source metadata.

    Args:
        recording: Imported recording to inspect.
        samples: Original complex samples in source order.
    """
    expected = np.stack((samples.real, samples.imag), axis=1)
    component_dtype = np.dtype(f"<f{samples.dtype.itemsize // 2}")
    assert recording.samples.shape == (5, 2, 4)
    assert recording.samples.dtype == component_dtype
    assert recording.sample_axes == ("iq", "time")
    np.testing.assert_array_equal(recording.samples[:], expected)
    mode_index = recording.index("mode_id")
    snr_index = recording.index("snr_db")
    np.testing.assert_array_equal(mode_index[:], [1, 2, 0, 0, 1])
    np.testing.assert_array_equal(snr_index[:], [12, 0, -8, -32768, 32767])
    assert mode_index.dtype == np.dtype(np.int32)
    assert mode_index.attrs["axis"] == "item"
    assert mode_index.attrs["field"] == "panoradio:mode"
    assert mode_index.attrs["labels"] == ["AM", "USB", "psk31"]
    assert snr_index.dtype == np.dtype(np.int16)
    assert snr_index.attrs["axis"] == "item"
    assert snr_index.attrs["field"] == "panoradio:snr"
    assert snr_index.attrs["unit"] == "dB"
    assert recording.global_metadata["core:sample_rate"] == 6000
    assert recording.global_metadata["core:datatype"] == (
        "cf32_le" if samples.dtype.itemsize == 8 else "cf64_le"
    )
    assert recording.has_item_metadata is False


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("dtype", ["<c8", ">c8", "<c16", ">c16"])
def test_panoradio_preserves_samples_and_aligns_shuffled_tags(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    dtype: str,
) -> None:
    """Preserve component precision and source order in both Zarr formats.

    Args:
        tmp_path: Pytest temporary directory.
        zarr_format: Physical Zarr format to exercise.
        dtype: Source complex dtype and byte order.
    """
    source, tags, samples = _write_source(tmp_path, dtype=dtype)
    destination = tmp_path / "dataset.zarr"

    import_panoradio_dataset(
        destination,
        source,
        tags,
        batch_size=2,
        zarr_format=zarr_format,
    )

    store = SigMFZarrStore.open(destination)
    assert store.zarr_format == zarr_format
    recording = store.recordings["panoradio"]
    _assert_recording(recording, samples)
    provenance = recording.global_metadata["sigmf-zarr-provenance:import"]
    assert [value["role"] for value in provenance["sources"]] == [
        "samples", "truth"
    ]
    assert provenance["parameters"]["source_rows"] == [0, len(samples)]
    assert recording.global_metadata["panoradio:source_dataset"] == (
        "Panoradio.small"
    )
    assert store.verify_integrity()


def test_panoradio_memory_maps_samples_read_only(tmp_path: Path) -> None:
    """Load source samples through a read-only NumPy memory map.

    Args:
        tmp_path: Pytest temporary directory.
    """
    source, _, expected = _write_source(tmp_path)

    samples = _load_panoradio_samples(source)

    assert isinstance(samples, np.memmap)
    assert not samples.flags.writeable
    np.testing.assert_array_equal(samples, expected)


def test_panoradio_accepts_reordered_tag_columns_and_integral_snr(
    tmp_path: Path,
) -> None:
    """Resolve CSV columns by stripped header names and retain label case.

    Args:
        tmp_path: Pytest temporary directory.
    """
    source, tags, samples = _write_source(tmp_path)
    tags.write_text(
        " snr , idx , mode \n"
        "1.2e1,0,USB\n"
        "0.0,1,psk31\n"
        "-8.000,2,AM\n"
        "-32768,3,AM\n"
        "32767,4,USB\n",
        encoding="ascii",
    )

    store = import_panoradio_dataset(tmp_path / "dataset.zarr", source, tags)

    _assert_recording(store.recordings["panoradio"], samples)


def test_panoradio_applies_source_metadata_without_mutating_caller(
    tmp_path: Path,
) -> None:
    """Apply source-derived fields while preserving other supplied metadata.

    Args:
        tmp_path: Pytest temporary directory.
    """
    source, tags, samples = _write_source(tmp_path, dtype="<c16")
    metadata = {
        "core:sample_rate": 1,
        "core:datatype": "cf32_le",
        "panoradio:source_dataset": "wrong-release",
        "core:description": "Synthetic Panoradio fixture",
    }
    original_metadata = dict(metadata)

    store = import_panoradio_dataset(
        tmp_path / "dataset.zarr",
        source,
        tags,
        source_dataset="explicit-release",
        recording_name="custom",
        global_metadata=metadata,
    )

    recording = store.recordings["custom"]
    _assert_recording(recording, samples)
    assert recording.global_metadata["panoradio:source_dataset"] == (
        "explicit-release"
    )
    assert recording.global_metadata["core:description"] == (
        "Synthetic Panoradio fixture"
    )
    assert metadata == original_metadata


def test_panoradio_bounds_writes_and_applies_storage_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Limit sample conversion batches and honor explicit chunks and shards.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
    """
    source, tags, samples = _write_source(tmp_path, dtype=">c16")
    original_append = SigMFRecording.append_samples
    write_sizes: list[int] = []

    def append(recording: SigMFRecording, batch: npt.ArrayLike) -> None:
        """Inspect each converted batch before a real append.

        Args:
            recording: Destination recording.
            batch: Canonical real sample batch.
        """
        values = np.asarray(batch)
        assert values.dtype == np.dtype("<f8")
        assert values.shape[1:] == (2, 4)
        write_sizes.append(len(values))
        original_append(recording, batch)

    monkeypatch.setattr(SigMFRecording, "append_samples", append)
    store = import_panoradio_dataset(
        tmp_path / "dataset.zarr",
        source,
        tags,
        batch_size=2,
        sample_chunks=(2, 2, 4),
        sample_shards=(4, 2, 4),
        sample_compressor=None,
        zarr_format=3,
    )

    assert write_sizes == [2, 2, 1]
    recording = store.recordings["panoradio"]
    _assert_recording(recording, samples)
    assert recording.samples.chunks == (2, 2, 4)
    assert recording.samples.shards == (4, 2, 4)
    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_panoradio_failed_overwrite_restores_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zarr_format: ZarrFormat,
) -> None:
    """Restore an existing recording after a replacement partially writes.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
        zarr_format: Physical Zarr format to exercise.
    """
    source, tags, samples = _write_source(tmp_path)
    destination = tmp_path / "dataset.zarr"
    store = import_panoradio_dataset(
        destination,
        source,
        tags,
        source_dataset="original-release",
        zarr_format=zarr_format,
    )
    original_metadata = dict(store.recordings["panoradio"].global_metadata)
    original_hash = store.metadata_sha512
    np.save(source, samples + 1000)
    tags.write_text(
        "idx,mode,snr\n" + "".join(f"{idx},NEW,20\n" for idx in range(5)),
        encoding="ascii",
    )
    original_append = SigMFRecording.append_samples
    write_sizes: list[int] = []

    def append_then_fail(
        recording: SigMFRecording,
        batch: npt.ArrayLike,
    ) -> None:
        """Inject failure after the second successful sample append.

        Args:
            recording: Replacement recording.
            batch: Canonical real sample batch.

        Raises:
            RuntimeError: After two batches have been written.
        """
        original_append(recording, batch)
        write_sizes.append(len(np.asarray(batch)))
        if len(write_sizes) == 2:
            raise RuntimeError("Injected failure after sample writes")

    monkeypatch.setattr(SigMFRecording, "append_samples", append_then_fail)
    with pytest.raises(RuntimeError, match="after sample writes"):
        import_panoradio_dataset(
            destination,
            source,
            tags,
            source_dataset="replacement-release",
            overwrite_recording=True,
            batch_size=1,
        )

    assert write_sizes == [1, 1]
    reopened = SigMFZarrStore.open(destination)
    recording = reopened.recordings["panoradio"]
    _assert_recording(recording, samples)
    assert recording.global_metadata == original_metadata
    assert reopened.metadata_sha512 == original_hash
    assert reopened.verify_integrity()


@pytest.mark.parametrize("existing_destination", [False, True])
@pytest.mark.parametrize(
    "candidate",
    [
        np.zeros((5, 4), dtype=np.float32),
        np.zeros((5, 4), dtype=np.int16),
        np.zeros((5, 4), dtype=object),
        np.zeros((5,), dtype=np.complex64),
        np.zeros((5, 2, 4), dtype=np.complex64),
        np.zeros((0, 4), dtype=np.complex64),
        np.zeros((5, 0), dtype=np.complex64),
    ],
    ids=[
        "real", "integer", "object", "one-axis", "three-axes", "no-items",
        "no-samples",
    ],
)
def test_panoradio_rejects_bad_samples_before_destination_changes(
    tmp_path: Path,
    existing_destination: bool,
    candidate: npt.NDArray,
) -> None:
    """Validate sample shape and dtype before an overwrite removes files.

    Args:
        tmp_path: Pytest temporary directory.
        existing_destination: Whether a destination directory already exists.
        candidate: Invalid NumPy sample array.
    """
    source, tags, _ = _write_source(tmp_path)
    np.save(source, candidate)
    destination = tmp_path / "dataset.zarr"
    sentinel = destination / "preserve.txt"
    if existing_destination:
        destination.mkdir()
        sentinel.write_text("Original destination", encoding="ascii")

    with pytest.raises((TypeError, ValueError)):
        import_panoradio_dataset(
            destination, source, tags, overwrite_store=True
        )

    if existing_destination:
        assert list(destination.iterdir()) == [sentinel]
        assert sentinel.read_text(encoding="ascii") == "Original destination"
    else:
        assert not destination.exists()


@pytest.mark.parametrize("failure", ["source-missing", "tags-missing", "npy"])
def test_panoradio_rejects_unreadable_sources_before_destination_changes(
    tmp_path: Path,
    failure: str,
) -> None:
    """Preserve an overwrite destination when required source files fail.

    Args:
        tmp_path: Pytest temporary directory.
        failure: Missing source, missing tags, or invalid NumPy file contents.
    """
    source, tags, _ = _write_source(tmp_path)
    if failure == "source-missing":
        source.unlink()
    elif failure == "tags-missing":
        tags.unlink()
    else:
        source.write_bytes(b"Invalid NumPy file")
    destination = tmp_path / "dataset.zarr"
    destination.mkdir()
    sentinel = destination / "preserve.txt"
    sentinel.write_text("Original destination", encoding="ascii")

    with pytest.raises((OSError, TypeError, ValueError)):
        import_panoradio_dataset(
            destination, source, tags, overwrite_store=True
        )

    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="ascii") == "Original destination"


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "idx,mode,snr\n",
        "idx,mode\n" + "".join(f"{idx},AM\n" for idx in range(5)),
        "idx,mode,snr,extra\n" + "".join(
            f"{idx},AM,0,x\n" for idx in range(5)
        ),
        "idx,mode,mode\n" + "".join(
            f"{idx},AM,AM\n" for idx in range(5)
        ),
        "idx,mode,snr\n0,AM,0\n",
    ],
    ids=[
        "empty", "no-rows", "missing-column", "extra-column",
        "duplicate-column", "incomplete-indexes",
    ],
)
def test_panoradio_rejects_bad_tags_before_destination_changes(
    tmp_path: Path,
    contents: str,
) -> None:
    """Reject malformed tag tables before overwriting an existing path.

    Args:
        tmp_path: Pytest temporary directory.
        contents: CSV contents that violate the tag schema or coverage.
    """
    source, tags, _ = _write_source(tmp_path)
    tags.write_text(contents, encoding="ascii")
    destination = tmp_path / "dataset.zarr"
    destination.mkdir()
    sentinel = destination / "preserve.txt"
    sentinel.write_text("Original destination", encoding="ascii")

    with pytest.raises((TypeError, ValueError)):
        import_panoradio_dataset(
            destination, source, tags, overwrite_store=True
        )

    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="ascii") == "Original destination"


@pytest.mark.parametrize(
    "first_row",
    [
        "1,AM,0",
        "-1,AM,0",
        "5,AM,0",
        "0.5,AM,0",
        "NaN,AM,0",
        "0,,0",
        "0,   ,0",
        "0,AM,",
        "0,AM,1.5",
        "0,AM,1.0000000000000001",
        "0,AM,NaN",
        "0,AM,inf",
        "0,AM,32768",
        "0,AM,-32769",
        "0,AM",
        "0,AM,0,extra",
    ],
    ids=[
        "duplicate-index", "negative-index", "out-of-range-index",
        "fractional-index", "nonnumeric-index", "empty-mode", "blank-mode",
        "empty-snr", "fractional-snr", "rounded-fractional-snr",
        "nan-snr", "infinite-snr",
        "high-snr", "low-snr", "short-row", "long-row",
    ],
)
def test_panoradio_rejects_invalid_tag_values(
    tmp_path: Path,
    first_row: str,
) -> None:
    """Validate each tag value with otherwise complete sample coverage.

    Args:
        tmp_path: Pytest temporary directory.
        first_row: Invalid first row followed by four valid tag rows.
    """
    source, tags, _ = _write_source(tmp_path)
    tags.write_text(
        f"idx,mode,snr\n{first_row}\n" + "".join(
            f"{idx},AM,0\n" for idx in range(1, 5)
        ),
        encoding="ascii",
    )
    destination = tmp_path / "dataset.zarr"
    destination.mkdir()
    sentinel = destination / "preserve.txt"
    sentinel.write_text("Original destination", encoding="ascii")

    with pytest.raises((TypeError, ValueError)):
        import_panoradio_dataset(
            destination, source, tags, overwrite_store=True
        )

    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="ascii") == "Original destination"


@pytest.mark.parametrize("batch_size", [0, -1])
def test_panoradio_rejects_nonpositive_batch_size(
    tmp_path: Path,
    batch_size: int,
) -> None:
    """Reject invalid batch sizes without creating a destination.

    Args:
        tmp_path: Pytest temporary directory.
        batch_size: Invalid maximum number of items per sample write.
    """
    source, tags, _ = _write_source(tmp_path)
    destination = tmp_path / "dataset.zarr"

    with pytest.raises(ValueError, match="batch_size"):
        import_panoradio_dataset(
            destination, source, tags, batch_size=batch_size
        )

    assert not destination.exists()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_panoradio_cli_imports_dataset(
    tmp_path: Path,
    zarr_format: ZarrFormat,
) -> None:
    """Import samples through the nested command and shared options.

    Args:
        tmp_path: Pytest temporary directory.
        zarr_format: Physical Zarr format to exercise.
    """
    source, tags, samples = _write_source(tmp_path)
    destination = tmp_path / "dataset.zarr"

    assert SigMFCommand().run([
        "import",
        "panoradio",
        str(source),
        str(destination),
        "--tags-file", str(tags),
        "--zarr-format", str(zarr_format),
        "--recording-name", "custom",
        "--source-dataset", "cli-release",
        "--batch-size", "2",
        "--sample-compression", "none",
        "--no-sample-sharding",
    ]) == 0

    store = SigMFZarrStore.open(destination)
    assert store.zarr_format == zarr_format
    recording = store.recordings["custom"]
    _assert_recording(recording, samples)
    assert recording.samples.shards is None
    assert recording.global_metadata["panoradio:source_dataset"] == (
        "cli-release"
    )
    assert store.verify_integrity()


def test_panoradio_cli_requires_tags_and_exposes_shared_options() -> None:
    """Require a tags file and expose the shared sample storage controls."""
    command = ImportPanoradioCommand()
    assert isinstance(command, ImportCommand)
    parser = SigMFCommand().build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["import", "panoradio", "source.npy", "store.zarr"])
    assert exc_info.value.code == 2

    arguments = [
        "import", "panoradio", "source.npy", "store.zarr",
        "--tags-file", "tags.csv",
    ]
    defaults = parser.parse_args(arguments)
    assert isinstance(defaults.command, ImportPanoradioCommand)
    assert defaults.handler == defaults.command.handle
    assert defaults.recording_name == "panoradio"
    assert defaults.tags_file == Path("tags.csv")
    assert defaults.batch_size == 4096
    assert defaults.zarr_format is None
    assert defaults.source_dataset is None

    options = parser.parse_args(arguments + [
        "--overwrite-store", "--overwrite-recording",
        "--sample-shard-batch", "32",
        "--sample-compression", "lz4",
        "--sample-compression-level", "4",
    ])
    assert options.overwrite_store
    assert options.overwrite_recording
    assert options.sample_shard_batch == 32
    assert options.sample_compression == "lz4"
    assert options.sample_compression_level == 4
