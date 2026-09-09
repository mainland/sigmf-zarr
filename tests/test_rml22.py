"""Integration and regression tests for RML22 pickle imports."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sigmf_zarr import import_radioml2016_dataset
from sigmf_zarr.cli import ImportCommand
from sigmf_zarr.cli.import_radioml2016 import ImportRadioML2016Command
from sigmf_zarr.cli.import_rml22 import ImportRML22Command
from sigmf_zarr.cli.sigmf import SigMFCommand
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat


@pytest.fixture
def dataset() -> dict[tuple[str, int], npt.NDArray[np.float64]]:
    """Provide unsorted classes and SNRs with unequal bucket sizes.

    Returns:
        Four buckets with distinct sample values and three classes.
    """
    return {
        ("QPSK", 10): np.arange(16, dtype=np.float64).reshape(2, 2, 4) + 200,
        ("BPSK", 8): np.arange(8, dtype=np.float64).reshape(1, 2, 4) + 10,
        ("QPSK", -4): np.arange(24, dtype=np.float64).reshape(3, 2, 4) + 100,
        ("8PSK", 0): np.arange(8, dtype=np.float64).reshape(1, 2, 4) + 50,
    }


def _assert_aligned_recording(
    recording: SigMFRecording,
    dataset: dict[tuple[str, int], npt.NDArray[np.float64]],
) -> None:
    """Check sample order and both per-item metadata indexes.

    Args:
        recording: Imported recording to inspect.
        dataset: Source buckets with known expected ordering.
    """
    expected = np.concatenate(
        [
            dataset[("8PSK", 0)],
            dataset[("BPSK", 8)],
            dataset[("QPSK", -4)],
            dataset[("QPSK", 10)],
        ]
    ).astype(np.float32)
    assert recording.samples.shape == (7, 2, 4)
    assert recording.samples.dtype == np.dtype(np.float32)
    assert recording.sample_axes == ("iq", "time")
    np.testing.assert_array_equal(recording.samples[:], expected)
    class_index = recording.index("mod_class_id")
    snr_index = recording.index("snr_db")
    np.testing.assert_array_equal(class_index[:], [0, 1, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(snr_index[:], [0, 8, -4, -4, -4, 10, 10])
    assert class_index.dtype == np.dtype(np.int16)
    assert snr_index.dtype == np.dtype(np.int16)
    assert class_index.attrs["labels"] == ["8PSK", "BPSK", "QPSK"]
    assert class_index.attrs["axis"] == "item"
    assert class_index.attrs["field"] == "radioml:mod_class"
    assert snr_index.attrs["axis"] == "item"
    assert snr_index.attrs["field"] == "radioml:snr"
    assert snr_index.attrs["unit"] == "dB"
    assert recording.global_metadata["radioml:dataset_version"] == "2022"


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("source_dataset", [None, "RML22-release"])
def test_rml22_cli_imports_pickle(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    dataset: dict[tuple[str, int], npt.NDArray[np.float64]],
    zarr_format: ZarrFormat,
    source_dataset: str | None,
) -> None:
    """Import real pickles with aligned samples and source provenance.

    Args:
        tmp_path: Pytest temporary directory.
        caplog: Pytest log capture fixture.
        dataset: Unsorted input buckets.
        zarr_format: Physical Zarr format to exercise.
        source_dataset: Optional explicit source dataset name.
    """
    source = tmp_path / "RML22.small.pkl"
    source.write_bytes(pickle.dumps(dataset, protocol=2))
    destination = tmp_path / "dataset.zarr"
    arguments = [
        "import",
        "rml22",
        str(source),
        str(destination),
        "--zarr-format",
        str(zarr_format),
        "--batch-size",
        "2",
        "--sample-compression",
        "none",
    ]
    if source_dataset is not None:
        arguments.extend(["--source-dataset", source_dataset])

    assert SigMFCommand().run(arguments) == 0

    store = SigMFZarrStore.open(destination)
    assert store.zarr_format == zarr_format
    recording = store.recordings["rml22"]
    _assert_aligned_recording(recording, dataset)
    assert recording.global_metadata["radioml:source_dataset"] == (
        source_dataset or "RML22.small"
    )
    assert str(source) in caplog.text
    assert "Pickle files can execute code" in caplog.text
    assert store.verify_integrity()


def test_rml22_cli_matches_radioml2016_options() -> None:
    """Expose the shared pickle importer options and RML22 defaults."""
    command = ImportRML22Command()
    assert isinstance(command, ImportCommand)
    defaults = SigMFCommand().build_parser().parse_args(
        ["import", "rml22", "source.pkl", "store.zarr"]
    )
    assert isinstance(defaults.command, ImportRML22Command)
    assert defaults.handler == defaults.command.handle
    assert defaults.recording_name == "rml22"
    assert defaults.encoding == "latin1"
    assert defaults.batch_size == 4096
    assert defaults.zarr_format is None
    assert defaults.source_dataset is None
    arguments = [
        "source.pkl",
        "store.zarr",
        "--recording-name",
        "custom",
        "--overwrite-store",
        "--overwrite-recording",
        "--batch-size",
        "7",
        "--encoding",
        "ASCII",
        "--source-dataset",
        "release",
        "--sample-shard-batch",
        "32",
        "--sample-compression",
        "lz4",
        "--sample-compression-level",
        "4",
        "--zarr-format",
        "3",
    ]
    assert vars(command.build_parser().parse_args(arguments)) == vars(
        ImportRadioML2016Command().build_parser().parse_args(arguments)
    )


def test_rml22_bounds_writes_and_applies_storage_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: dict[tuple[str, int], npt.NDArray[np.float64]],
) -> None:
    """Bound conversion writes while honoring explicit chunks and shards.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
        dataset: Unsorted input buckets.
    """
    write_sizes: list[int] = []
    original_append = SigMFRecording.append_samples

    def append(recording: SigMFRecording, samples: npt.ArrayLike) -> None:
        """Record write sizes before appending to the real sample array.

        Args:
            recording: Destination recording.
            samples: Converted sample batch.
        """
        batch = np.asarray(samples)
        assert batch.dtype == np.dtype(np.float32)
        write_sizes.append(len(batch))
        original_append(recording, samples)

    monkeypatch.setattr(SigMFRecording, "append_samples", append)
    store = import_radioml2016_dataset(
        tmp_path / "dataset.zarr",
        dataset,
        dataset_version="2022",
        recording_name="custom",
        batch_size=2,
        iq_chunks=(2, 2, 4),
        sample_shards=(4, 2, 4),
        sample_compressor=None,
    )

    assert write_sizes == [1, 1, 2, 1, 2]
    assert store.zarr_format == 3
    recording = store.recordings["custom"]
    _assert_aligned_recording(recording, dataset)
    assert recording.samples.chunks == (2, 2, 4)
    assert recording.samples.shards == (4, 2, 4)
    assert recording.global_metadata["radioml:source_dataset"] is None
    assert store.verify_integrity()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_rml22_failed_overwrite_restores_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: dict[tuple[str, int], npt.NDArray[np.float64]],
    zarr_format: ZarrFormat,
) -> None:
    """Restore samples, indexes, metadata, and hashes after partial writes.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patch fixture.
        dataset: Original input buckets.
        zarr_format: Physical Zarr format to exercise.
    """
    destination = tmp_path / "dataset.zarr"
    store = import_radioml2016_dataset(
        destination,
        dataset,
        dataset_version="2022",
        recording_name="rml22",
        source_dataset="original-release",
        zarr_format=zarr_format,
    )
    original_hash = store.metadata_sha512
    original_metadata = dict(store.recordings["rml22"].global_metadata)
    original_append = SigMFRecording.append_samples
    write_sizes: list[int] = []

    def append_then_fail(
        recording: SigMFRecording, samples: npt.ArrayLike
    ) -> None:
        """Fail after writing two batches into the replacement recording.

        Args:
            recording: Destination recording.
            samples: Converted sample batch.

        Raises:
            RuntimeError: After the second successful append.
        """
        original_append(recording, samples)
        write_sizes.append(len(np.asarray(samples)))
        if len(write_sizes) == 2:
            raise RuntimeError("Injected failure after sample writes")

    monkeypatch.setattr(SigMFRecording, "append_samples", append_then_fail)
    with pytest.raises(RuntimeError, match="after sample writes"):
        import_radioml2016_dataset(
            destination,
            {("NEW", 20): np.ones((3, 2, 4), dtype=np.float32)},
            dataset_version="2022",
            recording_name="rml22",
            source_dataset="replacement-release",
            overwrite_recording=True,
            batch_size=1,
        )

    assert write_sizes == [1, 1]
    reopened = SigMFZarrStore.open(destination)
    recording = reopened.recordings["rml22"]
    _assert_aligned_recording(recording, dataset)
    assert recording.global_metadata == original_metadata
    assert reopened.metadata_sha512 == original_hash
    assert reopened.verify_integrity()


@pytest.mark.parametrize("existing_destination", [False, True])
@pytest.mark.parametrize(
    ("candidate", "batch_size", "error", "message"),
    [
        (None, 1, TypeError, "mapping"),
        ({}, 1, TypeError, "non-empty"),
        ({"BPSK": np.zeros((1, 2, 4))}, 1, TypeError, "keys"),
        ({("BPSK", 0): np.zeros((1, 4))}, 1, TypeError, "3D"),
        ({("BPSK", 0): np.zeros((1, 3, 4))}, 1, TypeError, "shape"),
        ({("BPSK", 0): np.zeros((1, 2, 0))}, 1, ValueError, "positive"),
        ({("BPSK", 0): np.zeros((0, 2, 4))}, 1, ValueError, "one item"),
        ({("BPSK", 32768): np.zeros((1, 2, 4))}, 1, ValueError, "int16"),
        (
            {
                ("BPSK", 0): np.zeros((1, 2, 4)),
                ("QPSK", 0): np.zeros((1, 2, 5)),
            },
            1,
            ValueError,
            "same sample shape",
        ),
        ({("BPSK", 0): np.zeros((1, 2, 4))}, 0, ValueError, "batch_size"),
    ],
)
def test_rml22_rejects_invalid_input_before_destination_changes(
    tmp_path: Path,
    existing_destination: bool,
    candidate: object,
    batch_size: int,
    error: type[Exception],
    message: str,
) -> None:
    """Reject invalid input before creating or overwriting a destination.

    Args:
        tmp_path: Pytest temporary directory.
        existing_destination: Whether the destination already contains data.
        candidate: Invalid source object or valid input with bad batch size.
        batch_size: Requested maximum sample items per write.
        error: Expected exception type.
        message: Expected validation error text.
    """
    destination = tmp_path / "dataset.zarr"
    sentinel = destination / "preserve.txt"
    if existing_destination:
        destination.mkdir()
        sentinel.write_text("Original destination", encoding="ascii")

    with pytest.raises(error, match=message):
        import_radioml2016_dataset(
            destination,
            candidate,
            dataset_version="2022",
            recording_name="rml22",
            overwrite_store=True,
            batch_size=batch_size,
        )

    if existing_destination:
        assert list(destination.iterdir()) == [sentinel]
        assert sentinel.read_text(encoding="ascii") == "Original destination"
    else:
        assert not destination.exists()


def test_rml22_keeps_dataset_versions_distinct(
    tmp_path: Path,
    dataset: dict[tuple[str, int], npt.NDArray[np.float64]],
) -> None:
    """Keep 2016 metadata unchanged when adding an RML22 recording.

    Args:
        tmp_path: Pytest temporary directory.
        dataset: Input buckets shared by both importer families.
    """
    destination = tmp_path / "dataset.zarr"
    import_radioml2016_dataset(
        destination,
        dataset,
        source_dataset="RML2016.10a",
    )
    metadata = {
        "core:description": "Synthetic import fixture",
        "radioml:dataset_version": "2016",
    }
    store = import_radioml2016_dataset(
        destination,
        dataset,
        dataset_version="2022",
        recording_name="rml22",
        global_metadata=metadata,
    )

    legacy = store.recordings["radioml2016"]
    assert legacy.global_metadata["radioml:dataset_version"] == "2016"
    assert legacy.global_metadata["radioml:source_dataset"] == "RML2016.10a"
    recording = store.recordings["rml22"]
    _assert_aligned_recording(recording, dataset)
    assert recording.global_metadata["radioml:source_dataset"] is None
    assert recording.global_metadata["core:description"] == (
        "Synthetic import fixture"
    )
    assert metadata["radioml:dataset_version"] == "2016"
    assert store.verify_integrity()
