"""Tests for the unified SigMF CLI."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import struct
import warnings
from pathlib import Path

import h5py
import numpy as np
from zarr.codecs import BloscCodec

import sigmf_zarr.cli.import_cspb as cspb_cli
import sigmf_zarr.cli.import_radioml2016 as radioml2016_cli
import sigmf_zarr.cli.import_radioml2018 as radioml2018_cli
import sigmf_zarr.cli.sigmf as sigmf_cli
import sigmf_zarr.cli.store as store_cli
import sigmf_zarr.radioml2016 as radioml2016
from sigmf_zarr.cli import Command, ImportCommand, positive_int
from sigmf_zarr.cli.import_command import (
    compression_level,
    resolve_sample_compressor,
    resolve_sample_shards,
)
from sigmf_zarr.store import SigMFZarrStore


def test_command_base_class_is_abstract() -> None:
    """Base command should require a concrete handler.

    Raises:
        AssertionError: If the base class can be instantiated.
    """
    try:
        Command("abstract")
    except TypeError as exc:
        assert "abstract class" in str(exc)
        assert "handle" in str(exc)
    else:
        raise AssertionError("Expected Command to be abstract")


def test_command_run_configures_logging(monkeypatch) -> None:
    """Command.run should configure logging before handling arguments.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    captured: dict[str, object] = {}

    class ConcreteCommand(Command):
        def __init__(self) -> None:
            """Initialize a concrete test command."""
            super().__init__("concrete")

        def handle(self, args: argparse.Namespace) -> int:
            """Record parsed arguments.

            Args:
                args: Parsed command arguments.

            Returns:
                Process exit status.
            """
            captured["args"] = args
            return 7

    def fake_basic_config(**kwargs: object) -> None:
        """Capture logging configuration kwargs.

        Args:
            **kwargs: Logging configuration keyword arguments.
        """
        captured["basic_config"] = kwargs

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    status = ConcreteCommand().run(["--verbose"])

    assert status == 7
    assert captured["basic_config"] == {
        "format": Command.log_format,
        "level": logging.INFO,
    }
    assert captured["args"].log is None


def test_store_info_command_supports_json(tmp_path, capsys) -> None:
    """Resource commands should expose simple machine-readable output.

    Args:
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest capture fixture.
    """
    store_path = tmp_path / "empty.sigmf-zarr"
    SigMFZarrStore.create(store_path)

    status = sigmf_cli.SigMFCommand().run(
        ["store", str(store_path), "info", "--format", "json"]
    )
    result = json.loads(capsys.readouterr().out)

    assert status == 0
    assert result["schema_name"] == "sigmf-zarr"
    assert result["zarr_format"] == 3
    assert result["recordings"] == []
    assert result["collections"] == []


def test_sigmf_command_registers_subcommand_classes() -> None:
    """Top-level parser should dispatch to command objects."""
    parser = sigmf_cli.SigMFCommand().build_parser()

    verbose_args = parser.parse_args(
        ["--verbose", "import", "sigmf", "input.sigmf-meta", "store.zarr"]
    )
    import_args = parser.parse_args(
        ["import", "sigmf", "input.sigmf-meta", "store.zarr"]
    )
    export_args = parser.parse_args(
        ["export", "store.zarr", "output.sigmf-meta"]
    )
    store_args = parser.parse_args(["store", "store.zarr", "info"])

    assert verbose_args.loglevel == logging.INFO
    assert isinstance(import_args.command, sigmf_cli.ImportSigMFCommand)
    assert isinstance(import_args.command, ImportCommand)
    assert import_args.handler == import_args.command.handle
    assert import_args.zarr_format is None
    assert isinstance(export_args.command, sigmf_cli.ExportCommand)
    assert export_args.handler == export_args.command.handle
    assert isinstance(store_args.command, store_cli.StoreInfoCommand)


def test_radioml_importers_have_nested_parsers() -> None:
    """Dispatch RadioML import subcommands with dataset-specific defaults."""
    parser = sigmf_cli.SigMFCommand().build_parser()
    radioml2016_args = parser.parse_args(
        ["import", "radioml2016", "input.pkl", "store.zarr"]
    )
    radioml2018_args = parser.parse_args(
        ["import", "radioml2018", "input.hdf5", "store.zarr"]
    )

    assert isinstance(
        radioml2016_args.command, radioml2016_cli.ImportRadioML2016Command
    )
    assert radioml2016_args.handler == radioml2016_args.command.handle
    assert radioml2016_args.source == Path("input.pkl")
    assert radioml2016_args.recording_name == "radioml2016"
    assert radioml2016_args.zarr_format is None
    assert isinstance(
        radioml2018_args.command, radioml2018_cli.ImportRadioML2018Command
    )
    assert radioml2018_args.handler == radioml2018_args.command.handle
    assert radioml2018_args.source == Path("input.hdf5")
    assert radioml2018_args.recording_name == "radioml2018"
    assert radioml2018_args.zarr_format is None


def test_cspb_importer_has_nested_parser() -> None:
    """The CSPB importer should expose shared and dataset-specific options."""
    args = sigmf_cli.SigMFCommand().build_parser().parse_args(
        [
            "import",
            "cspb",
            "batches",
            "store.zarr",
            "--truth-file",
            "truth-a.txt",
            "--truth-file",
            "truth-b.txt",
        ]
    )

    assert isinstance(args.command, cspb_cli.ImportCSPBCommand)
    assert args.handler == args.command.handle
    assert args.source == Path("batches")
    assert args.recording_name == "cspb"
    assert args.zarr_format is None
    assert args.truth_file == [Path("truth-a.txt"), Path("truth-b.txt")]


def test_import_commands_share_import_base() -> None:
    """All concrete importers should inherit shared import behavior."""
    assert issubclass(sigmf_cli.ImportSigMFCommand, ImportCommand)
    assert issubclass(
        radioml2016_cli.ImportRadioML2016Command,
        ImportCommand,
    )
    assert issubclass(
        radioml2018_cli.ImportRadioML2018Command,
        ImportCommand,
    )
    assert issubclass(cspb_cli.ImportCSPBCommand, ImportCommand)


def test_import_commands_share_sample_compression_options() -> None:
    """All import parsers should support the same compression codecs."""
    commands_and_sources = (
        (sigmf_cli.ImportSigMFCommand(), "input.sigmf-meta"),
        (radioml2016_cli.ImportRadioML2016Command(), "input.pkl"),
        (radioml2018_cli.ImportRadioML2018Command(), "input.hdf5"),
        (cspb_cli.ImportCSPBCommand(), "batch.zip"),
    )

    for command, source in commands_and_sources:
        args = command.build_parser().parse_args(
            [
                source,
                "store.zarr",
                "--sample-compression",
                "lz4",
                "--sample-compression-level",
                "4",
            ]
        )
        compressor = resolve_sample_compressor(
            args.sample_compression,
            level=args.sample_compression_level,
        )

        assert isinstance(compressor, BloscCodec)
        compressor_name = compressor.cname
        assert getattr(compressor_name, "value", compressor_name) == "lz4"
        assert compressor.clevel == 4


def test_sample_compression_supports_lz4hc_and_auto() -> None:
    """Compression resolution should support LZ4HC and native defaults."""
    compressor = resolve_sample_compressor("lz4hc", level=6)

    assert isinstance(compressor, BloscCodec)
    compressor_name = compressor.cname
    assert getattr(compressor_name, "value", compressor_name) == "lz4hc"
    assert compressor.clevel == 6
    assert resolve_sample_compressor("auto", level=3) == "auto"


def test_sample_compression_can_be_disabled() -> None:
    """Compression resolution should support uncompressed samples."""
    assert resolve_sample_compressor("none", level=3) is None


def test_sample_compression_rejects_unknown_codec() -> None:
    """Compression resolution should reject unsupported codec names.

    Raises:
        AssertionError: If an unsupported codec name is accepted.
    """
    try:
        resolve_sample_compressor("unsupported", level=3)
    except ValueError as exc:
        assert "Unsupported sample compression codec" in str(exc)
    else:
        raise AssertionError("Expected unsupported codec to be rejected")


def test_compression_level_validates_blosc_range() -> None:
    """Blosc compression levels should be integers from zero through nine.

    Raises:
        AssertionError: If an invalid compression level is accepted.
    """
    assert compression_level("0") == 0
    assert compression_level("9") == 9

    for value in ("invalid", "-1", "10"):
        try:
            compression_level(value)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(
                f"Expected compression level {value!r} to be rejected"
            )


def test_import_parsers_accept_zarr_format_2() -> None:
    """All import commands should expose format-2 store creation."""
    sigmf_args = sigmf_cli.ImportSigMFCommand().build_parser().parse_args(
        ["input.sigmf-meta", "store.zarr", "--zarr-format", "2"]
    )
    radioml2016_args = (
        radioml2016_cli.ImportRadioML2016Command()
        .build_parser()
        .parse_args(["input.pkl", "store.zarr", "--zarr-format", "2"])
    )
    radioml2018_args = (
        radioml2018_cli.ImportRadioML2018Command()
        .build_parser()
        .parse_args(["input.hdf5", "store.zarr", "--zarr-format", "2"])
    )
    cspb_args = cspb_cli.ImportCSPBCommand().build_parser().parse_args(
        ["batch.zip", "store.zarr", "--zarr-format", "2"]
    )

    assert sigmf_args.zarr_format == 2
    assert sigmf_args.no_sample_sharding is False
    assert radioml2016_args.zarr_format == 2
    assert radioml2016_args.no_sample_sharding is False
    assert radioml2018_args.zarr_format == 2
    assert radioml2018_args.no_sample_sharding is False
    assert cspb_args.zarr_format == 2
    assert cspb_args.no_sample_sharding is False


def test_sigmf_import_passes_sample_compression(monkeypatch, capsys) -> None:
    """The standard SigMF importer should pass compression to the API.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.
    """
    captured: dict[str, object] = {}

    class FakeRecording:
        def info(self) -> str:
            """Return fake recording information.

            Returns:
                Recording information.
            """
            return "recording-info"

    def fake_import_sigmf(
        store_path: object,
        metadata_path: Path,
        **kwargs: object,
    ) -> FakeRecording:
        """Capture standard SigMF import arguments.

        Args:
            store_path: Target store path.
            metadata_path: Source metadata path.
            **kwargs: Import options.

        Returns:
            Fake recording.
        """
        captured["store_path"] = store_path
        captured["metadata_path"] = metadata_path
        captured.update(kwargs)
        return FakeRecording()

    monkeypatch.setattr(sigmf_cli, "import_sigmf", fake_import_sigmf)

    status = sigmf_cli.SigMFCommand().run(
        [
            "import",
            "sigmf",
            "input.sigmf-meta",
            "store.zarr",
            "--sample-compression",
            "lz4hc",
            "--sample-compression-level",
            "6",
        ]
    )
    output = capsys.readouterr().out

    compressor = captured["sample_compressor"]
    assert status == 0
    assert isinstance(compressor, BloscCodec)
    compressor_name = compressor.cname
    assert getattr(compressor_name, "value", compressor_name) == "lz4hc"
    assert compressor.clevel == 6
    assert "recording-info" in output


def test_cspb_command_auto_detects_existing_zarr_format_2(
    tmp_path,
) -> None:
    """The CSPB command should preserve an existing format-2 store.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "signal_1.tim"
    store_path = tmp_path / "store-v2.zarr"
    components = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4")
    source_path.write_bytes(
        struct.pack("<ii", 2, 2) + components.tobytes()
    )
    SigMFZarrStore.create(store_path, zarr_format=2)

    status = sigmf_cli.SigMFCommand().run(
        ["import", "cspb", str(source_path), str(store_path)]
    )

    store = SigMFZarrStore.open(store_path)
    assert status == 0
    assert store.zarr_format == 2
    assert store.recordings["cspb"].samples.shape == (1, 2, 2)


def test_radioml2016_command_auto_detects_existing_zarr_format_2(
    tmp_path,
) -> None:
    """The 2016 command should not force v3 onto an existing v2 store.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "RML2016.10a.pkl"
    store_path = tmp_path / "store-v2.zarr"
    with source_path.open("wb") as handle:
        pickle.dump(
            {
                ("BPSK", 0): np.arange(16, dtype=np.float32).reshape(
                    2, 2, 4
                )
            },
            handle,
        )
    SigMFZarrStore.create(store_path, zarr_format=2)

    status = sigmf_cli.SigMFCommand().run(
        ["import", "radioml2016", str(source_path), str(store_path)]
    )

    store = SigMFZarrStore.open(store_path)
    assert status == 0
    assert store.zarr_format == 2
    assert store.recordings["radioml2016"].samples.shape == (2, 2, 4)


def test_radioml2018_command_auto_detects_existing_zarr_format_2(
    tmp_path,
) -> None:
    """The 2018 command should not force v3 onto an existing v2 store.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "RML2018.hdf5"
    store_path = tmp_path / "store-v2.zarr"
    samples = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
    labels = np.zeros(
        (2, len(radioml2018_cli.RADIOML2018_MODULATION_CLASSES)),
        dtype=np.float32,
    )
    labels[0, 0] = 1.0
    labels[1, 1] = 1.0
    with h5py.File(source_path, "w") as source:
        source.create_dataset("X", data=samples)
        source.create_dataset("Y", data=labels)
        source.create_dataset("Z", data=np.array([[0], [2]], dtype=np.int16))
    SigMFZarrStore.create(store_path, zarr_format=2)

    status = sigmf_cli.SigMFCommand().run(
        [
            "import", "radioml2018", str(source_path), str(store_path),
            "--batch-size", "1",
        ]
    )

    store = SigMFZarrStore.open(store_path)
    assert status == 0
    assert store.zarr_format == 2
    assert store.recordings["radioml2018"].samples.shape == (2, 2, 4)


def test_import_command_format_choices_match_literal() -> None:
    """The --format parser choices should derive from ImportFormat."""
    parser = sigmf_cli.ImportSigMFCommand().build_parser()
    format_action = next(
        action for action in parser._actions if action.dest == "format"
    )

    assert tuple(format_action.choices) == sigmf_cli.IMPORT_FORMAT_CHOICES


def test_detect_import_format_from_suffixes() -> None:
    """Import format should auto-detect from common path suffixes."""
    assert (
        sigmf_cli.detect_import_format(
            Path("input.sigmf-meta"),
            "auto",
            archive=False,
        )
        == "sigmf"
    )
    assert (
        sigmf_cli.detect_import_format(
            Path("input.sigmf"),
            "auto",
            archive=False,
        )
        == "sigmf-archive"
    )
    try:
        sigmf_cli.detect_import_format(
            Path("RML2016.10a.pkl"),
            "auto",
            archive=False,
        )
    except ValueError as exc:
        assert "sigmf-zarr import --help" in str(exc)
    else:
        raise AssertionError("Expected non-SigMF input to require conversion")


def test_detect_import_format_prefers_explicit_archive_flag() -> None:
    """Archive flag should force SigMF archive handling."""
    assert (
        sigmf_cli.detect_import_format(
            Path("input.anything"),
            "auto",
            archive=True,
        )
        == "sigmf-archive"
    )


def test_radioml2016_command_dispatches_pickle(monkeypatch, capsys) -> None:
    """The explicit 2016 command should route pickles through the shared API.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.
    """
    captured: dict[str, object] = {}

    class FakeStore:
        def info(self) -> str:
            """Return fake store info.

            Returns:
                Store info string.
            """
            return "store-info"

    def fake_load_pickle(path: Path, *, encoding: str):
        """Capture pickle loading arguments.

        Args:
            path: Pickle path.
            encoding: Pickle string encoding.

        Returns:
            Fake pickle payload.
        """
        captured["pickle_path"] = path
        captured["encoding"] = encoding
        return {("BPSK", 0): "raw"}

    def fake_as_radioml_dict(obj: object):
        """Return a fake RadioML mapping.

        Args:
            obj: Raw object to validate.

        Returns:
            Fake RadioML mapping.
        """
        captured["raw_obj"] = obj
        return {
            ("BPSK", 0): np.zeros((2, 2, 128), dtype=np.float32),
        }

    def fake_import_radioml_dataset(
        store_path,
        dataset,
        *,
        source_dataset=None,
        recording_name="radioml",
        overwrite_store=False,
        overwrite_recording=False,
        batch_size=4096,
        sample_shards=None,
        automatic_sharding=True,
        sample_compressor="auto",
        global_metadata=None,
        captures=None,
        annotations=None,
        iq_chunks=None,
        zarr_format=3,
    ):
        """Capture RadioML import arguments.

        Args:
            store_path: Target store path.
            dataset: RadioML dataset.
            source_dataset: Optional source dataset name.
            recording_name: Recording name.
            overwrite_store: Whether to recreate the store.
            overwrite_recording: Whether to replace a recording.
            batch_size: Maximum sample items copied per write.
            sample_shards: Sample shard shape.
            automatic_sharding: Whether automatic sharding is enabled.
            sample_compressor: Sample compressor.
            global_metadata: Optional global metadata.
            captures: Optional capture metadata.
            annotations: Optional annotation metadata.
            iq_chunks: Optional IQ chunk shape.
            zarr_format: Physical Zarr format.

        Returns:
            Fake store.
        """
        captured["store_path"] = store_path
        captured["dataset"] = dataset
        captured["source_dataset"] = source_dataset
        captured["recording_name"] = recording_name
        captured["overwrite_store"] = overwrite_store
        captured["overwrite_recording"] = overwrite_recording
        captured["batch_size"] = batch_size
        captured["sample_shards"] = sample_shards
        captured["automatic_sharding"] = automatic_sharding
        captured["sample_compressor"] = sample_compressor
        captured["global_metadata"] = global_metadata
        captured["captures"] = captures
        captured["annotations"] = annotations
        captured["iq_chunks"] = iq_chunks
        captured["zarr_format"] = zarr_format
        return FakeStore()

    monkeypatch.setattr(
        radioml2016_cli,
        "load_radioml2016_pickle",
        fake_load_pickle,
    )
    monkeypatch.setattr(
        radioml2016_cli,
        "as_radioml2016_dict",
        fake_as_radioml_dict,
    )
    monkeypatch.setattr(
        radioml2016_cli,
        "import_radioml2016_dataset",
        fake_import_radioml_dataset,
    )

    args = argparse.Namespace(
        source=Path("RML2016.10a.pkl"),
        store="store.zarr",
        recording_name="radioml2016",
        overwrite_store=True,
        overwrite_recording=False,
        encoding="latin1",
        batch_size=2,
        source_dataset=None,
        sample_shard_batch=8,
        no_sample_sharding=False,
        sample_compression="zstd",
        sample_compression_level=7,
        zarr_format=3,
    )

    status = radioml2016_cli.ImportRadioML2016Command().handle(args)
    output = capsys.readouterr().out

    assert status == 0
    assert captured["pickle_path"] == Path("RML2016.10a.pkl")
    assert captured["encoding"] == "latin1"
    assert set(captured["dataset"]) == {("BPSK", 0)}
    assert captured["store_path"] == "store.zarr"
    assert captured["source_dataset"] == "RML2016.10a"
    assert captured["recording_name"] == "radioml2016"
    assert captured["overwrite_store"] is True
    assert captured["overwrite_recording"] is False
    assert captured["sample_shards"] == (8, 2, 128)
    assert captured["automatic_sharding"] is True
    assert isinstance(captured["sample_compressor"], BloscCodec)
    compressor_name = captured["sample_compressor"].cname
    assert getattr(compressor_name, "value", compressor_name) == "zstd"
    assert captured["sample_compressor"].clevel == 7
    assert captured["zarr_format"] == 3
    assert "store-info" in output


def test_export_command_passes_force_to_single_export(
    monkeypatch, capsys
) -> None:
    """The export CLI should pass --force to single-recording exports.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.
    """
    captured: dict[str, object] = {}
    fake_store = object()

    def fake_open(path: Path) -> object:
        """Capture the requested store path.

        Args:
            path: Store path to open.

        Returns:
            Fake store object.
        """
        captured["store_path"] = path
        return fake_store

    def fake_export_sigmf(
        store: object,
        recording_name: str,
        output_path: Path,
        *,
        overwrite: bool = False,
        pretty: bool = True,
        force: bool = False,
    ) -> Path:
        """Capture single-recording export arguments.

        Args:
            store: Source store.
            recording_name: Recording name to export.
            output_path: Target output path.
            overwrite: Whether to replace existing files.
            pretty: Whether to pretty-print metadata.
            force: Whether to bypass stale metadata validation.

        Returns:
            Fake metadata path.
        """
        captured["store"] = store
        captured["recording_name"] = recording_name
        captured["output_path"] = output_path
        captured["overwrite"] = overwrite
        captured["pretty"] = pretty
        captured["force"] = force
        return Path("out.sigmf-meta")

    monkeypatch.setattr(sigmf_cli.SigMFZarrStore, "open", fake_open)
    monkeypatch.setattr(sigmf_cli, "export_sigmf", fake_export_sigmf)

    args = argparse.Namespace(
        store=Path("store.zarr"),
        output=Path("out.sigmf-meta"),
        archive=False,
        recording_name="rec",
        recording_names=None,
        collection_name=None,
        overwrite=True,
        compact=True,
        force=True,
    )

    status = sigmf_cli.ExportCommand().handle(args)
    output = capsys.readouterr().out

    assert status == 0
    assert captured["store_path"] == Path("store.zarr")
    assert captured["store"] is fake_store
    assert captured["recording_name"] == "rec"
    assert captured["output_path"] == Path("out.sigmf-meta")
    assert captured["overwrite"] is True
    assert captured["pretty"] is False
    assert captured["force"] is True
    assert "exported recording: out.sigmf-meta" in output


def test_export_command_passes_force_to_archive_export(
    monkeypatch, capsys
) -> None:
    """The export CLI should pass --force to archive exports.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.
    """
    captured: dict[str, object] = {}
    fake_store = object()

    def fake_open(path: Path) -> object:
        """Capture the requested store path.

        Args:
            path: Store path to open.

        Returns:
            Fake store object.
        """
        captured["store_path"] = path
        return fake_store

    def fake_export_sigmf_archive(
        store: object,
        archive_target: Path,
        *,
        recording_names: list[str] | None = None,
        collection_name: str | None = None,
        overwrite: bool = False,
        pretty: bool = True,
        force: bool = False,
    ) -> Path:
        """Capture archive export arguments.

        Args:
            store: Source store.
            archive_target: Target archive path.
            recording_names: Optional recording names.
            collection_name: Optional collection name.
            overwrite: Whether to replace an existing archive.
            pretty: Whether to pretty-print metadata.
            force: Whether to bypass stale metadata validation.

        Returns:
            Fake archive path.
        """
        captured["store"] = store
        captured["archive_target"] = archive_target
        captured["recording_names"] = recording_names
        captured["collection_name"] = collection_name
        captured["overwrite"] = overwrite
        captured["pretty"] = pretty
        captured["force"] = force
        return Path("bundle.sigmf")

    monkeypatch.setattr(sigmf_cli.SigMFZarrStore, "open", fake_open)
    monkeypatch.setattr(
        sigmf_cli,
        "export_sigmf_archive",
        fake_export_sigmf_archive,
    )

    args = argparse.Namespace(
        store=Path("store.zarr"),
        output=Path("bundle.sigmf"),
        archive=True,
        recording_name=None,
        recording_names=["rec-a", "rec-b"],
        collection_name="paired",
        overwrite=True,
        compact=False,
        force=True,
    )

    status = sigmf_cli.ExportCommand().handle(args)
    output = capsys.readouterr().out

    assert status == 0
    assert captured["store_path"] == Path("store.zarr")
    assert captured["store"] is fake_store
    assert captured["archive_target"] == Path("bundle.sigmf")
    assert captured["recording_names"] == ["rec-a", "rec-b"]
    assert captured["collection_name"] == "paired"
    assert captured["overwrite"] is True
    assert captured["pretty"] is True
    assert captured["force"] is True
    assert "exported archive: bundle.sigmf" in output


def test_sample_shard_arguments_validate_and_resolve_batch_size() -> None:
    """Shard sizing should validate and expand a sample batch count.

    Raises:
        AssertionError: If invalid shard sizing does not raise.
    """
    try:
        positive_int("0")
    except argparse.ArgumentTypeError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Expected error for non-positive shard batch")

    assert resolve_sample_shards(None, (2, 128), zarr_format=3) is None
    assert resolve_sample_shards(16, (2, 128), zarr_format=3) == (16, 2, 128)
    try:
        resolve_sample_shards(16, (2, 128), zarr_format=2)
    except ValueError as exc:
        assert "requires Zarr format 3" in str(exc)
    else:
        raise AssertionError("Expected format-2 shard sizing to be rejected")


def test_load_modulation_classes_supports_json(tmp_path) -> None:
    """The 2018 converter should accept an explicit class-order file.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    classes_path = tmp_path / "classes-fixed.json"
    classes_path.write_text('["BPSK", "QPSK"]', encoding="utf-8")

    assert radioml2018_cli.load_modulation_classes(classes_path) == (
        "BPSK",
        "QPSK",
    )


def test_radioml2018_default_uses_corrected_class_order() -> None:
    """The default should match the corrected classes-fixed.json mapping."""
    assert radioml2018_cli.RADIOML2018_MODULATION_CLASSES == (
        "OOK",
        "4ASK",
        "8ASK",
        "BPSK",
        "QPSK",
        "8PSK",
        "16PSK",
        "32PSK",
        "16APSK",
        "32APSK",
        "64APSK",
        "128APSK",
        "16QAM",
        "32QAM",
        "64QAM",
        "128QAM",
        "256QAM",
        "AM-SSB-WC",
        "AM-SSB-SC",
        "AM-DSB-WC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
    )


def test_load_classes_supports_standard_assignment(tmp_path) -> None:
    """The 2018 importer should parse the distributed classes file safely.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    classes_path = tmp_path / "classes.txt"
    classes_path.write_text(
        "classes = ['32PSK',\n '16APSK',\n '32QAM',\n 'FM']\n",
        encoding="utf-8",
    )

    assert radioml2018_cli.load_modulation_classes(classes_path) == (
        "32PSK",
        "16APSK",
        "32QAM",
        "FM",
    )


def test_load_pickle_suppresses_known_numpy_pickle_warning(
    monkeypatch, tmp_path
) -> None:
    """Loading legacy pickles should suppress the NumPy align warning.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary path fixture.
    """
    pickle_path = tmp_path / "dataset.pkl"
    pickle_path.write_bytes(b"placeholder")

    def fake_pickle_load(handle, *, encoding: str):
        """Emit the known warning while loading.

        Args:
            handle: Open pickle file handle.
            encoding: Pickle string encoding.

        Returns:
            Fake decoded payload.
        """
        del handle, encoding
        warnings.warn(
            (
                "dtype(): align should be passed as Python or NumPy "
                "boolean but got `align=0`."
            ),
            np.exceptions.VisibleDeprecationWarning,
            stacklevel=1,
        )
        return {"ok": True}

    monkeypatch.setattr(radioml2016.pickle, "load", fake_pickle_load)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = radioml2016_cli.load_radioml2016_pickle(
            pickle_path,
            encoding="latin1",
        )

    assert result == {"ok": True}
    assert caught == []
