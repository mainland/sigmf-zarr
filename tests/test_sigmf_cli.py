"""Tests for the unified SigMF CLI."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import numpy.typing as npt
import pytest
from zarr.codecs import BloscCodec

import sigmf_zarr.cli.import_radioml2016 as radioml2016_cli
import sigmf_zarr.cli.import_radioml2018 as radioml2018_cli
import sigmf_zarr.cli.sigmf as sigmf_cli
import sigmf_zarr.cli.store as store_cli
from sigmf_zarr.cli import Command, ImportCommand, positive_int
from sigmf_zarr.cli.command import EscapingFormatter, format_escaped_warning
from sigmf_zarr.store import SigMFZarrStore


def test_command_base_class_is_abstract() -> None:
    """Base command should require a concrete handler."""
    with pytest.raises(TypeError) as excinfo:
        Command("abstract")  # type: ignore[abstract]
    assert "abstract class" in str(excinfo.value)
    assert "handle" in str(excinfo.value)


def test_command_run_configures_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    config = captured["basic_config"]
    assert isinstance(config, dict)
    assert config["level"] == logging.INFO
    (console,) = config["handlers"]
    assert isinstance(console.formatter, EscapingFormatter)
    assert console.formatter._fmt == Command.log_format
    parsed = captured["args"]
    assert isinstance(parsed, argparse.Namespace)
    assert parsed.log is None


def test_command_run_reports_runtime_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Operational RuntimeError failures should become concise CLI errors.

    Args:
        capsys: Pytest output capture fixture.
    """

    class FailingCommand(Command):
        def __init__(self) -> None:
            """Initialize a failing test command."""
            super().__init__("failing")

        def handle(self, args: argparse.Namespace) -> int:
            """Fail as a Zarr format 2 checksum mismatch does.

            Args:
                args: Parsed command arguments.

            Raises:
                RuntimeError: Always.
            """
            del args
            raise RuntimeError("Stored and computed checksum do not match")

    with pytest.raises(SystemExit) as exit_info:
        FailingCommand().run([])
    assert exit_info.value.code == 2
    assert "checksum do not match" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        FailingCommand().run(["--debug"])
    assert exit_info.value.code == 1
    assert isinstance(exit_info.value.__cause__, RuntimeError)
    assert "Traceback" in capsys.readouterr().err


def test_command_run_escapes_untrusted_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Errors, warnings, and log records cannot carry control sequences.

    Args:
        capsys: Pytest output capture fixture.
    """
    hostile = "name\x1b[2K\r\x9b\u202e\udcff"
    escaped = "name\\x1b[2K\\r\\x9b\\u202e\\udcff"

    class HostileCommand(Command):
        def __init__(self) -> None:
            """Initialize a command that fails with untrusted text."""
            super().__init__("hostile")

        def handle(self, args: argparse.Namespace) -> int:
            """Check warning formatting, then fail with untrusted text.

            Args:
                args: Parsed command arguments.

            Raises:
                ValueError: Always.
            """
            del args
            assert warnings.formatwarning is format_escaped_warning
            shown = warnings.formatwarning(hostile, UserWarning, "f.py", 1)
            assert escaped in shown
            raise ValueError(hostile)

    saved = warnings.formatwarning
    with pytest.raises(SystemExit) as exit_info:
        HostileCommand().run([])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert escaped in err
    assert "\x1b" not in err
    assert warnings.formatwarning is saved

    record = logging.LogRecord(
        "x", logging.WARNING, "f.py", 1, hostile, (), None
    )
    assert EscapingFormatter("%(message)s").format(record) == escaped


class RaisingCommand(Command):
    """Command that fails with a given exception."""

    error: BaseException
    """Exception raised by `handle`."""

    def __init__(self, error: BaseException) -> None:
        """Initialize a command that raises one exception.

        Args:
            error: Exception to raise.
        """
        super().__init__("raising")
        self.error = error

    def handle(self, args: argparse.Namespace) -> int:
        """Fail with the configured exception.

        Args:
            args: Parsed command arguments.

        Raises:
            BaseException: Always, the configured exception.
        """
        del args
        raise self.error


def test_untrusted_line_feeds_cannot_forge_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Errors, warnings, and log messages stay on one output line.

    Args:
        capsys: Pytest output capture fixture.
    """
    forged = "omits indexes/snr\nsigmf-zarr: note: export plan verified"
    with pytest.raises(SystemExit) as exit_info:
        RaisingCommand(ValueError(forged)).run([])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "error: omits indexes/snr\\nsigmf-zarr: note:" in err
    assert not any(
        line.startswith("sigmf-zarr: note") for line in err.splitlines()
    )

    record = logging.LogRecord(
        "x", logging.WARNING, "f.py", 1, forged, (), None
    )
    formatted = EscapingFormatter("%(levelname)s:%(message)s").format(record)
    assert formatted.splitlines() == [
        "WARNING:omits indexes/snr\\nsigmf-zarr: note: export plan verified"
    ]
    shown = format_escaped_warning(forged, UserWarning, "f.py", 1, "")
    assert len(shown.splitlines()) == 1


def test_debug_traceback_is_escaped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A debug traceback escapes untrusted text and keeps the exit status.

    Args:
        capsys: Pytest output capture fixture.
    """
    hostile = "key\x1b]8;;http://x\x07\x1b[2J"
    with pytest.raises(SystemExit) as exit_info:
        RaisingCommand(ValueError(hostile)).run(["--debug"])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback (most recent call last):" in err
    assert "ValueError: key\\x1b]8;;http://x\\x07\\x1b[2J" in err
    assert "\x1b" not in err


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (OverflowError("shape too large"), "shape too large"),
        (MemoryError(), "MemoryError"),
        (PermissionError(13, "Permission denied"), "[Errno 13] Permission"),
    ],
)
def test_input_errors_exit_with_status_two(
    capsys: pytest.CaptureFixture[str], error: BaseException, message: str
) -> None:
    """Input failures print one error line and exit with status 2.

    Args:
        capsys: Pytest output capture fixture.
        error: Exception raised by the command.
        message: Expected message text.
    """
    with pytest.raises(SystemExit) as exit_info:
        RaisingCommand(error).run([])
    assert exit_info.value.code == 2
    assert f": error: {message}" in capsys.readouterr().err


def test_http_client_errors_exit_with_status_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An HTTP client error that escapes the store is an input error.

    Args:
        capsys: Pytest output capture fixture.
    """
    aiohttp = pytest.importorskip("aiohttp")
    error = aiohttp.ClientPayloadError("Response payload is not completed")
    with pytest.raises(SystemExit) as exit_info:
        RaisingCommand(error).run([])
    assert exit_info.value.code == 2
    assert "payload is not completed" in capsys.readouterr().err


def test_help_text_has_no_markdown() -> None:
    """Help strings and descriptions do not contain Markdown backticks."""
    texts: list[str] = []

    def collect(parser: argparse.ArgumentParser) -> None:
        """Collect help text from a parser and its subcommands.

        Args:
            parser: Parser to inspect.
        """
        texts.append(parser.description or "")
        for action in parser._actions:
            texts.append(action.help or "")
            if isinstance(action, argparse._SubParsersAction):
                texts.extend(a.help or "" for a in action._choices_actions)
                for child in action.choices.values():
                    collect(child)

    collect(sigmf_cli.SigMFCommand().build_parser())
    assert len(texts) > 50
    assert not [text for text in texts if "`" in text]


def test_command_line_program_name_is_sigmf_zarr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Usage and error messages name the installed command.

    Args:
        capsys: Pytest output capture fixture.
    """
    with pytest.raises(SystemExit) as exit_info:
        sigmf_cli.SigMFCommand().run(["export", "missing.zarr", "out"])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: sigmf-zarr ")
    assert "sigmf-zarr: error: --recording-name is required" in err


def test_store_info_command_supports_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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


def test_import_parsers_accept_zarr_format_2() -> None:
    """All import commands should expose format-2 store creation."""
    sigmf_args = (
        sigmf_cli.ImportSigMFCommand()
        .build_parser()
        .parse_args(["input.sigmf-meta", "store.zarr", "--zarr-format", "2"])
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
    assert sigmf_args.zarr_format == 2
    assert radioml2016_args.zarr_format == 2
    assert radioml2018_args.zarr_format == 2


def test_radioml2016_command_auto_detects_existing_zarr_format_2(
    tmp_path: Path,
) -> None:
    """The 2016 command should not force v3 onto an existing v2 store.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    source_path = tmp_path / "RML2016.10a.pkl"
    store_path = tmp_path / "store-v2.zarr"
    with source_path.open("wb") as handle:
        pickle.dump(
            {("BPSK", 0): np.arange(16, dtype=np.float32).reshape(2, 2, 4)},
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
    tmp_path: Path,
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
            "import",
            "radioml2018",
            str(source_path),
            str(store_path),
            "--batch-size",
            "1",
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

    assert format_action.choices is not None
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
    with pytest.raises(ValueError, match="sigmf-zarr import --help"):
        sigmf_cli.detect_import_format(
            Path("RML2016.10a.pkl"),
            "auto",
            archive=False,
        )


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


def test_radioml2016_command_dispatches_pickle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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

    def fake_load_pickle(
        path: Path, *, encoding: str
    ) -> dict[tuple[str, int], str]:
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

    def fake_as_radioml_dict(
        obj: object,
    ) -> dict[tuple[str, int], npt.NDArray[np.float32]]:
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
        store_path: object,
        dataset: object,
        *,
        source_dataset: str | None = None,
        recording_name: str = "radioml",
        overwrite_store: bool = False,
        overwrite_recording: bool = False,
        batch_size: int = 4096,
        sample_shards: tuple[int, ...] | None = None,
        sample_compressor: object = "auto",
        global_metadata: object = None,
        captures: object = None,
        annotations: object = None,
        iq_chunks: tuple[int, int, int] | None = None,
        zarr_format: int = 3,
    ) -> FakeStore:
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
        sample_compression="zstd",
        sample_compression_level=7,
        zarr_format=3,
    )

    status = radioml2016_cli.ImportRadioML2016Command().handle(args)
    output = capsys.readouterr().out

    assert status == 0
    assert captured["pickle_path"] == Path("RML2016.10a.pkl")
    assert captured["encoding"] == "latin1"
    assert isinstance(captured["dataset"], dict)
    assert set(captured["dataset"]) == {("BPSK", 0)}
    assert captured["store_path"] == "store.zarr"
    assert captured["source_dataset"] == "RML2016.10a"
    assert captured["recording_name"] == "radioml2016"
    assert captured["overwrite_store"] is True
    assert captured["overwrite_recording"] is False
    assert captured["sample_shards"] == (8, 2, 128)
    assert isinstance(captured["sample_compressor"], BloscCodec)
    compressor_name = captured["sample_compressor"].cname
    assert getattr(compressor_name, "value", compressor_name) == "zstd"
    assert captured["sample_compressor"].clevel == 7
    assert captured["zarr_format"] == 3
    assert "store-info" in output


def test_export_command_passes_force_to_single_export(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The export CLI should pass --force to single-recording exports.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.
    """
    captured: dict[str, object] = {}
    fake_store = SimpleNamespace(recordings={"rec"}, collections=set())

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

    monkeypatch.setattr(SigMFZarrStore, "open", fake_open)
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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The export CLI should pass --force to archive exports.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capture fixture.
    """
    captured: dict[str, object] = {}
    fake_store = SimpleNamespace(
        recordings={"rec-a", "rec-b"}, collections={"paired"}
    )

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

    monkeypatch.setattr(SigMFZarrStore, "open", fake_open)
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


def test_resolve_sample_shards_validates_batch_size() -> None:
    """Shard sizing should require a positive sample batch count."""
    with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
        positive_int("0")


def test_load_modulation_classes_supports_json(tmp_path: Path) -> None:
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


def test_load_modulation_classes_detects_json_by_content(
    tmp_path: Path,
) -> None:
    """A JSON array without a `.json` suffix must not load as lines.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    classes_path = tmp_path / "classes.txt"
    classes_path.write_text('[\n"OOK",\n"4ASK"\n]\n', encoding="utf-8")

    assert radioml2018_cli.load_modulation_classes(classes_path) == (
        "OOK",
        "4ASK",
    )

    classes_path.write_text('["OOK",\n"4ASK",\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON RadioML class file"):
        radioml2018_cli.load_modulation_classes(classes_path)


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


def test_load_classes_supports_standard_assignment(
    tmp_path: Path,
) -> None:
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
