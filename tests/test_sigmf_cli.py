"""Tests for the unified SigMF CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import sigmf_zarr.cli.sigmf as sigmf_cli
import sigmf_zarr.cli.store as store_cli
from sigmf_zarr.cli import Command, ImportCommand
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
    assert captured["basic_config"] == {
        "format": Command.log_format,
        "level": logging.INFO,
    }
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
    with pytest.raises(RuntimeError, match="checksum"):
        FailingCommand().run(["--debug"])


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


def test_import_commands_share_import_base() -> None:
    """All concrete importers should inherit shared import behavior."""
    assert issubclass(sigmf_cli.ImportSigMFCommand, ImportCommand)


def test_import_parsers_accept_zarr_format_2() -> None:
    """All import commands should expose format-2 store creation."""
    sigmf_args = (
        sigmf_cli.ImportSigMFCommand()
        .build_parser()
        .parse_args(["input.sigmf-meta", "store.zarr", "--zarr-format", "2"])
    )
    assert sigmf_args.zarr_format == 2


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
            Path("dataset.pkl"),
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
