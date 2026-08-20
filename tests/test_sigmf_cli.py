"""Tests for the unified SigMF CLI."""

from __future__ import annotations

import argparse
import logging

import pytest

from sigmf_zarr.cli import Command


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
