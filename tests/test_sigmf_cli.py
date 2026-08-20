"""Tests for the unified SigMF CLI."""

from __future__ import annotations

import argparse
import logging

from sigmf_zarr.cli import Command


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
