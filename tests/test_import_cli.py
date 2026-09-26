"""Tests for nested importer selection and argument validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import sigmf_zarr.cli.sigmf as sigmf_cli


def test_import_group_and_sigmf_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose importer selection and SigMF options at separate levels.

    Args:
        capsys: Pytest output capture fixture.
    """
    command = sigmf_cli.SigMFCommand()
    with pytest.raises(SystemExit) as exc_info:
        command.run(["import", "--help"])
    assert exc_info.value.code == 0
    group_help = capsys.readouterr().out
    assert "sigmf" in group_help
    assert "--recording-name" not in group_help

    with pytest.raises(SystemExit) as exc_info:
        command.run(["import", "sigmf", "--help"])
    assert exc_info.value.code == 0
    leaf_help = capsys.readouterr().out
    assert " import sigmf " in leaf_help
    assert "--recording-name" in leaf_help
    assert "--archive" in leaf_help
    assert "--format" in leaf_help


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["import"], "required"),
        (["import", "unknown"], "invalid choice"),
        (["import", "input.sigmf-meta", "store.zarr"], "invalid choice"),
    ],
)
def test_import_group_requires_known_importer(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject missing importers, unknown importers, and the bare source form.

    Args:
        arguments: Invalid command-line arguments.
        message: Expected parser diagnostic.
        capsys: Pytest output capture fixture.
    """
    with pytest.raises(SystemExit) as exc_info:
        sigmf_cli.SigMFCommand().run(arguments)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_import_group_runs_leaf_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run leaf validation when an importer is selected through both groups.

    Args:
        monkeypatch: Pytest patch fixture.
        capsys: Pytest output capture fixture.
    """

    def reject_import(
        self: sigmf_cli.ImportSigMFCommand,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        """Reject parsed arguments before opening the source or store.

        Args:
            self: Selected importer.
            parser: Root command parser.
            args: Parsed leaf arguments.
        """
        assert args.command is self
        assert args.source == Path("input.sigmf-meta")
        parser.error("leaf validation failed")

    monkeypatch.setattr(
        sigmf_cli.ImportSigMFCommand, "validate_arguments", reject_import
    )
    with pytest.raises(SystemExit) as exc_info:
        sigmf_cli.SigMFCommand().run(
            ["import", "sigmf", "input.sigmf-meta", "store.zarr"]
        )
    assert exc_info.value.code == 2
    assert "leaf validation failed" in capsys.readouterr().err
