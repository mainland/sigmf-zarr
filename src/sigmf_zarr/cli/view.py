"""Lazy entry point for the optional desktop dataset viewer."""

from __future__ import annotations

import argparse
from importlib.util import find_spec

from sigmf_zarr.cli.command import Command


class ViewCommand(Command):
    """Launch the optional Qt and Matplotlib dataset viewer."""

    name = "view"
    help = "Browse records, filter items, and visualize samples"

    def __init__(self) -> None:
        """Initialize the desktop viewer command."""
        super().__init__("Open a read-only SigMF-Zarr dataset viewer.")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the store location argument.

        Args:
            parser: Command-line argument parser.
        """
        parser.add_argument("store", help="Store path or URL.")

    def handle(self, args: argparse.Namespace) -> int:
        """Import optional dependencies only when launching the viewer.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Application exit code.

        Raises:
            ValueError: If the optional viewer dependencies are missing.
        """
        if any(
            find_spec(name) is None
            for name in ("PySide6", "matplotlib")
        ):
            raise ValueError(
                "Install the optional viewer with: "
                "pip install 'sigmf-zarr[viewer]'"
            )
        from sigmf_zarr.viewer.qt import launch

        return launch(args.store)
