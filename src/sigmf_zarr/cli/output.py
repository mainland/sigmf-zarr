"""Human-readable and JSON command output helpers."""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal

from sigmf_zarr._text import escape_lines
from sigmf_zarr.cli.command import Command

type OutputFormat = Literal["text", "json"]
"""Supported formats for inspection command results."""


def print_text(text: str) -> None:
    """Print human-readable text with nonprintable characters escaped.

    Text output can contain store names and metadata, so control characters
    must not reach the terminal. JSON output needs no escaping, because the
    encoder escapes every non-ASCII character.

    Args:
        text: Human-readable output. Line feeds separate lines.
    """
    print(escape_lines(text))


class OutputCommand(Command):
    """Base command for results available as text or plain JSON."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the common output-format selector.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument(
            "--format",
            choices=("text", "json"),
            default="text",
            dest="output_format",
            help="Select human-readable text or JSON.",
        )

    def write_output(
        self,
        args: argparse.Namespace,
        *,
        text: str,
        value: Any,
    ) -> None:
        """Write one result in the requested output format.

        Args:
            args: Parsed command-line arguments.
            text: Human-readable result.
            value: JSON-serializable result.
        """
        if args.output_format == "json":
            print(json.dumps(value, indent=2, sort_keys=True))
            return
        print_text(text)


__all__ = ["OutputCommand", "OutputFormat", "print_text"]
