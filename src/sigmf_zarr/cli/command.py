"""Base class for explicit command-line commands."""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ClassVar


def package_version() -> str:
    """Return the installed SigMF-Zarr package version.

    Returns:
        Installed package version, or a development fallback when the package
        metadata is unavailable.
    """
    try:
        return version("sigmf-zarr")
    except PackageNotFoundError:
        return "1.0.0a1"


def positive_int(value: str) -> int:
    """Parse a positive integer for ``argparse``.

    Args:
        value: Command-line value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


class Command(ABC):
    """Base class for command-line entrypoints.

    Args:
        description: Command-line description.
    """

    name: ClassVar[str | None] = None
    """Subcommand name when registered under another command."""

    help: ClassVar[str | None] = None
    """Short help text used in subcommand listings."""

    log_format: ClassVar[str] = (
        "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    )
    """Logging format used by command-line entrypoints."""

    description: str
    """Command-line description."""

    def __init__(self, description: str) -> None:
        """Initialize the command.

        Args:
            description: Command-line description.
        """
        self.description = description

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments.

        Args:
            parser: Argument parser to extend.
        """
        return None

    def build_parser(self) -> argparse.ArgumentParser:
        """Build an argument parser for this command.

        Returns:
            Configured argument parser.
        """
        parser = argparse.ArgumentParser(description=self.description)
        self.add_common_arguments(parser)
        self.add_arguments(parser)
        return parser

    def add_common_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add standard arguments shared by top-level commands.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument(
            "-d",
            "--debug",
            action="store_const",
            const=logging.DEBUG,
            dest="loglevel",
            default=logging.WARNING,
            help="Print debugging information.",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_const",
            const=logging.INFO,
            dest="loglevel",
            help="Enable verbose output.",
        )
        parser.add_argument(
            "-l",
            "--log",
            type=Path,
            help="Also write logs to the specified file.",
        )
        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {package_version()}",
        )

    def validate_arguments(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        """Validate parsed arguments before executing the command.

        Args:
            parser: Configured argument parser.
            args: Parsed command-line arguments.
        """
        return None

    @abstractmethod
    def handle(self, args: argparse.Namespace) -> int | None:
        """Run the command.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Optional process exit status.
        """

    def register(
        self,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> argparse.ArgumentParser:
        """Register this command as a subcommand.

        Args:
            subparsers: Subparser collection to extend.

        Returns:
            Configured subcommand parser.

        Raises:
            ValueError: If the command has no subcommand name.
        """
        if self.name is None:
            raise ValueError("Subcommands must define a name")

        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description,
        )
        # Store command objects directly in the namespace so nested groups can
        # dispatch without reconstructing behavior from command names.
        parser.set_defaults(command=self, handler=self.handle)
        self.add_arguments(parser)
        return parser

    def run(self, argv: Sequence[str] | None = None) -> int | None:
        """Parse arguments, configure logging, and run the command.

        Args:
            argv: Optional argument sequence. Defaults to `sys.argv`.

        Returns:
            Optional process exit status.
        """
        parser = self.build_parser()
        args = parser.parse_args(argv)
        self.validate_arguments(parser, args)
        self.configure_logging(parser=parser, args=args)
        try:
            return self.handle(args)
        except (KeyError, OSError, ValueError) as exc:
            # Debug mode preserves tracebacks. Normal command-line use reports
            # concise argparse errors with the same exit behavior as parsing.
            if args.loglevel == logging.DEBUG:
                raise
            parser.error(str(exc))

    def configure_logging(
        self,
        *,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        """Configure console and optional file logging.

        Args:
            parser: Configured argument parser.
            args: Parsed command-line arguments.
        """
        logging.basicConfig(format=self.log_format, level=args.loglevel)
        if args.log is None:
            return

        try:
            handler = logging.FileHandler(args.log, encoding="utf-8")
        except OSError as exc:
            parser.error(f"could not open log file '{args.log}': {exc}")

        handler.setLevel(args.loglevel)
        handler.setFormatter(logging.Formatter(self.log_format))
        logging.getLogger().addHandler(handler)


__all__ = [
    "Command",
    "package_version",
    "positive_int",
]
