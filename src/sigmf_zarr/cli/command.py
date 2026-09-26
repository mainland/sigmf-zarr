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


class ImportCommand(Command):
    """Base class for commands that import into a SigMF-Zarr store."""

    source_help: str
    """Help text for the positional source argument."""

    default_recording_name: str | None
    """Default target recording name, or ``None`` for source-derived names."""

    def __init__(
        self,
        description: str,
        *,
        source_help: str,
        default_recording_name: str | None = None,
    ) -> None:
        """Initialize a shared import command.

        Args:
            description: Command-line description.
            source_help: Help text for the positional source argument.
            default_recording_name: Optional default recording name.
        """
        super().__init__(description)
        self.source_help = source_help
        self.default_recording_name = default_recording_name

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments common to every importer.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument("source", type=Path, help=self.source_help)
        parser.add_argument(
            "store",
            help="Target SigMF-Zarr store path or URL.",
        )
        parser.add_argument(
            "--zarr-format",
            type=int,
            choices=(2, 3),
            default=None,
            help=(
                "Physical format for a new target store. Existing stores "
                "are auto-detected. New stores default to Zarr format 3."
            ),
        )
        parser.add_argument(
            "--recording-name",
            default=self.default_recording_name,
            help="Target recording name.",
        )
        parser.add_argument(
            "--overwrite-store",
            action="store_true",
            help="Recreate the target store before importing.",
        )
        parser.add_argument(
            "--overwrite-recording",
            action="store_true",
            help="Replace an existing recording with the same name.",
        )
        self.add_import_arguments(parser)

    def add_import_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add importer-specific arguments.

        Args:
            parser: Argument parser to extend.
        """
        return None


__all__ = [
    "Command",
    "ImportCommand",
    "package_version",
]
