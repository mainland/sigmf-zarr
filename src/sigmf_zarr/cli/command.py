"""Base class for explicit command-line commands."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, NoReturn

from sigmf_zarr._text import escape_lines, escape_text
from sigmf_zarr._version import __version__


def package_version() -> str:
    """Return the installed SigMF-Zarr package version.

    Returns:
        Installed package version, or `0+unknown` when the package metadata
        is unavailable.
    """
    return __version__


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


class CommandParser(argparse.ArgumentParser):
    """Argument parser whose errors name the program, not the subcommand.

    A subcommand parser prints its own usage line, so the line shows the
    options that apply. The message prefix names the program alone, as
    `sigmf-zarr: error:`, for every subcommand.
    """

    def error(self, message: str) -> NoReturn:
        """Print the usage line and message, then exit with status 2.

        Args:
            message: Error message.

        Raises:
            SystemExit: Always, with status 2.
        """
        self.print_usage(sys.stderr)
        program = self.prog.split(" ", 1)[0]
        self.exit(2, f"{program}: error: {message}\n")


class NotFoundError(KeyError):
    """A named resource that a command selects does not exist."""

    kind: str
    """Resource kind, such as `recording` or `collection`."""

    name: str
    """Requested resource name."""

    def __init__(self, kind: str, name: str) -> None:
        """Record the missing resource.

        Args:
            kind: Resource kind.
            name: Requested resource name.
        """
        super().__init__(name)
        self.kind = kind
        self.name = name

    def __str__(self) -> str:
        """Return a message that names the resource kind.

        Returns:
            Message such as `recording not found: 'name'`.
        """
        return f"{self.kind} not found: {self.name!r}"


def error_message(
    exc: BaseException, options: Mapping[str, str] | None = None
) -> str:
    """Return a one-line, escaped command-line message for an exception.

    Library messages name Python parameters, such as `overwrite=True`.
    Each parameter in `options` is replaced by its command-line option. The
    whole message is escaped as one line, because it can quote untrusted
    names that contain line feeds.

    Args:
        exc: Exception to report.
        options: Python parameter names mapped to option names.

    Returns:
        Printable message. An exception with an empty message, such as
        `MemoryError`, is reported by its type name.
    """
    if isinstance(exc, NotFoundError):
        message = str(exc)
    elif isinstance(exc, KeyError) and exc.args:
        # KeyError renders its key with quotes and no context.
        message = f"not found: {exc.args[0]!r}"
    else:
        message = str(exc) or type(exc).__name__
    for parameter, option in (options or {}).items():
        name = re.escape(parameter)
        message = re.sub(rf"\b{name}=True\b", option, message)
        if "_" in parameter:
            # A bare identifier with an underscore is a parameter name.
            # Plain words, such as overwrite, also occur in prose. Quoted
            # names and path segments are store content, not parameters.
            message = re.sub(
                rf"(?<![\w\-/'\"]){name}(?![\w\-/'\"=])", option, message
            )
    return escape_text(message)


def _input_errors() -> tuple[type[BaseException], ...]:
    """Return the exception types that a command reports as input errors.

    Returns:
        Exception types. Backend client errors are included only when the
        backend is already loaded, so this check imports no backend.
    """
    errors: tuple[type[BaseException], ...] = (
        KeyError,
        MemoryError,
        OSError,
        OverflowError,
        RuntimeError,
        ValueError,
    )
    # Remote stores should translate backend errors to OSError. These
    # types cover any that still escape, such as a truncated HTTP body.
    aiohttp = sys.modules.get("aiohttp")
    if aiohttp is not None:
        errors += (aiohttp.ClientError,)
    return errors


class EscapingFormatter(logging.Formatter):
    """Log formatter that escapes nonprintable characters.

    Log messages can quote store names and metadata. Escaping keeps their
    control characters, including line feeds, from reaching the terminal or
    log file. Tracebacks keep their line structure.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Format the message line and escape it as one line.

        Args:
            record: Log record to format.

        Returns:
            Formatted message containing only printable characters.
        """
        return escape_text(super().formatMessage(record))

    def format(self, record: logging.LogRecord) -> str:
        """Format one record and escape the result line by line.

        Args:
            record: Log record to format.

        Returns:
            Formatted record containing only printable characters per line.
        """
        return escape_lines(super().format(record))


_format_warning = warnings.formatwarning
"""Standard warning formatter, saved before any replacement."""


def format_escaped_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """Format a warning like `warnings.formatwarning`, with text escaped.

    Args:
        message: Warning message or instance.
        category: Warning category.
        filename: File that issued the warning.
        lineno: Line that issued the warning.
        line: Optional source line to show.

    Returns:
        Formatted warning containing only printable characters per line.
    """
    # Escape the message as one line before formatting, so its line feeds
    # cannot start lines that look like other output.
    return escape_lines(
        _format_warning(
            escape_text(str(message)), category, filename, lineno, line
        )
    )


class Command(ABC):
    """Base class for command-line entrypoints.

    Args:
        description: Command-line description.
    """

    name: ClassVar[str | None] = None
    """Subcommand name when registered under another command."""

    help: ClassVar[str | None] = None
    """Short help text used in subcommand listings."""

    prog: ClassVar[str | None] = None
    """Program name for top-level parsers, or `None` to use `sys.argv[0]`."""

    log_format: ClassVar[str] = (
        "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    )
    """Logging format used by command-line entrypoints."""

    parameter_options: ClassVar[Mapping[str, str]] = {}
    """Python parameter names in library messages mapped to option names."""

    description: str
    """Command-line description."""

    parser: argparse.ArgumentParser | None = None
    """Parser built or registered for this command, used for its errors."""

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
        parser = CommandParser(prog=self.prog, description=self.description)
        self.parser = parser
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
        self.parser = parser
        self.add_arguments(parser)
        return parser

    def run(self, argv: Sequence[str] | None = None) -> int | None:
        """Parse arguments, configure logging, and run the command.

        Usage errors and invalid input, including a missing or unreadable
        store, exit through the selected command's `parser.error` with
        status 2. Messages, log records, warnings, and debug tracebacks have
        nonprintable characters escaped, because they can quote untrusted
        store content.

        Args:
            argv: Optional argument sequence. Defaults to `sys.argv`.

        Returns:
            Optional process exit status.

        Raises:
            SystemExit: With status 1 after printing the traceback of an
                input error in debug mode.
        """
        parser = self.build_parser()
        args = parser.parse_args(argv)
        self.validate_arguments(parser, args)
        self.configure_logging(parser=parser, args=args)
        # Report errors with the selected subcommand's usage line.
        command: Command = getattr(args, "command", self)
        error_parser = command.parser or parser
        # Replacing the formatter rather than `showwarning` leaves warnings
        # recorded by `warnings.catch_warnings(record=True)` unchanged.
        saved_format_warning = warnings.formatwarning
        warnings.formatwarning = format_escaped_warning
        try:
            return self.handle(args)
        except _input_errors() as exc:
            # Normal command-line use reports concise argparse errors with
            # the same exit behavior as parsing. numcodecs reports Zarr
            # format 2 checksum mismatches as RuntimeError, so data
            # corruption is reported the same way.
            if args.loglevel == logging.DEBUG:
                # The traceback quotes untrusted text. Print it escaped
                # instead of letting the interpreter print it raw.
                sys.stderr.write(escape_lines(traceback.format_exc()))
                raise SystemExit(1) from exc
            error_parser.error(error_message(exc, command.parameter_options))
        finally:
            warnings.formatwarning = saved_format_warning

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
        formatter = EscapingFormatter(self.log_format)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logging.basicConfig(level=args.loglevel, handlers=[console])
        if args.log is None:
            return

        try:
            handler = logging.FileHandler(args.log, encoding="utf-8")
        except OSError as exc:
            parser.error(f"could not open log file '{args.log}': {exc}")

        handler.setLevel(args.loglevel)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)


__all__ = [
    "Command",
    "CommandParser",
    "EscapingFormatter",
    "NotFoundError",
    "error_message",
    "format_escaped_warning",
    "package_version",
    "positive_int",
]
