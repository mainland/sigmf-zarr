"""Explicit command groups for nested command-line interfaces."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import cast

from sigmf_zarr.cli.command import Command


class CommandGroup(Command):
    """Command that dispatches to an explicit list of child commands.

    Args:
        description: Command-line description.
        commands: Child commands registered under this group.
        destination: Parsed namespace field for the selected child name.
    """

    commands: tuple[Command, ...]
    """Commands registered directly under this group."""

    destination: str
    """Namespace destination used for the selected command name."""

    def __init__(
        self,
        description: str,
        commands: Sequence[Command],
        *,
        destination: str = "command_name",
    ) -> None:
        """Initialize the command group.

        Args:
            description: Command-line description.
            commands: Child commands registered under this group.
            destination: Parsed namespace field for the selected child name.
        """
        super().__init__(description)
        self.commands = tuple(commands)
        self.destination = destination

    def add_group_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments that precede this group's subcommand.

        Args:
            parser: Argument parser to extend.
        """
        return None

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add group arguments and register all explicit children.

        Args:
            parser: Argument parser to extend.
        """
        self.add_group_arguments(parser)
        subparsers = parser.add_subparsers(
            dest=self.destination,
            required=True,
        )
        for command in self.commands:
            command.register(subparsers)

    def handle(self, args: argparse.Namespace) -> int | None:
        """Dispatch to the selected leaf command.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Optional process exit status from the leaf command.
        """
        handler = cast(
            Callable[[argparse.Namespace], int | None],
            args.handler,
        )
        return handler(args)

    def validate_arguments(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        """Run validation supplied by the selected leaf command.

        Args:
            parser: Configured top-level parser.
            args: Parsed command-line arguments.
        """
        # Each nested registration overwrites this default, leaving the leaf
        # command responsible for its own argument validation.
        command = args.command
        if command is not self:
            command.validate_arguments(parser, args)


__all__ = ["CommandGroup"]
